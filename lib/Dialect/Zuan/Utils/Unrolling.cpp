//===- Unrolling.cpp - Zuan Unrolling Utilities -----------------*- C++ -*-===//
//
// This file implements the unrolling utilities for Zuan operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cassert>

#include "Zuan/IR/Zuan.h"
#include "Zuan/Interfaces/ZuanUnrollingInterface.h"
#include "Zuan/Utils/ShapeInference.h"
#include "Zuan/Utils/Unrolling.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "zuan-unrolling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "] ")

namespace mlir {
namespace zuan {

Value getUnrolledValue(OpBuilder &builder, Value operand, UnrollOptions options,
                       UnrollState &state) {
  if (state.valueMap.contains(operand)) {
    // This is defined above, or fallback to clone.
    return state.valueMap.lookup(operand);
  }
  auto definingOp = operand.getDefiningOp();
  if (!definingOp && isa<BlockArgument>(operand)) {
    // Block argument inside the dynamic op, and the parent block is not cloned.
    return operand;
  }
  auto opResult = dyn_cast<OpResult>(operand);
  assert(opResult && "expected an op result");
  auto newOp = unrollOp(builder, opResult.getOwner(), options, state);
  return newOp->getResult(opResult.getResultNumber());
}

bool isMemrefDefinedInsideDynamicOp(Value value) {
  auto *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return false;
  }
  auto dynamicOp = definingOp->getParentOfType<DynamicOp>();
  return dynamicOp != nullptr;
}

Value getUnrolledMemref(OpBuilder &builder, Value memref, UnrollOptions options,
                        UnrollState &state) {
  if (isMemrefDefinedInsideDynamicOp(memref)) {
    /// Unroll the subview or other memref operations.
    return getUnrolledValue(builder, memref, options, state);
  } else {
    /// Slice the memref itself.
    return createMemrefSlice(builder, memref, options);
  }
}

Value createMemrefSlice(OpBuilder &rewriter, Value memref,
                        UnrollOptions options) {
  assert(!isMemrefDefinedInsideDynamicOp(memref) &&
         "creating slice on a memref defined inside dynamic op is not "
         "well-defined.");

  auto loc = memref.getLoc();

  auto memrefType = cast<MemRefType>(memref.getType());
  auto memrefShape = memrefType.getShape();

  auto rank = memrefShape.size();

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  for (size_t i = 0; i < rank; ++i) {
    if (i == options.getUnrollIdx()) {
      // This memref is not defined inside the dynamic op, so it should be safe
      // to directly reuse the op fold results.
      offsets[i] = options.getOffset();
      sizes.push_back(options.getChunkSize());
    } else if (ShapedType::isDynamic(memrefShape[i])) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(memref);
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value dim = memref::DimOp::create(rewriter, loc, memref, idx);
      sizes.push_back(dim);
    } else {
      sizes.push_back(rewriter.getIndexAttr(memrefShape[i]));
    }
  }

  // Get the unreduced memref type. The offset will be just the same after rank
  // reduction.
  Type unreducedMemrefType =
      memref::SubViewOp::inferResultType(memrefType, offsets, sizes, strides);
  auto [unreducedStrides, unreducedOffset] =
      cast<MemRefType>(unreducedMemrefType).getStridesAndOffset();
  auto unreducedSizes = cast<MemRefType>(unreducedMemrefType).getShape();

  SmallVector<int64_t> resultSizes;
  SmallVector<int64_t> resultStrides;

  // Unroll the requested dim.
  for (size_t i = 0; i < rank; ++i) {
    if (i == options.getUnrollIdx() && options.shouldReduce()) {
      // Drop this dim.
    } else {
      resultSizes.push_back(unreducedSizes[i]);
      resultStrides.push_back(unreducedStrides[i]);
    }
  }
  /// Manually create the layout here because when multiple contiguous unit dim
  /// presents, the strides cannot be correctly inferred by reduced type
  /// inference in subview op
  auto layout = StridedLayoutAttr::get(unreducedMemrefType.getContext(),
                                       unreducedOffset, resultStrides);
  auto resultType =
      MemRefType::get(resultSizes, memrefType.getElementType(), layout);
  Value subview = memref::SubViewOp::create(rewriter, loc, resultType, memref,
                                                     offsets, sizes, strides);
  return subview;
}

static Operation *unrollSubviewOp(OpBuilder &builder,
                                  memref::SubViewOp subviewOp,
                                  UnrollOptions options, UnrollState &state) {
  auto oldSource = subviewOp.getSource();

  auto oldOffsets = subviewOp.getMixedOffsets();
  auto oldSizes = subviewOp.getMixedSizes();
  auto oldStrides = subviewOp.getMixedStrides();

  auto droppedDims = subviewOp.getDroppedDims();
  auto originalIdx =
      computeUnreducedIdx(options.getUnrollIdx(), oldSource.getType().getRank(),
                          [&](unsigned idx) { return droppedDims.test(idx); });

  // Project the dim-to-expand to the original shape.
  UnrollOptions subOptions = options;
  subOptions.overrideUnrollIdx(originalIdx);

  SmallVector<OpFoldResult> offsets, sizes, strides;

  for (size_t i = 0; i < oldOffsets.size(); ++i) {
    if (i == static_cast<size_t>(originalIdx)) {
      // This is the unrolling dim, use the given offset and size.
      offsets.push_back(subOptions.getOffset());
      sizes.push_back(subOptions.getChunkSize());
      strides.push_back(builder.getIndexAttr(1));
    } else {
      // Otherwise, just copy the old offsets, sizes, and strides.
      if (auto offsetValue = dyn_cast<Value>(oldOffsets[i])) {
        auto newValue =
            getUnrolledValue(builder, offsetValue, subOptions, state);
        offsets.push_back(newValue);
      } else {
        offsets.push_back(oldOffsets[i]);
      }

      if (auto sizeValue = dyn_cast<Value>(oldSizes[i])) {
        auto newValue = getUnrolledValue(builder, sizeValue, subOptions, state);
        sizes.push_back(newValue);
      } else {
        sizes.push_back(oldSizes[i]);
      }

      if (auto strideValue = dyn_cast<Value>(oldStrides[i])) {
        auto newValue =
            getUnrolledValue(builder, strideValue, subOptions, state);
        strides.push_back(newValue);
      } else {
        strides.push_back(oldStrides[i]);
      }
    }
  }

  Type unreducedMemrefType = memref::SubViewOp::inferResultType(
      oldSource.getType(), offsets, sizes, strides);
  auto [unreducedStrides, unreducedOffset] =
      cast<MemRefType>(unreducedMemrefType).getStridesAndOffset();
  auto unreducedSizes = cast<MemRefType>(unreducedMemrefType).getShape();

  SmallVector<int64_t> resultSizes;
  SmallVector<int64_t> resultStrides;

  for (size_t i = 0; i < unreducedSizes.size(); ++i) {
    // The rank is not reduced by the original dim, and it is not the
    // dim-to-reduce.
    if (!droppedDims.test(i) &&
        !(subOptions.shouldReduce() && i == originalIdx)) {
      resultSizes.push_back(unreducedSizes[i]);
      resultStrides.push_back(unreducedStrides[i]);
    }
  }

  auto targetType = MemRefType::get(
      resultSizes, cast<MemRefType>(unreducedMemrefType).getElementType(),
      StridedLayoutAttr::get(builder.getContext(), unreducedOffset,
                             resultStrides));
  auto newSubviewOp = memref::SubViewOp::create(builder, 
      subviewOp.getLoc(), cast<MemRefType>(targetType), oldSource, offsets,
      sizes, strides);
  return newSubviewOp;
}

static Operation *unrollSCFForOp(OpBuilder &builder, scf::ForOp forOp,
                                 UnrollOptions options, UnrollState &state) {
  OpBuilder::InsertionGuard guard(builder);

  SmallVector<Value> newInits;
  for (auto init : forOp.getInitArgs()) {
    auto newInit = getUnrolledValue(builder, init, options, state);
    newInits.push_back(newInit);
  }

  auto lb = getUnrolledValue(builder, forOp.getLowerBound(), options, state);
  auto ub = getUnrolledValue(builder, forOp.getUpperBound(), options, state);
  auto step = getUnrolledValue(builder, forOp.getStep(), options, state);

  UnrollState inLoopState = {IRMapping{state.valueMap}, state.yieldBlock};
  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), lb, ub, step, newInits);

  inLoopState.valueMap.map(forOp.getInductionVar(), newForOp.getInductionVar());
  inLoopState.valueMap.map(forOp.getRegionIterArgs(),
                           newForOp.getRegionIterArgs());

  builder.setInsertionPointToStart(newForOp.getBody());
  auto yieldedValues = forOp.getYieldedValues();
  SmallVector<Value> newYieldedValues;
  for (auto value : yieldedValues) {
    auto newYieldedValue =
        getUnrolledValue(builder, value, options, inLoopState);
    newYieldedValues.push_back(newYieldedValue);
  }
  scf::YieldOp::create(builder, forOp.getLoc(), newYieldedValues);
  return newForOp;
}

Operation *unrollOp(OpBuilder &builder, Operation *op, UnrollOptions options,
                    UnrollState &state) {
  assert(op && "expected a non-null operation");

  LLVM_DEBUG(DBGS() << "unrolling: " << op->getName() << "\n");

  if (auto iface = dyn_cast<ZuanUnrollingInterface>(op)) {
    return iface.unroll(builder, options, state);
  }

  if (op->hasTrait<OpTrait::Elementwise>() &&
      op->hasTrait<OpTrait::SameOperandsAndResultType>()) {
    SmallVector<Value> operands;
    for (auto operand : op->getOperands()) {
      operands.push_back(getUnrolledValue(builder, operand, options, state));
    }
    Type commonType = operands.front().getType();
    SmallVector<Type> resultTypes(op->getNumResults(), commonType);
    return builder.create(op->getLoc(), op->getName().getIdentifier(), operands,
                          resultTypes, op->getAttrs());
  }

  if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
    return unrollSubviewOp(builder, subviewOp, options, state);
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return unrollSCFForOp(builder, forOp, options, state);
  }

  if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
    // Should be mapped in the state.
    auto memref = getUnrolledValue(builder, dimOp.getSource(), options, state);
    // Clone it.
    auto index = getUnrolledValue(builder, dimOp.getIndex(), options, state);
    return memref::DimOp::create(builder, op->getLoc(), memref, index);
  }

  if (isa<arith::CmpIOp, arith::CmpFOp>(op)) {
    auto lhs = getUnrolledValue(builder, op->getOperand(0), options, state);
    auto rhs = getUnrolledValue(builder, op->getOperand(1), options, state);
    if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
      return arith::CmpIOp::create(builder, op->getLoc(), cmpi.getPredicate(),
                                           lhs, rhs);
    } else if (auto cmpf = dyn_cast<arith::CmpFOp>(op)) {
      return arith::CmpFOp::create(builder, op->getLoc(), cmpf.getPredicate(),
                                           lhs, rhs);
    } else {
      llvm_unreachable("unexpected comparison operation");
    }
  }

  if (op->getNumSuccessors() == 0 && op->getNumRegions() == 0) {
    // Recursively clone its operands. Assume no regions here.
    SmallVector<Value> operands;
    for (auto operand : op->getOperands()) {
      operands.push_back(getUnrolledValue(builder, operand, options, state));
    }
    return builder.create(op->getLoc(), op->getName().getIdentifier(), operands,
                          op->getResultTypes(), op->getAttrs());
  }

  LLVM_DEBUG(DBGS() << "Fallback to clone: " << op->getName() << "\n");
  return builder.clone(*op, state.valueMap);
}

SmallVector<int64_t> getUnrolledShape(ArrayRef<int64_t> shape,
                                      UnrollOptions options) {
  SmallVector<int64_t> newShape;
  for (auto [i, dim] : llvm::enumerate(shape)) {
    if (i == options.getUnrollIdx()) {
      if (!options.shouldReduce()) {
        auto chunkSize = options.getChunkSize();
        if (auto attr = chunkSize.dyn_cast<Attribute>()) {
          auto chunkSizeInt = cast<IntegerAttr>(attr).getInt();
          newShape.push_back(chunkSizeInt);
        } else {
          newShape.push_back(ShapedType::kDynamic);
        }
      }
    } else {
      newShape.push_back(dim);
    }
  }
  return newShape;
}

TileType getUnrolledTileType(TileType tileType, UnrollOptions options) {
  auto shape = tileType.getShape();
  auto elementType = tileType.getElementType();
  auto newShape = getUnrolledShape(shape, options);
  return TileType::get(newShape, elementType);
}

Value createCombiningOp(OpBuilder &b, Location loc, zuan::CombiningKind kind,
                        Value lhs, Value rhs) {
  bool isInteger = false;
  if (auto shapedType = dyn_cast<ShapedType>(lhs.getType())) {
    isInteger = shapedType.getElementType().isInteger();
  }
  Value result;
  switch (kind) {
  case zuan::CombiningKind::ADD:
    if (isInteger) {
      result = arith::AddIOp::create(b, loc, lhs, rhs);
    } else {
      result = arith::AddFOp::create(b, loc, lhs, rhs);
    }
    break;
  case zuan::CombiningKind::MUL:
    if (isInteger) {
      result = arith::MulIOp::create(b, loc, lhs, rhs);
    } else {
      result = arith::MulFOp::create(b, loc, lhs, rhs);
    }
    break;
  case zuan::CombiningKind::MINIMUMF:
    assert(!isInteger && "MINIMUMF is only supported for floating point types");
    result = arith::MinimumFOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXIMUMF:
    assert(!isInteger && "MAXIMUMF is only supported for floating point types");
    result = arith::MaximumFOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXNUMF:
    assert(!isInteger && "MAXNUMF is only supported for floating point types");
    result = arith::MaxNumFOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MINNUMF:
    assert(!isInteger && "MINNUMF is only supported for floating point types");
    result = arith::MinNumFOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::AND:
    assert(isInteger && "ANDI is only supported for integer types");
    result = arith::AndIOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::OR:
    assert(isInteger && "ORI is only supported for integer types");
    result = arith::OrIOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::XOR:
    assert(isInteger && "XORI is only supported for integer types");
    result = arith::XOrIOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXUI:
    assert(isInteger && "MAXU is only supported for integer types");
    result = arith::MaxUIOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MINUI:
    assert(isInteger && "MINU is only supported for integer types");
    result = arith::MinUIOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXSI:
    assert(isInteger && "MAXS is only supported for integer types");
    result = arith::MaxSIOp::create(b, loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MINSI:
    assert(isInteger && "MINS is only supported for integer types");
    result = arith::MinSIOp::create(b, loc, lhs, rhs);
    break;
  }

  return result;
}

void splitDynamicOpForUnrolling(RewriterBase &rewriter, DynamicOp dynamicOp,
                                unsigned unrollIdx, ShapeInfo &shapeInfo) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto bodyBlock = &dynamicOp.getBody().front();
  assert(bodyBlock->mightHaveTerminator() && "expected `zuan.yield`");
  auto yieldOp = cast<YieldOp>(bodyBlock->getTerminator());

  /// Map from dim to the operations that use the dim.
  std::map<DimSize, SmallVector<Operation *>> dimToOps;
  auto yieldRegion = &yieldOp.getRegion();

  for (auto &op : yieldRegion->getOps()) {
    if (auto iface = dyn_cast<ZuanUnrollingInterface>(&op)) {
      if (auto shape = iface.getShapeToUnroll(shapeInfo)) {
        if (unrollIdx >= shape->size()) {
          continue;
        }
        auto dim = (*shape)[unrollIdx];
        if (dimToOps.count(dim) == 0) {
          dimToOps[dim] = {};
        }
        dimToOps[dim].push_back(&op);
      }
    }
  }

  if (dimToOps.size() <= 1) {
    return;
  }

  rewriter.setInsertionPoint(dynamicOp);

  // Create a new dynamic op for each dim except for the first one.
  for (auto [i, pair] : llvm::enumerate(dimToOps)) {
    if (i == 0) {
      continue;
    }
    auto dim = pair.first;
    auto ops = pair.second;

    DynamicOp::create(rewriter, 
        dynamicOp->getLoc(), dynamicOp.getInits(),
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          OpBuilder::InsertionGuard guard(builder);

          UnrollOptions options(builder.getIndexAttr(0),
                                dim.getOrCreateOpFoldResult(builder, loc),
                                unrollIdx, false);
          UnrollState state;
          state.valueMap.map(bodyBlock->getArguments(), args);
          SetVector<Value> valuesDefinedAbove;
          mlir::getUsedValuesDefinedAbove(dynamicOp.getBody(),
                                          valuesDefinedAbove);
          state.valueMap.map(valuesDefinedAbove.getArrayRef(),
                             valuesDefinedAbove.getArrayRef());

          auto yieldOp = YieldOp::create(builder, loc);
          state.yieldBlock = &yieldOp.getBody().front();

          builder.setInsertionPoint(yieldOp);

          for (auto op : ops) {
            unrollOp(builder, op, options, state);
          }
        });
  }

  for (auto [i, pair] : llvm::enumerate(dimToOps)) {
    if (i == 0) {
      continue;
    }
    for (auto op : pair.second) {
      rewriter.eraseOp(op);
    }
  }
}

bool isDynamicOpUnrolled(DynamicOp dynamicOp, unsigned targetRank,
                         ShapeInfo &shapeInfo) {
  auto yieldOp = dynamicOp.getYieldOp();
  auto yieldRegion = &yieldOp.getRegion();

  bool unrolled = true;
  for (auto &op : yieldRegion->getOps()) {
    if (auto iface = dyn_cast<ZuanUnrollingInterface>(&op)) {
      if (auto shape = iface.getShapeToUnroll(shapeInfo)) {
        if (shape->size() > targetRank) {
          unrolled = false;
          break;
        }
      }
    }
  }

  return unrolled;
}

void UnrollState::initialize(DynamicOp op) {
  SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(op.getBody(), valuesDefinedAbove);
  this->valueMap.map(valuesDefinedAbove.getArrayRef(),
                     valuesDefinedAbove.getArrayRef());
}

} // namespace zuan
} // namespace mlir
