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
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cassert>

#include "Zuan/IR/Zuan.h"
#include "Zuan/Interfaces/ZuanUnrollingInterface.h"
#include "Zuan/Utils/ShapeInference.h"
#include "Zuan/Utils/Unrolling.h"

#define DEBUG_TYPE "zuan-unrolling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "] ")

namespace mlir {
namespace zuan {

Value getUnrolledValue(OpBuilder &builder, Value operand, UnrollOptions options,
                       UnrollState &state) {
  if (state.valueMap.contains(operand)) {
    return state.valueMap.lookup(operand);
  }
  auto definingOp = operand.getDefiningOp();
  if (!definingOp) {
    // Block argument inside the dynamic op, and the parent block is not cloned.
    return operand;
  }

  auto newOp = unrollOp(builder, definingOp, options, state);
  assert(newOp->getNumResults() == 1 && "expected a single result");
  return newOp->getResult(0);
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
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value dim = rewriter.create<memref::DimOp>(loc, memref, idx);
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
  Value subview = rewriter.create<memref::SubViewOp>(loc, resultType, memref,
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
  auto newSubviewOp = builder.create<memref::SubViewOp>(
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
      builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, newInits);

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
  builder.create<scf::YieldOp>(forOp.getLoc(), newYieldedValues);
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
      op->hasTrait<OpTrait::SameOperandsAndResultElementType>()) {
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

  if (isa<arith::CmpIOp, arith::CmpFOp>(op)) {
    auto lhs = getUnrolledValue(builder, op->getOperand(0), options, state);
    auto rhs = getUnrolledValue(builder, op->getOperand(1), options, state);
    if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
      return builder.create<arith::CmpIOp>(op->getLoc(), cmpi.getPredicate(),
                                           lhs, rhs);
    } else if (auto cmpf = dyn_cast<arith::CmpFOp>(op)) {
      return builder.create<arith::CmpFOp>(op->getLoc(), cmpf.getPredicate(),
                                           lhs, rhs);
    } else {
      llvm_unreachable("unexpected comparison operation");
    }
  }

  LLVM_DEBUG(DBGS() << "Fallback to default: " << op->getName() << "\n");
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
      result = b.create<arith::AddIOp>(loc, lhs, rhs);
    } else {
      result = b.create<arith::AddFOp>(loc, lhs, rhs);
    }
    break;
  case zuan::CombiningKind::MUL:
    if (isInteger) {
      result = b.create<arith::MulIOp>(loc, lhs, rhs);
    } else {
      result = b.create<arith::MulFOp>(loc, lhs, rhs);
    }
    break;
  case zuan::CombiningKind::MINIMUMF:
    assert(!isInteger && "MINIMUMF is only supported for floating point types");
    result = b.create<arith::MinimumFOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXIMUMF:
    assert(!isInteger && "MAXIMUMF is only supported for floating point types");
    result = b.create<arith::MaximumFOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXNUMF:
    assert(!isInteger && "MAXNUMF is only supported for floating point types");
    result = b.create<arith::MaxNumFOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MINNUMF:
    assert(!isInteger && "MINNUMF is only supported for floating point types");
    result = b.create<arith::MinNumFOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::AND:
    assert(isInteger && "ANDI is only supported for integer types");
    result = b.create<arith::AndIOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::OR:
    assert(isInteger && "ORI is only supported for integer types");
    result = b.create<arith::OrIOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::XOR:
    assert(isInteger && "XORI is only supported for integer types");
    result = b.create<arith::XOrIOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXUI:
    assert(isInteger && "MAXU is only supported for integer types");
    result = b.create<arith::MaxUIOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MINUI:
    assert(isInteger && "MINU is only supported for integer types");
    result = b.create<arith::MinUIOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MAXSI:
    assert(isInteger && "MAXS is only supported for integer types");
    result = b.create<arith::MaxSIOp>(loc, lhs, rhs);
    break;
  case zuan::CombiningKind::MINSI:
    assert(isInteger && "MINS is only supported for integer types");
    result = b.create<arith::MinSIOp>(loc, lhs, rhs);
    break;
  }

  return result;
}

} // namespace zuan
} // namespace mlir
