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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

#include "Zuan/IR/Zuan.h"
#include "Zuan/Interfaces/ZuanUnrollingInterface.h"
#include "Zuan/Utils/ShapeInference.h"
#include "Zuan/Utils/Unrolling.h"
#include "mlir/Analysis/SliceAnalysis.h"

#define DEBUG_TYPE "zuan-unrolling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "] ")

namespace mlir {
namespace zuan {

static Value getMemrefValueInCurrentScope(OpBuilder &builder, Value operand,
                                          UnrollState &state) {
  if (state.valueMap.contains(operand)) {
    return state.valueMap.lookup(operand);
  }
  if (!operand.getDefiningOp()) {
    // Function/block arguments are already in scope here. If they needed
    // remapping into a freshly cloned control-flow region, `valueMap` would
    // have been populated first.
    return operand;
  }
  // Otherwise the memref is produced by an op result that is not yet available
  // in the current insertion scope. This happens, for example, when stripmining
  // is rebuilding a new `scf.for`/`scf.while` body and a memref SSA value from
  // the old slice lives inside that control-flow region. Clone the producer
  // chain into the current scope first, then slice the cloned value.
  return getUnrolledValue(builder, operand, getCloneOptions(), state);
}

Value getUnrolledValue(OpBuilder &builder, Value operand, UnrollOptions options,
                       UnrollState &state) {
  if (isa<MemRefType>(operand.getType()) &&
      options.getUnrollIdx() != UnrollOptions::kNoUnrollIdx) {
    // Memrefs are sliced in the coordinate space of the current SSA value. The
    // helper above makes sure that value is actually valid at the current
    // insertion point, which is important when unrolling reconstructs new SCF
    // regions instead of reusing the old ones in place.
    Value memref = getMemrefValueInCurrentScope(builder, operand, state);
    return createMemrefSlice(builder, memref, options);
  }
  if (state.valueMap.contains(operand)) {
    // This is defined above, or fallback to clone.
    return state.valueMap.lookup(operand);
  }
  auto definingOp = operand.getDefiningOp();
  if (!definingOp && isa<BlockArgument>(operand)) {
    // Block arguments are remapped explicitly when cloning loop/control-flow
    // bodies, so a bare argument can be reused here.
    return operand;
  }
  auto opResult = dyn_cast<OpResult>(operand);
  assert(opResult && "expected an op result");
  auto newOp = unrollOp(builder, opResult.getOwner(), options, state);
  if (!newOp) {
    llvm::report_fatal_error("unrollOp failed; see previous diagnostic");
  }
  return newOp->getResult(opResult.getResultNumber());
}

Value getUnrolledMemref(OpBuilder &builder, Value memref, UnrollOptions options,
                        UnrollState &state) {
  // MemRef handler will not branch on the ops anymore. It will just add a
  // subview op on the result. The folding of subview is expected to be handled
  // by fold-memref-alias-ops pass. This function is kept in case future
  // optimizations are needed on the memref.
  return getUnrolledValue(builder, memref, options, state);
}

Value createMemrefSlice(OpBuilder &rewriter, Value memref,
                        UnrollOptions options) {
  auto loc = memref.getLoc();

  auto memrefType = cast<MemRefType>(memref.getType());
  auto memrefShape = memrefType.getShape();

  auto rank = memrefShape.size();

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  for (size_t i = 0; i < rank; ++i) {
    if (i == options.getUnrollIdx()) {
      // The slice is created directly on the current memref value, so the
      // caller-provided offset/chunk pair is already in the right coordinate
      // space for this view.
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

static Operation *unrollSCFForOp(OpBuilder &builder, scf::ForOp forOp,
                                 UnrollOptions options, UnrollState &state) {
  OpBuilder::InsertionGuard guard(builder);

  SmallVector<Value> newInits;
  for (auto init : forOp.getInitArgs()) {
    auto newInit = getUnrolledValue(builder, init, options, state);
    newInits.push_back(newInit);
  }

  // The surrounding stripmining/root unrolling decides which tile dimension is
  // being split. Nested loop bounds usually describe independent iteration
  // spaces such as a contraction dimension, so cloning them preserves the
  // original loop semantics while still allowing carried tile values to change
  // shape.
  auto lb = getUnrolledValue(builder, forOp.getLowerBound(), getCloneOptions(),
                             state);
  auto ub = getUnrolledValue(builder, forOp.getUpperBound(), getCloneOptions(),
                             state);
  auto step =
      getUnrolledValue(builder, forOp.getStep(), getCloneOptions(), state);

  UnrollState inLoopState;
  inLoopState.valueMap = IRMapping{state.valueMap};
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

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return unrollSCFForOp(builder, forOp, options, state);
  }

  if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
    auto memref = getUnrolledValue(builder, dimOp.getSource(), options, state);
    auto index = getUnrolledValue(builder, dimOp.getIndex(), options, state);
    if (options.getUnrollIdx() == UnrollOptions::kNoUnrollIdx) {
      return memref::DimOp::create(builder, op->getLoc(), memref, index);
    }

    auto constantIndex = dyn_cast_or_null<arith::ConstantIndexOp>(
        index.getDefiningOp());
    if (!constantIndex) {
      return memref::DimOp::create(builder, op->getLoc(), memref, index);
    }

    int64_t dim = constantIndex.value();
    int64_t unrollIdx = options.getUnrollIdx();
    if (options.shouldReduce()) {
      if (dim == unrollIdx) {
        // The unrolled dimension disappeared via rank reduction, so its extent
        // is the unit chunk that triggered the reduction.
        return arith::ConstantIndexOp::create(builder, op->getLoc(), 1);
      }
      if (dim > unrollIdx) {
        dim -= 1;
      }
    }

    auto newIndex = arith::ConstantIndexOp::create(builder, op->getLoc(), dim);
    return memref::DimOp::create(builder, op->getLoc(), memref, newIndex);
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

  if (!isMemoryEffectFree(op)) {
    op->emitError()
        << "generic unrolling would clone side-effectful operation '"
        << op->getName()
        << "'; add explicit unrolling support or map the dominating value instead";
    return nullptr;
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

void UnrollState::initialize(Operation *root) {
  BackwardSliceOptions options;
  options.inclusive = true;

  SetVector<Operation *> slice;
  (void)getBackwardSlice(root, &slice, options);
  llvm::SmallDenseSet<Operation *> inSlice(slice.begin(), slice.end());
  DominanceInfo dominance(root);

  auto reuseDominatingResult = [&](Value result) {
    if (isa<MemRefType>(result.getType())) {
      // Reuse any dominating memref view/buffer instead of cloning it into
      // each generated stripmining loop body. This matters for nested SCF
      // rewrites: a root inside a new `scf.while` body may still depend on
      // staged memref views defined in a parent block, and cloning those
      // producers would recreate fresh alloc/copy chains inside the loop.
      valueMap.map(result, result);
      return;
    }
    if (auto tileType = dyn_cast<TileType>(result.getType())) {
      if (tileType.getRank() == 0) {
        // The same dominance rule applies to scalar summaries: if the rank-0
        // tile already dominates the rewritten root, carry it in directly
        // instead of cloning the producing scalar path again.
        valueMap.map(result, result);
      }
    }
  };

  for (Operation *op : slice) {
    for (Value operand : op->getOperands()) {
      auto *def = operand.getDefiningOp();
      if (!def || !inSlice.contains(def)) {
        valueMap.map(operand, operand);
      }
    }
    if (op != root && dominance.properlyDominates(op, root)) {
      for (Value result : op->getResults()) {
        reuseDominatingResult(result);
      }
    }
  }

  Operation *scope = root;
  while (Operation *parent = scope->getParentOp()) {
    scope = parent;
    if (scope->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      break;
    }
  }

  // Nested SCF bodies can capture dominating memrefs without making them
  // explicit operands of the slice root. Walk the enclosing isolated scope so
  // those captured staging views are also reused instead of cloned.
  scope->walk([&](Operation *op) {
    if (op == root || !dominance.properlyDominates(op, root)) {
      return;
    }
    for (Value result : op->getResults()) {
      reuseDominatingResult(result);
    }
  });
}

} // namespace zuan
} // namespace mlir
