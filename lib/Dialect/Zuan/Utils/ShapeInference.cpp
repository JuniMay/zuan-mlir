//===- ShapeInference.cpp - Local shape reification for Zuan ---*- C++ -*-===//
//
// This file implements local, demand-driven shape reification helpers for
// Zuan operations.
//
//===----------------------------------------------------------------------===//

#include "Zuan/Utils/ShapeInference.h"

#include "Zuan/IR/Zuan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace zuan {

namespace {

static SmallVector<OpFoldResult> getStaticShape(Builder &builder,
                                                TileType tileType) {
  SmallVector<OpFoldResult> shape;
  for (int64_t dim : tileType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      shape.push_back(OpFoldResult());
      continue;
    }
    shape.push_back(builder.getIndexAttr(dim));
  }
  return shape;
}

static void overlayStaticShape(Builder &builder, TileType tileType,
                               SmallVector<OpFoldResult> &shape) {
  assert(shape.size() == tileType.getRank() && "expected matching ranks");
  for (auto [idx, dim] : llvm::enumerate(tileType.getShape())) {
    if (!ShapedType::isDynamic(dim)) {
      shape[idx] = builder.getIndexAttr(dim);
    }
  }
}

static FailureOr<Value> getForwardedShapeOperand(Operation *op) {
  if (auto passthru =
          op->getAttrOfType<IntegerAttr>("zuan_passthru_operand")) {
    int64_t index = passthru.getInt();
    if (index < 0 || index >= static_cast<int64_t>(op->getNumOperands())) {
      return failure();
    }
    Value operand = op->getOperand(index);
    if (!isa<TileType>(operand.getType())) {
      return failure();
    }
    return operand;
  }

  if (op->getNumResults() != 1 || op->getNumRegions() != 0 ||
      !isa<TileType>(op->getResult(0).getType())) {
    return failure();
  }

  if (op->hasTrait<OpTrait::Elementwise>() &&
      op->hasTrait<OpTrait::SameOperandsAndResultType>()) {
    if (!op->getOperands().empty() && isa<TileType>(op->getOperand(0).getType())) {
      return op->getOperand(0);
    }
  }

  if (isa<arith::CmpIOp, arith::CmpFOp>(op) &&
      !op->getOperands().empty() && isa<TileType>(op->getOperand(0).getType())) {
    return op->getOperand(0);
  }

  return failure();
}

static FailureOr<SmallVector<OpFoldResult>>
reifyBlockArgumentShape(OpBuilder &builder, BlockArgument arg) {
  auto tileType = dyn_cast<TileType>(arg.getType());
  if (!tileType) {
    return failure();
  }

  if (auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
    if (arg.getArgNumber() == 0) {
      return failure();
    }
    return reifyZuanShape(builder, forOp.getInitArgs()[arg.getArgNumber() - 1]);
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(arg.getOwner()->getParentOp())) {
    return reifyZuanShape(builder, whileOp.getInits()[arg.getArgNumber()]);
  }

  auto shape = getStaticShape(builder, tileType);
  if (llvm::all_of(shape, [](OpFoldResult ofr) { return static_cast<bool>(ofr); })) {
    return shape;
  }
  return failure();
}

static FailureOr<SmallVector<OpFoldResult>>
reifyLoadShape(OpBuilder &builder, LoadOp loadOp) {
  if (auto subview = loadOp.getBase().getDefiningOp<memref::SubViewOp>()) {
    SmallVector<OpFoldResult> shape;
    auto droppedDims = subview.getDroppedDims();
    auto sizes = subview.getMixedSizes();
    // Preserve the subview's mixed sizes directly so downstream rewrites see
    // the exact chunking values, e.g. a stripmined `vp.getvl`, instead of
    // re-materializing generic memref.dim queries on the subview result.
    for (unsigned idx = 0, e = loadOp.getResult().getType().getRank(); idx < e;
         ++idx) {
      OpFoldResult size = computeUnreducedDim<OpFoldResult>(
          idx, sizes, [&](unsigned dim) { return droppedDims.test(dim); });
      if (auto attr = size.dyn_cast<Attribute>()) {
        shape.push_back(
            builder.getIndexAttr(cast<IntegerAttr>(attr).getInt()));
      } else {
        shape.push_back(size.dyn_cast<Value>());
      }
    }
    overlayStaticShape(builder, loadOp.getResult().getType(), shape);
    return shape;
  }

  auto baseType = cast<MemRefType>(loadOp.getBase().getType());
  auto loc = loadOp.getLoc();
  SmallVector<OpFoldResult> shape;
  for (auto [idx, dim] : llvm::enumerate(baseType.getShape())) {
    if (!ShapedType::isDynamic(dim)) {
      shape.push_back(builder.getIndexAttr(dim));
      continue;
    }
    shape.push_back(
        memref::DimOp::create(builder, loc, loadOp.getBase(), idx).getResult());
  }
  return shape;
}

static FailureOr<SmallVector<OpFoldResult>>
reifySplatShape(OpBuilder &builder, SplatOp splatOp) {
  SmallVector<OpFoldResult> mixedDims = splatOp.getMixedDims();
  SmallVector<OpFoldResult> shape(mixedDims.begin(), mixedDims.end());
  if (isa<TileType>(splatOp.getValue().getType())) {
    auto sourceShape = reifyZuanShape(builder, splatOp.getValue());
    if (failed(sourceShape)) {
      return failure();
    }
    shape.append(sourceShape->begin(), sourceShape->end());
  }
  overlayStaticShape(builder, splatOp.getResult().getType(), shape);
  return shape;
}

static FailureOr<SmallVector<OpFoldResult>>
reifyStepShape(OpBuilder &builder, StepOp stepOp) {
  SmallVector<OpFoldResult> mixedSizes = stepOp.getMixedSizes();
  SmallVector<OpFoldResult> shape(mixedSizes.begin(), mixedSizes.end());
  overlayStaticShape(builder, stepOp.getResult().getType(), shape);
  return shape;
}

static FailureOr<SmallVector<OpFoldResult>>
reifyMatmulShape(OpBuilder &builder, MatmulOp matmulOp) {
  auto lhsShape = reifyZuanShape(builder, matmulOp.getLhs());
  auto rhsShape = reifyZuanShape(builder, matmulOp.getRhs());
  if (failed(lhsShape) || failed(rhsShape)) {
    return failure();
  }

  auto resultType = matmulOp.getResult().getType();
  SmallVector<OpFoldResult> result;
  unsigned leadingSize = matmulOp.getLeadingSize();
  for (unsigned i = 0; i < leadingSize; ++i) {
    result.push_back((*lhsShape)[i]);
  }

  auto lhsRank = matmulOp.getLhs().getType().getRank();
  auto rhsRank = matmulOp.getRhs().getType().getRank();
  if (lhsRank >= rhsRank) {
    result.push_back((*lhsShape)[lhsShape->size() - 2]);
  }
  if (rhsRank >= lhsRank) {
    result.push_back((*rhsShape)[rhsShape->size() - 1]);
  }

  overlayStaticShape(builder, resultType, result);
  return result;
}

static FailureOr<SmallVector<OpFoldResult>>
reifyReductionShape(OpBuilder &builder, ReductionOp reductionOp) {
  auto sourceShape = reifyZuanShape(builder, reductionOp.getTile());
  if (failed(sourceShape)) {
    return failure();
  }

  SetVector<int64_t> reducedDims(reductionOp.getDims().begin(),
                                 reductionOp.getDims().end());
  SmallVector<OpFoldResult> result;
  for (auto [idx, dim] : llvm::enumerate(*sourceShape)) {
    if (!reducedDims.contains(idx)) {
      result.push_back(dim);
    }
  }
  overlayStaticShape(builder, reductionOp.getResult().getType(), result);
  return result;
}

static FailureOr<SmallVector<OpFoldResult>>
reifyOuterShape(OpBuilder &builder, OuterOp outerOp) {
  auto lhsShape = reifyZuanShape(builder, outerOp.getLhs());
  auto rhsShape = reifyZuanShape(builder, outerOp.getRhs());
  if (failed(lhsShape) || failed(rhsShape)) {
    return failure();
  }

  SmallVector<OpFoldResult> result(lhsShape->begin(), lhsShape->end());
  if (outerOp.getLhs().getType().getRank() <= outerOp.getRhs().getType().getRank()) {
    result.push_back(rhsShape->back());
  }
  overlayStaticShape(builder, outerOp.getResult().getType(), result);
  return result;
}

static FailureOr<SmallVector<OpFoldResult>>
reifyMaskShape(OpBuilder &builder, MaskOp maskOp, unsigned resultNumber) {
  if (auto *maskedOp = maskOp.getMaskedOp()) {
    return reifyZuanShape(builder, maskedOp->getResult(resultNumber));
  }
  if (auto maskedoff = maskOp.getMaskedoff()) {
    return reifyZuanShape(builder, maskedoff);
  }
  return reifyZuanShape(builder, maskOp.getMask());
}

static FailureOr<SmallVector<OpFoldResult>>
reifyGatherShape(OpBuilder &builder, GatherOp gatherOp) {
  if (gatherOp.getIndices().empty()) {
    return SmallVector<OpFoldResult>{};
  }
  auto shape = reifyZuanShape(builder, gatherOp.getIndices().front());
  if (failed(shape)) {
    return failure();
  }
  overlayStaticShape(builder, gatherOp.getResult().getType(), *shape);
  return *shape;
}

static FailureOr<SmallVector<OpFoldResult>>
reifySCFForResultShape(OpBuilder &builder, OpResult result) {
  auto forOp = cast<scf::ForOp>(result.getOwner());
  return reifyZuanShape(builder, forOp.getInitArgs()[result.getResultNumber()]);
}

static FailureOr<SmallVector<OpFoldResult>>
reifySCFWhileResultShape(OpBuilder &builder, OpResult result) {
  auto whileOp = cast<scf::WhileOp>(result.getOwner());
  return reifyZuanShape(builder, whileOp.getInits()[result.getResultNumber()]);
}

static FailureOr<SmallVector<OpFoldResult>>
reifyOpResultShape(OpBuilder &builder, OpResult result) {
  if (!isa<TileType>(result.getType())) {
    return failure();
  }

  Operation *op = result.getOwner();
  return TypeSwitch<Operation *, FailureOr<SmallVector<OpFoldResult>>>(op)
      .Case<LoadOp>([&](LoadOp loadOp) { return reifyLoadShape(builder, loadOp); })
      .Case<SplatOp>([&](SplatOp splatOp) { return reifySplatShape(builder, splatOp); })
      .Case<StepOp>([&](StepOp stepOp) { return reifyStepShape(builder, stepOp); })
      .Case<MatmulOp>([&](MatmulOp matmulOp) { return reifyMatmulShape(builder, matmulOp); })
      .Case<ReductionOp>(
          [&](ReductionOp reductionOp) { return reifyReductionShape(builder, reductionOp); })
      .Case<OuterOp>([&](OuterOp outerOp) { return reifyOuterShape(builder, outerOp); })
      .Case<CastOp, SelectOp>(
          [&](auto shapePreservingOp) -> FailureOr<SmallVector<OpFoldResult>> {
        auto shape = reifyZuanShape(builder, shapePreservingOp->getOperand(0));
        if (failed(shape)) {
          return FailureOr<SmallVector<OpFoldResult>>(failure());
        }
        overlayStaticShape(builder, cast<TileType>(result.getType()), *shape);
        return *shape;
      })
      .Case<MaskOp>([&](MaskOp maskOp) {
        return reifyMaskShape(builder, maskOp, result.getResultNumber());
      })
      .Case<GatherOp>([&](GatherOp gatherOp) { return reifyGatherShape(builder, gatherOp); })
      .Case<scf::ForOp>(
          [&](scf::ForOp) { return reifySCFForResultShape(builder, result); })
      .Case<scf::WhileOp>(
          [&](scf::WhileOp) { return reifySCFWhileResultShape(builder, result); })
      .Default([&](Operation *unknownOp) -> FailureOr<SmallVector<OpFoldResult>> {
        auto forwarded = getForwardedShapeOperand(unknownOp);
        if (failed(forwarded)) {
          return failure();
        }
        auto shape = reifyZuanShape(builder, *forwarded);
        if (failed(shape)) {
          return failure();
        }
        overlayStaticShape(builder, cast<TileType>(result.getType()), *shape);
        return *shape;
      });
}

} // namespace

FailureOr<SmallVector<OpFoldResult>> reifyZuanShape(OpBuilder &builder,
                                                    Value value) {
  auto tileType = dyn_cast<TileType>(value.getType());
  if (!tileType) {
    return failure();
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    return reifyBlockArgumentShape(builder, blockArg);
  }
  if (auto result = dyn_cast<OpResult>(value)) {
    return reifyOpResultShape(builder, result);
  }

  auto shape = getStaticShape(builder, tileType);
  if (llvm::all_of(shape, [](OpFoldResult ofr) { return static_cast<bool>(ofr); })) {
    return shape;
  }
  return failure();
}

FailureOr<OpFoldResult> reifyZuanDim(OpBuilder &builder, Value value,
                                     unsigned dim) {
  auto tileType = dyn_cast<TileType>(value.getType());
  if (!tileType || dim >= tileType.getRank()) {
    return failure();
  }
  int64_t staticDim = tileType.getShape()[dim];
  if (!ShapedType::isDynamic(staticDim)) {
    return OpFoldResult(builder.getIndexAttr(staticDim));
  }

  auto shape = reifyZuanShape(builder, value);
  if (failed(shape) || dim >= shape->size() || !(*shape)[dim]) {
    return failure();
  }
  return (*shape)[dim];
}

LogicalResult resolveDimUsersOfResult(OpResult result,
                                      PatternRewriter &rewriter) {
  if (!isa<TileType>(result.getType())) {
    return success();
  }

  for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
    // Only direct `zuan.dim` users need special handling here. Any other live
    // use still needs the tile result itself, so the producer cannot be
    // destructively erased yet.
    auto dimOp = dyn_cast<DimOp>(use.getOwner());
    if (!dimOp || use.getOperandNumber() != 0) {
      continue;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(dimOp);
    // Destructive rewrites use this helper before erasing a tile producer so
    // direct `zuan.dim` users are rewritten while the original tile semantics
    // are still available.
    auto reified = reifyZuanDim(rewriter, result, dimOp.getDim());
    if (failed(reified)) {
      return dimOp.emitOpError("failed to reify queried dimension");
    }
    rewriter.replaceOp(dimOp,
                       getOrCreateIndexValue(rewriter, *reified, dimOp.getLoc()));
  }

  return success();
}

LogicalResult resolveDimUsersOfOp(Operation *op, PatternRewriter &rewriter) {
  for (OpResult result : op->getResults()) {
    if (failed(resolveDimUsersOfResult(result, rewriter))) {
      return failure();
    }
  }
  return success();
}

std::optional<int64_t> getConstantZuanIntValue(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    return cast<IntegerAttr>(attr).getInt();
  }
  return std::nullopt;
}

unsigned computeUnreducedIdx(unsigned idx, size_t unreducedRank,
                             function_ref<bool(unsigned)> isReduced) {
  /// The index in original shape that we're trying to match with the current
  /// result shape idx.
  unsigned currSourceIdx = 0;
  /// The index in result shape that is not matched yet.
  unsigned currResultIdx = 0;
  while (currSourceIdx < unreducedRank) {
    if (isReduced(currSourceIdx)) {
      // This dim is reduced, have not found the source dim for currResultIdx
      currSourceIdx += 1;
      continue;
    }
    // This dim is not reduced, found the source dim for currResultIdx.
    if (currResultIdx == idx) {
      // We found the source dim for the current result dim.
      break;
    }
    // Not found yet, trying to find for the next result dim.
    currSourceIdx += 1;
    currResultIdx += 1;
  }
  return currSourceIdx;
}

Value getOrCreateIndexValue(OpBuilder &builder, OpFoldResult ofr,
                            Location loc) {
  if (auto value = ofr.dyn_cast<Value>()) {
    return value;
  }
  auto integer = cast<IntegerAttr>(ofr.dyn_cast<Attribute>()).getInt();
  return arith::ConstantIndexOp::create(builder, loc, integer);
}

} // namespace zuan
} // namespace mlir
