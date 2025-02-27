//===- ZuanOps.cpp - Zuan Operations ----------------------------*- C++ -*-===//
//
// This file implements the Zuan operations.
//
//===----------------------------------------------------------------------===//

#include "Zuan/IR/Zuan.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

#define GET_OP_CLASSES
#include "Zuan/IR/ZuanOps.cpp.inc"

namespace mlir {
namespace zuan {

void DynamicOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto &operand : getDpsInitsMutable()) {
    effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

void DynamicOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    ValueRange inits,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodybuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(inits);
  result.addTypes(resultTypes);

  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  for (auto init : inits) {
    auto memrefType = dyn_cast<MemRefType>(init.getType());
    assert(memrefType && "expected memref type");
    auto tileType =
        TileType::get(memrefType.getShape(), memrefType.getElementType());
    bodyBlock->addArgument(tileType, init.getLoc());
  }
  if (bodybuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodybuilder(builder, result.location, bodyBlock->getArguments());
  }
}

void DynamicOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inits,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodybuilder) {
  build(builder, result, {}, inits, bodybuilder);
}

LogicalResult DynamicOp::verify() {
  auto block = &getBody().front();
  // Check terminator
  if (block->mightHaveTerminator() && !isa<YieldOp>(block->getTerminator())) {
    return emitOpError("expected a `zuan.yield` terminator");
  }
  // Verify that the only memory writes are happening in the yield op.
  for (auto &op : getBody().getOps()) {
    if (isa<YieldOp>(op)) {
      // Yield op is the terminator, just break.
      break;
    }
    auto effects = *mlir::getEffectsRecursively(&op);
    for (auto effect : effects) {
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        return emitOpError("expected only memory reads in the dynamic region");
      }
    }
  }
  return success();
}

/// Elide the dynamic op if there are no memory write effects and all the
/// yielded scalars are defined outside the dynamic op.
struct ElideEmptyDynamicOp : OpRewritePattern<DynamicOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicOp op,
                                PatternRewriter &rewriter) const override {
    Block *block = &op.getBody().front();
    auto terminator = cast<YieldOp>(block->getTerminator());

    // The only memory write effects are in the yield op, so only check if the
    // yielded region has memory effects. Other operations in the dynamic region
    // will not be checked.
    auto effects = *mlir::getEffectsRecursively(terminator);
    for (auto effect : effects) {
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        return rewriter.notifyMatchFailure(op, "memory write effect");
      }
    }

    auto yieldedOperands = terminator.getScalars();
    if (yieldedOperands.empty()) {
      rewriter.eraseOp(op);
    } else {
      // Check if the yielded scalars are defined outside the dynamic op.
      for (auto operand : yieldedOperands) {
        if (auto defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == block) {
            // The operand is defined inside the dynamic op.
            return rewriter.notifyMatchFailure(
                op, "operand defined inside the dynamic op");
          }
        }
      }
      // All the operands are defined outside the dynamic op, replace the
      // dynamic op with the yielded operands.
      rewriter.replaceOp(op, yieldedOperands);
    }
    return success();
  }
};

void DynamicOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ElideEmptyDynamicOp>(context);
}

void YieldOp::build(OpBuilder &builder, OperationState &result) {
  (void)result.addRegion();
}

void YieldOp::build(OpBuilder &builder, OperationState &result,
                    ValueRange scalars,
                    function_ref<void(OpBuilder &, Location)> bodyBuilder) {
  result.addOperands(scalars);
  Region *bodyRegion = result.addRegion();
  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);

    Block *bodyBlock = builder.createBlock(bodyRegion);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location);
  }
}

LogicalResult YieldOp::verify() {
  auto parentOp = dyn_cast<DynamicOp>((*this)->getParentOp());
  if (!parentOp) {
    return emitOpError("expected parent op to be a `zuan.dynamic`");
  }
  // Check the scalars and the parent results.
  if (parentOp.getNumResults() != this->getScalars().size()) {
    return emitOpError(
        "expected the number of parent results and scalars to be the same");
  }
  // Check the types of the scalars and the parent results.
  for (auto [scalar, result] :
       llvm::zip(this->getScalars(), parentOp.getResults())) {
    if (scalar.getType() != result.getType()) {
      return emitOpError("expected the type of the scalar and the parent "
                         "result to be the same");
    }
  }
  return success();
}

LogicalResult MatmulOp::inferReturnTypes(MLIRContext *context,
                                         std::optional<Location> location,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type> &inferred) {
  auto lhsType = cast<TileType>(adaptor.getLhs().getType());
  auto rhsType = cast<TileType>(adaptor.getRhs().getType());
  if (lhsType.getElementType() != rhsType.getElementType()) {
    return emitOptionalError(
        location,
        "expected the element type of the lhs and rhs to be the same");
  }
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();

  auto lhsK = lhsShape.back();
  auto rhsK = rhsShape[rhsShape.size() - 2];
  if (!TileType::isDimCompatible(lhsK, rhsK)) {
    return emitOptionalError(
        location,
        "expected the inner dimensions of the lhs and rhs to be compatible");
  }

  ArrayRef<int64_t> lhsLeadingDims(lhsShape.begin(), lhsShape.size() - 2);
  ArrayRef<int64_t> rhsLeadingDims(rhsShape.begin(), rhsShape.size() - 2);
  auto resultShape =
      TileType::selectStaticShape(lhsLeadingDims, rhsLeadingDims);
  resultShape.push_back(lhsShape[lhsShape.size() - 2]);
  resultShape.push_back(rhsShape.back());

  auto tileType = TileType::get(resultShape, lhsType.getElementType());
  inferred.push_back(tileType);
  return success();
}

LogicalResult MultiReductionOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferred) {
  auto tile = adaptor.getTile();
  auto dims = adaptor.getDims();
  if (dims.empty()) {
    inferred.push_back(tile.getType());
    return success();
  }
  auto tileShape = cast<TileType>(tile.getType()).getShape();
  auto tileElementType = cast<TileType>(tile.getType()).getElementType();
  SmallVector<int64_t> resultShape;
  for (size_t i = 0, e = tileShape.size(); i < e; ++i) {
    if (!llvm::is_contained(dims, i)) {
      resultShape.push_back(tileShape[i]);
    }
  }
  inferred.push_back(TileType::get(resultShape, tileElementType));
  return success();
}

LogicalResult LoadOp::inferReturnTypes(MLIRContext *context,
                                       std::optional<Location> location,
                                       Adaptor adaptor,
                                       SmallVectorImpl<Type> &inferred) {
  auto baseType = cast<MemRefType>(adaptor.getBase().getType());
  auto tileType = TileType::get(baseType.getShape(), baseType.getElementType());
  inferred.push_back(tileType);
  return success();
}

LogicalResult SplatOp::inferReturnTypes(MLIRContext *context,
                                        std::optional<Location> location,
                                        Adaptor adaptor,
                                        SmallVectorImpl<Type> &inferred) {
  auto value = adaptor.getValue();
  auto dims = adaptor.getStaticDims();
  SmallVector<int64_t> splatShape(dims.begin(), dims.end());
  auto valueType = value.getType();
  if (auto tileType = dyn_cast<TileType>(valueType)) {
    splatShape.append(tileType.getShape().begin(), tileType.getShape().end());
    valueType = tileType.getElementType();
  }
  auto tileType = TileType::get(splatShape, valueType);
  inferred.push_back(tileType);
  return success();
}

void SplatOp::build(OpBuilder &builder, OperationState &result, Value value,
                    ArrayRef<int64_t> dims) {
  SmallVector<OpFoldResult> ofrDims =
      llvm::to_vector(llvm::map_range(dims, [&](int64_t v) -> OpFoldResult {
        return builder.getIndexAttr(v);
      }));
  build(builder, result, value, ofrDims);
}

void SplatOp::build(OpBuilder &builder, OperationState &result, Value value,
                    ArrayRef<OpFoldResult> dims) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);
  build(builder, result, value, dynamicDims, staticDims);
}

SmallVector<OpFoldResult> SplatOp::getMixedDims() {
  Builder builder((*this)->getContext());
  return getMixedValues(getStaticDims(), getDims(), builder);
}

LogicalResult OuterOp::inferReturnTypes(MLIRContext *context,
                                        std::optional<Location> location,
                                        Adaptor adaptor,
                                        SmallVectorImpl<Type> &inferred) {
  auto lhsType = cast<TileType>(adaptor.getLhs().getType());
  auto rhsType = cast<TileType>(adaptor.getRhs().getType());

  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();

  if (std::abs(lhsRank - rhsRank) > 1) {
    return emitOptionalError(
        location,
        "expected the rank of the lhs and rhs to differ by at most one");
  }
  unsigned leadingRank = lhsRank;
  if (lhsRank >= rhsRank) {
    leadingRank -= 1; // Ignore the last dimension.
  }

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  for (unsigned i = 0; i < leadingRank; ++i) {
    auto lhsDim = lhsShape[i];
    auto rhsDim = rhsShape[i];
    if (!TileType::isDimCompatible(lhsDim, rhsDim)) {
      return emitOptionalError(location, "expected the leading dimensions of "
                                         "the lhs and rhs to be compatible");
    }
  }
  SmallVector<int64_t> resultShape{lhsShape.begin(), lhsShape.end()};
  if (lhsRank <= rhsRank) {
    resultShape.push_back(rhsShape.back());
  }
  inferred.push_back(TileType::get(resultShape, lhsType.getElementType()));
  return success();
}

LogicalResult StepOp::inferReturnTypes(MLIRContext *context,
                                       std::optional<Location> location,
                                       Adaptor adaptor,
                                       SmallVectorImpl<Type> &inferred) {
  auto staticSizes = adaptor.getStaticSizes();
  auto start = adaptor.getStart();
  inferred.push_back(TileType::get(staticSizes, start.getType()));
  return success();
}

void StepOp::build(OpBuilder &builder, OperationState &result, Value start,
                   int64_t dim, ArrayRef<int64_t> staticSizes) {
  SmallVector<OpFoldResult> ofrSizes = llvm::to_vector(
      llvm::map_range(staticSizes, [&](int64_t v) -> OpFoldResult {
        return builder.getIndexAttr(v);
      }));
  build(builder, result, start, dim, ofrSizes);
}

void StepOp::build(OpBuilder &builder, OperationState &result, Value start,
                   int64_t dim, ArrayRef<OpFoldResult> staticSizes) {
  SmallVector<int64_t> staticSizesVec;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(staticSizes, dynamicSizes, staticSizesVec);
  build(builder, result, start, builder.getIndexAttr(dim), dynamicSizes,
        builder.getDenseI64ArrayAttr(staticSizesVec));
}

SmallVector<OpFoldResult> StepOp::getMixedSizes() {
  Builder builder((*this)->getContext());
  return getMixedValues(getStaticSizes(), getSizes(), builder);
}

} // namespace zuan
} // namespace mlir
