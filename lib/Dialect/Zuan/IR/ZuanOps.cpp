//===- ZuanOps.cpp - Zuan Operations ----------------------------*- C++ -*-===//
//
// This file implements the Zuan operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <cassert>
#include <cstdint>
#include <optional>

#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/ShapeInference.h"
#include "Zuan/Utils/Unrolling.h"

#define GET_OP_CLASSES
#include "Zuan/IR/ZuanOps.cpp.inc"

namespace mlir {
namespace zuan {

//===----------------------------------------------------------------------===//
// DynamicOp
//===----------------------------------------------------------------------===//

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
    // Aggressively check the memory writes.
    auto effects = mlir::getEffectsRecursively(&op).value_or(
        SmallVector<MemoryEffects::EffectInstance>{});
    for (auto effect : effects) {
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        return emitOpError("expected only memory reads in the dynamic region");
      }
    }
  }
  return success();
}

void DynamicOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto bodyBlock = &getBody().front();
  for (auto [bbArg, memref] :
       llvm::zip(bodyBlock->getArguments(), getDpsInits())) {
    auto memrefType = cast<MemRefType>(memref.getType());
    ShapeVector shape;
    for (unsigned i = 0, e = memrefType.getRank(); i < e; ++i) {
      shape.push_back(DimSize(bbArg, i));
    }
    shapeInfo.markEquivalent(bbArg, shape);
  }
  // Recursively infer the shape of the dynamic region.
  for (auto &op : bodyBlock->getOperations()) {
    shapeInfo.inferShape(&op, state);
  }
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
    // will not be checked. Here the check is aggressive, if no side-effects
    // interface provided, it is assumed to have no memory writes.
    auto effects = mlir::getEffectsRecursively(terminator)
                       .value_or(SmallVector<MemoryEffects::EffectInstance>{});
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

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void YieldOp::build(OpBuilder &builder, OperationState &result) {
  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  (void)bodyBlock;
}

void YieldOp::build(OpBuilder &builder, OperationState &result,
                    ValueRange scalars,
                    function_ref<void(OpBuilder &, Location)> bodyBuilder) {
  result.addOperands(scalars);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);

  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
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
    // 0-D tile or scalar.
    auto type = getElementTypeOrSelf(scalar.getType());
    if (type != result.getType()) {
      return emitOpError("expected the type of the scalar and the parent "
                         "result to be the same");
    }
  }
  return success();
}

void YieldOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto bodyBlock = &getBody().front();
  for (auto &op : bodyBlock->getOperations()) {
    shapeInfo.inferShape(&op, state);
  }
}

//===----------------------------------------------------------------------===//
// MaskOp
//===----------------------------------------------------------------------===//

void MaskOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                   Value mask,
                   function_ref<void(OpBuilder &, Location)> bodyBuilder,
                   Value maskedoff) {
  build(builder, result, mask, bodyBuilder, maskedoff);
  result.addTypes(resultType);
}

void MaskOp::build(OpBuilder &builder, OperationState &result, Value mask,
                   function_ref<void(OpBuilder &, Location)> bodyBuilder,
                   Value maskedoff) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(mask);
  if (maskedoff) {
    result.addOperands(maskedoff);
  }

  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);

  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location);
  } else {
    MaskOp::ensureTerminator(*bodyRegion, builder, result.location);
  }
}

LogicalResult MaskOp::verify() {
  auto mask = getMask();
  // Check the returned types are the same as the maskedoff value.
  if (auto maskedoff = getMaskedoff()) {
    auto maskedoffType = maskedoff.getType();
    if (!TileType::isShapeCompatible(mask.getType().getShape(),
                                     maskedoffType.getShape())) {
      return emitOpError(
          "expected the mask and maskedoff types to be compatible");
    }
  }
  // Expect only one operation inside.
  Block *bodyBlock = &getBody().front();
  if (bodyBlock->empty()) {
    return emitOpError("expected the masked region to have an operation");
  }
  if (bodyBlock->getOperations().size() > 2) {
    return emitOpError(
        "expected only one operation and terminator inside the masked region");
  }

  return success();
}

void MaskOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto mask = getMask();
  auto maskedoff = getMaskedoff();
  if (maskedoff) {
    // Mask and Maskedoff (passthru) are shape-equivalent.
    shapeInfo.markEquivalent(mask, maskedoff);
  }
  if (auto parentMask = state.getMask()) {
    // Nested masks are shape-equivalent.
    shapeInfo.markEquivalent(mask, *parentMask);
  }
  state.pushMask(mask);
  // Recursively infer the shape of the masked region.
  auto bodyBlock = &getBody().front();
  for (auto &op : bodyBlock->getOperations()) {
    shapeInfo.inferShape(&op, state);
  }
  state.popMask();
}

Operation *MaskOp::unroll(OpBuilder &builder, UnrollOptions options,
                          UnrollState &state) {
  Value mask = getUnrolledValue(builder, getMask(), options, state);
  Value maskedoff = getMaskedoff();
  if (maskedoff) {
    maskedoff = getUnrolledValue(builder, maskedoff, options, state);
  }
  auto bodyBlock = &getBody().front();
  assert(bodyBlock->mightHaveTerminator() &&
         "expected the masked region to have a terminator");
  auto terminator = cast<MaskYieldOp>(bodyBlock->getTerminator());

  auto resultTypes = getUnrolledTypes(options);
  assert(resultTypes.size() <= 1 && "expected a single result type");

  MaskOp maskOp;
  if (resultTypes.empty()) {
    maskOp = builder.create<MaskOp>(
        getLoc(), mask,
        [&](OpBuilder &b, Location loc) {
          terminator.unroll(b, options, state);
        },
        maskedoff);
    return maskOp;
  } else {
    maskOp = builder.create<MaskOp>(
        getLoc(), resultTypes[0], mask,
        [&](OpBuilder &b, Location loc) {
          terminator.unroll(b, options, state);
        },
        maskedoff);
    return maskOp;
  }
  return maskOp;
}

std::optional<ShapeVector> MaskOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto mask = getMask();
  return shapeInfo.getShapeWithEquivalence(mask);
}

SmallVector<Type> MaskOp::getUnrolledTypes(UnrollOptions options) {
  auto bodyBlock = &getBody().front();
  assert(bodyBlock->mightHaveTerminator() &&
         "expected the masked region to have a terminator");
  auto terminator = cast<MaskYieldOp>(bodyBlock->getTerminator());
  return terminator.getUnrolledTypes(options);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

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

LogicalResult MatmulOp::verify() {
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.matmul` cannot be nested inside a `zuan.yield`");
  }
  return success();
}

void MatmulOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto lhs = getLhs();
  auto rhs = getRhs();
  auto result = getResult();

  auto lhsShape = *shapeInfo.getShape(lhs);
  auto rhsShape = *shapeInfo.getShape(rhs);

  auto m = lhsShape[lhsShape.size() - 2];
  auto lhsK = lhsShape.back();
  auto rhsK = rhsShape[rhsShape.size() - 2];
  auto n = rhsShape.back();

  auto lhsLeadingDims = lhsShape.take_front(lhsShape.size() - 2);
  auto rhsLeadingDims = rhsShape.take_front(rhsShape.size() - 2);

  shapeInfo.markEquivalent(lhsK, rhsK);
  shapeInfo.markEquivalent(lhsLeadingDims, rhsLeadingDims);

  ShapeVector resultShape(lhsLeadingDims.begin(), lhsLeadingDims.end());
  resultShape.push_back(m);
  resultShape.push_back(n);
  shapeInfo.markEquivalent(result, resultShape);
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *MatmulOp::unroll(OpBuilder &builder, UnrollOptions options,
                            UnrollState &state) {
  auto rank = getResult().getType().getRank();

  UnrollOptions lhsOptions = options;
  UnrollOptions rhsOptions = options;

  auto unrollIdx = options.getUnrollIdx();

  if (unrollIdx == rank - 1) {
    // Unrolling on the rhs last dimension, no need to unroll the lhs.
    auto lhsOptions = options;
    lhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  } else if (unrollIdx == rank - 2) {
    // Unrolling on the lhs second last dimension.
    auto rhsOptions = options;
    rhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  } else if (unrollIdx < rank - 2) {
    // Unrolling the shared leading dims.
  } else {
    // Canonicalize the unroll index.
    options.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  }

  auto lhs = getUnrolledValue(builder, getLhs(), lhsOptions, state);
  auto rhs = getUnrolledValue(builder, getRhs(), rhsOptions, state);

  auto matmulOp = builder.create<MatmulOp>(getLoc(), lhs, rhs);
  return matmulOp;
}

std::optional<ShapeVector> MatmulOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> MatmulOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

LogicalResult ReductionOp::inferReturnTypes(MLIRContext *context,
                                            std::optional<Location> location,
                                            Adaptor adaptor,
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

LogicalResult ReductionOp::verify() {
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError(
        "`zuan.multi_reduction` cannot be nested inside a `zuan.yield`");
  }
  return success();
}

void ReductionOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto tile = getTile();
  auto result = getResult();
  auto dims = getDims();
  SetVector<int64_t> reductionDimSet(dims.begin(), dims.end());

  auto tileShape = *shapeInfo.getShape(tile);
  ShapeVector resultShape;
  for (size_t i = 0, e = tileShape.size(); i < e; ++i) {
    if (!reductionDimSet.contains(i)) {
      resultShape.push_back(tileShape[i]);
    }
  }
  shapeInfo.markEquivalent(result, resultShape);

  auto init = getInit();
  if (init) {
    shapeInfo.markEquivalent(result, init);
  }
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *ReductionOp::unroll(OpBuilder &builder, UnrollOptions options,
                               UnrollState &state) {
  auto resultRank = getResult().getType().getRank();
  SetVector<int64_t> reductionDimSet(getDims().begin(), getDims().end());

  auto unrollIdx = options.getUnrollIdx();
  if (unrollIdx < resultRank) {
    // Compute which dimension in the original tile corresponds to the unrolled
    // dimension in the result.
    unrollIdx = computeUnreducedIdx(
        unrollIdx, getTile().getType().getRank(),
        [&](unsigned dim) { return reductionDimSet.contains(dim); });
  } else {
    // Canonicalize the unroll index.
    options.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  }

  SmallVector<int64_t> newDims(getDims().begin(), getDims().end());
  for (size_t i = 0, e = newDims.size(); i < e; ++i) {
    if (newDims[i] > unrollIdx) {
      // There is a dimension before this that is unrolled, so this minor index
      // needs to be adjusted.
      newDims[i] -= 1;
    }
  }

  UnrollOptions newOptions = options;
  newOptions.overrideUnrollIdx(unrollIdx);

  Value tile = getUnrolledValue(builder, getTile(), newOptions, state);
  Value init = getInit();
  if (init) {
    init = getUnrolledValue(builder, init, options, state);
  }

  auto reductionOp =
      builder.create<ReductionOp>(getLoc(), getKind(), tile, newDims, init);
  return reductionOp;
}

std::optional<ShapeVector> ReductionOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> ReductionOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::inferReturnTypes(MLIRContext *context,
                                       std::optional<Location> location,
                                       Adaptor adaptor,
                                       SmallVectorImpl<Type> &inferred) {
  auto baseType = cast<MemRefType>(adaptor.getBase().getType());
  auto tileType = TileType::get(baseType.getShape(), baseType.getElementType());
  inferred.push_back(tileType);
  return success();
}

LogicalResult LoadOp::verify() {
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.load` cannot be nested inside a `zuan.yield`");
  }
  return success();
}

void LoadOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto base = getBase();
  auto result = getResult();
  ShapeVector shape;
  for (unsigned i = 0, e = base.getType().getRank(); i < e; ++i) {
    shape.push_back(DimSize(base, i));
  }
  shapeInfo.markEquivalent(result, shape);
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *LoadOp::unroll(OpBuilder &builder, UnrollOptions options,
                          UnrollState &state) {
  auto memref = getUnrolledMemref(builder, getBase(), options, state);
  auto loadOp = builder.create<LoadOp>(getLoc(), memref);
  return loadOp;
}

std::optional<ShapeVector> LoadOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> LoadOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto value = getValue();
  auto base = getBase();

  ShapeVector shape;
  for (unsigned i = 0, e = base.getType().getRank(); i < e; ++i) {
    shape.push_back(DimSize(base, i));
  }
  shapeInfo.markEquivalent(value, shape);
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(value, *mask);
  }
}

Operation *StoreOp::unroll(OpBuilder &builder, UnrollOptions options,
                           UnrollState &state) {
  OpBuilder::InsertionGuard guard(builder);

  auto memref = getUnrolledMemref(builder, getBase(), options, state);
  auto value = getUnrolledValue(builder, getValue(), options, state);

  builder.setInsertionPointToEnd(state.yieldBlock);
  auto storeOp = builder.create<StoreOp>(getLoc(), value, memref);
  return storeOp;
}

std::optional<ShapeVector> StoreOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto value = getValue();
  return shapeInfo.getShapeWithEquivalence(value);
}

SmallVector<Type> StoreOp::getUnrolledTypes(UnrollOptions options) {
  return {};
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

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

LogicalResult SplatOp::verify() {
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.splat` cannot be nested inside a `zuan.yield`");
  }
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

void SplatOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto value = getValue();
  auto dims = getMixedDims();

  ShapeVector shape =
      llvm::map_to_vector(dims, [&](OpFoldResult ofr) { return DimSize(ofr); });

  if (auto tileType = dyn_cast<TileType>(value.getType())) {
    auto tileShape = *shapeInfo.getShape(value);
    shape.append(tileShape.begin(), tileShape.end());
  }
  auto result = getResult();
  shapeInfo.markEquivalent(result, shape);
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *SplatOp::unroll(OpBuilder &builder, UnrollOptions options,
                           UnrollState &state) {
  auto dims = getMixedDims();
  auto value = getValue();
  auto resultRank = getResult().getType().getRank();

  auto unrollIdx = options.getUnrollIdx();
  SmallVector<OpFoldResult> newDims;
  if (unrollIdx >= dims.size() && unrollIdx < resultRank) {
    options.overrideUnrollIdx(unrollIdx - dims.size());
    newDims = dims;
  } else {
    for (size_t i = 0, e = dims.size(); i < e; ++i) {
      if (i == unrollIdx) {
        if (!options.shouldReduce()) {
          newDims.push_back(options.getChunkSize());
        }
      } else {
        newDims.push_back(dims[i]);
      }
    }
  }

  if (unrollIdx >= dims.size()) {
    options.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  }

  Value unrolledValue = getUnrolledValue(builder, value, options, state);
  auto splatOp = builder.create<SplatOp>(getLoc(), unrolledValue, newDims);
  return splatOp;
}

std::optional<ShapeVector> SplatOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> SplatOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// OuterOp
//===----------------------------------------------------------------------===//

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
  if (!TileType::isShapeCompatible(lhsShape.take_front(leadingRank),
                                   rhsShape.take_front(leadingRank))) {
    return emitOptionalError(
        location,
        "expected the leading dimensions of the lhs and rhs to be compatible");
  }
  SmallVector<int64_t> resultShape{lhsShape.begin(), lhsShape.end()};
  if (lhsRank <= rhsRank) {
    resultShape.push_back(rhsShape.back());
  }
  inferred.push_back(TileType::get(resultShape, lhsType.getElementType()));
  return success();
}

LogicalResult OuterOp::verify() {
  int64_t lhsRank = getLhs().getType().getRank();
  int64_t rhsRank = getRhs().getType().getRank();
  if (std::abs(lhsRank - rhsRank) > 1) {
    return emitOpError(
        "expected the rank of the lhs and rhs to differ by at most one");
  }
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.outer` cannot be nested inside a `zuan.yield`");
  }
  return success();
}

void OuterOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto lhs = getLhs();
  auto rhs = getRhs();
  auto result = getResult();

  auto lhsShape = *shapeInfo.getShape(lhs);
  auto rhsShape = *shapeInfo.getShape(rhs);

  unsigned leadingRank = lhsShape.size();
  if (lhsShape.size() >= rhsShape.size()) {
    leadingRank -= 1; // Ignore the last dimension.
  }

  ShapeVector resultShape(lhsShape.begin(), lhsShape.end());
  if (lhsShape.size() <= rhsShape.size()) {
    resultShape.push_back(rhsShape.back());
  }
  shapeInfo.markEquivalent(result, resultShape);
  shapeInfo.markEquivalent(lhsShape.take_front(leadingRank),
                           rhsShape.take_front(leadingRank));
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *OuterOp::unroll(OpBuilder &builder, UnrollOptions options,
                           UnrollState &state) {
  auto lhsRank = getLhs().getType().getRank();
  auto rhsRank = getRhs().getType().getRank();
  auto resultRank = getResult().getType().getRank();
  auto leadingRank = lhsRank;
  if (lhsRank >= rhsRank) {
    leadingRank -= 1; // Ignore the last dimension.
  }

  auto unrollIdx = options.getUnrollIdx();

  UnrollOptions lhsOptions = options;
  UnrollOptions rhsOptions = options;

  if (unrollIdx < leadingRank) {
    // Unrolling on shared leading dimensions.
  } else if (unrollIdx < resultRank) {
    // Unrolling on the last two/one dimensions.
    if (lhsRank == rhsRank) {
      if (unrollIdx == resultRank - 1) {
        // Unrolling on rhs last dimension.
        lhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
        rhsOptions.overrideUnrollIdx(unrollIdx - 1);
      } else {
        assert(unrollIdx == resultRank - 2);
        // Unrolling on lhs  last dimension.
        rhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
      }
    } else if (lhsRank < rhsRank) {
      assert(unrollIdx == resultRank - 1);
      // Unrolling on rhs last dimension.
      lhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
    } else if (lhsRank > rhsRank) {
      assert(unrollIdx == resultRank - 1);
      // Unrolling on lhs last dimension.
      rhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
    }
  } else {
    options.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  }

  auto lhs = getUnrolledValue(builder, getLhs(), lhsOptions, state);
  auto rhs = getUnrolledValue(builder, getRhs(), rhsOptions, state);

  auto outerOp = builder.create<OuterOp>(getLoc(), getKind(), lhs, rhs);
  return outerOp;
}

std::optional<ShapeVector> OuterOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> OuterOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// StepOp
//===----------------------------------------------------------------------===//

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

void StepOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto sizes = getMixedSizes();
  auto result = getResult();

  ShapeVector shape = llvm::map_to_vector(
      sizes, [&](OpFoldResult ofr) { return DimSize(ofr); });

  shapeInfo.markEquivalent(result, shape);
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

LogicalResult StepOp::verify() {
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.step` cannot be nested inside a `zuan.yield`");
  }
  return success();
}

Operation *StepOp::unroll(OpBuilder &builder, UnrollOptions options,
                          UnrollState &state) {
  auto dim = getDim().getZExtValue();
  auto sizes = getMixedSizes();
  for (auto &size : sizes) {
    if (auto val = size.dyn_cast<Value>()) {
      size = getUnrolledValue(builder, val, options, state);
    }
  }
  auto start = getUnrolledValue(builder, getStart(), options, state);
  auto unrollIdx = options.getUnrollIdx();

  SmallVector<OpFoldResult> newSizes;
  for (auto [i, size] : llvm::enumerate(sizes)) {
    if (i == unrollIdx) {
      if (!options.shouldReduce()) {
        newSizes.push_back(options.getChunkSize());
      }
    } else {
      newSizes.push_back(size);
    }
  }

  if (dim == unrollIdx) {
    // compute new start and splat.
    Value increment = start;
    if (auto offset = options.getOffset().dyn_cast<Value>()) {
      if (offset.getType() != start.getType()) {
        if (isa<IndexType>(offset.getType()) ||
            isa<IndexType>(start.getType())) {
          offset = builder.create<arith::IndexCastOp>(getLoc(), start.getType(),
                                                      offset);
        } else if (offset.getType().getIntOrFloatBitWidth() >
                   start.getType().getIntOrFloatBitWidth()) {
          offset = builder.create<arith::TruncIOp>(getLoc(), start.getType(),
                                                   offset);
        } else {
          offset =
              builder.create<arith::ExtUIOp>(getLoc(), start.getType(), offset);
        }
      }
      increment = builder.create<arith::AddIOp>(getLoc(), start, offset);
    } else {
      auto offsetInt =
          cast<IntegerAttr>(options.getOffset().dyn_cast<Attribute>()).getInt();
      auto offsetValue = builder.create<arith::ConstantOp>(
          getLoc(), start.getType(),
          builder.getIntegerAttr(start.getType(), offsetInt));
      increment = builder.create<arith::AddIOp>(getLoc(), start, offsetValue);
    }

    auto splatOp = builder.create<SplatOp>(getLoc(), increment, newSizes);
    return splatOp;
  } else {
    if (dim > unrollIdx && options.shouldReduce()) {
      dim -= 1;
    }
    auto stepOp = builder.create<StepOp>(getLoc(), start, dim, newSizes);
    return stepOp;
  }
}

std::optional<ShapeVector> StepOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> StepOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.cast` cannot be nested inside a `zuan.yield`");
  }
  // TODO: Verify the types.
  return success();
}

void CastOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto source = getTile();
  auto result = getResult();
  shapeInfo.markEquivalent(source, result);
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *CastOp::unroll(OpBuilder &builder, UnrollOptions options,
                          UnrollState &state) {
  auto tile = getUnrolledValue(builder, getTile(), options, state);
  auto targetType = getUnrolledTypes(options).front();
  auto castOp = builder.create<CastOp>(getLoc(), targetType, getKind(), tile);
  return castOp;
}

std::optional<ShapeVector> CastOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> CastOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.select` cannot be nested inside a `zuan.yield`");
  }
  return success();
}

void SelectOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto cond = getCond();
  auto lhs = getLhs();
  auto rhs = getRhs();
  auto result = getResult();

  shapeInfo.markEquivalent(lhs, cond);
  shapeInfo.markEquivalent(lhs, rhs);
  shapeInfo.markEquivalent(lhs, result);
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *SelectOp::unroll(OpBuilder &builder, UnrollOptions options,
                            UnrollState &state) {
  auto cond = getUnrolledValue(builder, getCond(), options, state);
  auto lhs = getUnrolledValue(builder, getLhs(), options, state);
  auto rhs = getUnrolledValue(builder, getRhs(), options, state);
  auto selectOp = builder.create<SelectOp>(getLoc(), cond, lhs, rhs);
  return selectOp;
}

std::optional<ShapeVector> SelectOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> SelectOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::inferReturnTypes(MLIRContext *context,
                                         std::optional<Location> location,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type> &inferred) {
  auto baseType = cast<MemRefType>(adaptor.getBase().getType());
  auto indices = adaptor.getIndices();
  auto elementType = baseType.getElementType();
  ArrayRef<int64_t> shape;
  if (!indices.empty()) {
    auto tileType = cast<TileType>(indices.front().getType());
    shape = tileType.getShape();
  }
  inferred.push_back(TileType::get(shape, elementType));
  return success();
}

LogicalResult GatherOp::verify() {
  auto baseType = cast<MemRefType>(getBase().getType());
  auto indices = getIndices();
  if (indices.size() != static_cast<size_t>(baseType.getRank())) {
    return emitOpError("expected the number of indices to be the same as the "
                       "rank of the base memref");
  }
  // The mask op is allowed to be inside the yield op. But only store is allowed
  // to be nested within the yield region.
  if ((*this)->getParentOfType<YieldOp>()) {
    return emitOpError("`zuan.gather` cannot be nested inside a `zuan.yield`");
  }
  // TODO: Verify index shapes.
  return success();
}

void GatherOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto indices = getIndices();
  auto result = getResult();
  if (indices.empty()) {
    shapeInfo.markEquivalent(result, ShapeVector{});
    return;
  }
  shapeInfo.markEquivalent(indices[0], result);
  for (unsigned i = 1, e = indices.size(); i < e; ++i) {
    shapeInfo.markEquivalent(indices[0], indices[i]);
  }
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(result, *mask);
  }
}

Operation *GatherOp::unroll(OpBuilder &builder, UnrollOptions options,
                            UnrollState &state) {
  auto memrefOptions = options;
  memrefOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  auto memref = getUnrolledMemref(builder, getBase(), memrefOptions, state);
  auto indices = getIndices();
  SmallVector<Value> unrolledIndices;
  for (auto index : indices) {
    unrolledIndices.push_back(getUnrolledValue(builder, index, options, state));
  }
  auto gatherOp = builder.create<GatherOp>(getLoc(), memref, unrolledIndices);
  return gatherOp;
}

std::optional<ShapeVector> GatherOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto result = getResult();
  return shapeInfo.getShapeWithEquivalence(result);
}

SmallVector<Type> GatherOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getResult().getType(), options);
  return {type};
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  auto baseType = cast<MemRefType>(getBase().getType());
  auto indices = getIndices();
  if (indices.size() != static_cast<size_t>(baseType.getRank())) {
    return emitOpError("expected the number of indices to be the same as the "
                       "rank of the base memref");
  }
  // TODO: Verify index shapes.
  return success();
}

void ScatterOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  auto value = getValue();
  auto indices = getIndices();
  if (indices.empty()) {
    return;
  }
  shapeInfo.markEquivalent(indices[0], value);
  for (unsigned i = 1, e = indices.size(); i < e; ++i) {
    shapeInfo.markEquivalent(indices[0], indices[i]);
  }
  if (auto mask = state.getMask()) {
    shapeInfo.markEquivalent(value, *mask);
  }
}

Operation *ScatterOp::unroll(OpBuilder &builder, UnrollOptions options,
                             UnrollState &state) {
  OpBuilder::InsertionGuard guard(builder);

  auto memrefOptions = options;
  memrefOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);

  auto memref = getUnrolledMemref(builder, getBase(), memrefOptions, state);
  auto value = getUnrolledValue(builder, getValue(), options, state);
  auto indices = getIndices();
  SmallVector<Value> unrolledIndices;
  for (auto index : indices) {
    unrolledIndices.push_back(getUnrolledValue(builder, index, options, state));
  }

  builder.setInsertionPointToEnd(state.yieldBlock);
  auto scatterOp =
      builder.create<ScatterOp>(getLoc(), value, memref, unrolledIndices);
  return scatterOp;
}

std::optional<ShapeVector> ScatterOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  auto value = getValue();
  return shapeInfo.getShapeWithEquivalence(value);
}

SmallVector<Type> ScatterOp::getUnrolledTypes(UnrollOptions options) {
  return {};
}

//===----------------------------------------------------------------------===//
// MaskYieldOp
//===----------------------------------------------------------------------===//

void MaskYieldOp::inferShape(ShapeInfo &shapeInfo, ShapeInferenceState &state) {
  // No need to infer anything for the mask yield op, as all the operand shapes
  // are inferred.
}

Operation *MaskYieldOp::unroll(OpBuilder &builder, UnrollOptions options,
                               UnrollState &state) {
  auto tile = getUnrolledValue(builder, getTile(), options, state);
  auto maskYieldOp = builder.create<MaskYieldOp>(getLoc(), tile);
  return maskYieldOp;
}

std::optional<ShapeVector> MaskYieldOp::getShapeToUnroll(ShapeInfo &shapeInfo) {
  return std::nullopt;
}

SmallVector<Type> MaskYieldOp::getUnrolledTypes(UnrollOptions options) {
  auto type = getUnrolledTileType(getTile().getType(), options);
  return {type};
}

} // namespace zuan
} // namespace mlir
