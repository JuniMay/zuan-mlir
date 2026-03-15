//===- DynoOps.cpp - Dyno Operations ----------------------------*- C++ -*-===//
//
// This file implements the Dyno operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <cassert>
#include <cstdint>
#include <optional>

#include "Dyno/IR/Dyno.h"
#include "Dyno/Utils/ShapeInference.h"

#define GET_OP_CLASSES
#include "Dyno/IR/DynoOps.cpp.inc"

namespace mlir {
namespace dyno {

static LogicalResult verifyDistinctInRangeDims(Operation *op, int64_t rank,
                                               ArrayRef<int64_t> dims,
                                               StringRef desc) {
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= rank) {
      return op->emitOpError()
             << "expected " << desc << " to be in range [0, " << rank << ")";
    }
    if (!seen.insert(dim).second) {
      return op->emitOpError() << "expected " << desc << " to be unique";
    }
  }
  return success();
}

static LogicalResult verifyGatherScatterIndexShapes(Operation *op,
                                                    ValueRange indices,
                                                    Type expectedType) {
  if (indices.empty()) {
    if (auto expectedTile = dyn_cast<TileType>(expectedType)) {
      if (expectedTile.getRank() != 0) {
        return op->emitOpError(
            "expected scalar result/value when no index tiles are provided");
      }
    }
    return success();
  }

  auto firstType = cast<TileType>(indices.front().getType());
  auto indexShape = firstType.getShape();
  for (Value index : llvm::drop_begin(indices)) {
    auto indexType = cast<TileType>(index.getType());
    if (!TileType::isShapeCompatible(indexShape, indexType.getShape())) {
      return op->emitOpError("expected all index tile shapes to match");
    }
  }

  if (auto expectedTile = dyn_cast<TileType>(expectedType)) {
    if (!TileType::isShapeCompatible(indexShape, expectedTile.getShape())) {
      return op->emitOpError(
          "expected the result/value tile shape to match the index tile shape");
    }
  }

  return success();
}

static LogicalResult verifyDynamicStaticIndexList(Operation *op,
                                                  ArrayRef<int64_t> staticVals,
                                                  ValueRange dynamicVals,
                                                  StringRef desc) {
  unsigned expectedDynamicVals = llvm::count_if(
      staticVals, [](int64_t dim) { return ShapedType::isDynamic(dim); });
  if (dynamicVals.size() != expectedDynamicVals) {
    return op->emitOpError()
           << "expected " << expectedDynamicVals << " dynamic " << desc
           << " operands for " << staticVals.size() << " mixed " << desc;
  }
  return success();
}

static LogicalResult verifyExactTileShape(Operation *op, ArrayRef<int64_t> lhs,
                                          ArrayRef<int64_t> rhs,
                                          StringRef desc) {
  if (lhs != rhs) {
    return op->emitOpError() << "expected " << desc;
  }
  return success();
}

static bool isSignlessIntegerOrIndex(Type type) {
  return isa<IndexType>(type) ||
         (isa<IntegerType>(type) && cast<IntegerType>(type).isSignless());
}

static LogicalResult verifyCastKind(Operation *op, CastKind kind, Type srcType,
                                    Type dstType) {
  auto isSignlessInteger = [](Type type) {
    return isa<IntegerType>(type) && cast<IntegerType>(type).isSignless();
  };
  auto isFloat = [](Type type) { return isa<FloatType>(type); };
  auto isIndex = [](Type type) { return isa<IndexType>(type); };
  auto fail = [&]() {
    return op->emitOpError() << "invalid cast kind " << stringifyEnum(kind)
                             << " for " << srcType << " to " << dstType;
  };
  auto getBitWidth = [](Type type) -> std::optional<unsigned> {
    if (isa<IntegerType, FloatType>(type)) {
      return type.getIntOrFloatBitWidth();
    }
    return std::nullopt;
  };

  switch (kind) {
  case CastKind::BITCAST:
    if (auto srcWidth = getBitWidth(srcType), dstWidth = getBitWidth(dstType);
        srcWidth && dstWidth && *srcWidth == *dstWidth &&
        ((isSignlessInteger(srcType) && isSignlessInteger(dstType)) ||
         (isFloat(srcType) && isFloat(dstType)) ||
         (isSignlessInteger(srcType) && isFloat(dstType)) ||
         (isFloat(srcType) && isSignlessInteger(dstType)))) {
      return success();
    }
    return fail();
  case CastKind::EXTF:
  case CastKind::TRUNCF:
    return (isFloat(srcType) && isFloat(dstType)) ? success() : fail();
  case CastKind::EXTSI:
  case CastKind::EXTUI:
  case CastKind::TRUNCI:
    return (isSignlessInteger(srcType) && isSignlessInteger(dstType))
               ? success()
               : fail();
  case CastKind::FPTOSI:
  case CastKind::FPTOUI:
    return (isFloat(srcType) && isSignlessInteger(dstType)) ? success()
                                                            : fail();
  case CastKind::SITOFP:
  case CastKind::UITOFP:
    return (isSignlessInteger(srcType) && isFloat(dstType)) ? success()
                                                            : fail();
  case CastKind::INDEXCAST:
  case CastKind::INDEXCASTUI:
    return ((isIndex(srcType) && isSignlessInteger(dstType)) ||
            (isSignlessInteger(srcType) && isIndex(dstType)))
               ? success()
               : fail();
  }

  llvm_unreachable("unexpected cast kind");
}

//===----------------------------------------------------------------------===//
// MaskOp
//===----------------------------------------------------------------------===//

void createMaskOpRegion(OpBuilder &builder, Location loc, Operation *maskedOp) {
  if (!maskedOp) {
    MaskYieldOp::create(builder, loc);
    return;
  }
  assert(maskedOp->getBlock() && "maskedOp must be inserted into a block");
  Block *insBlock = builder.getInsertionBlock();
  insBlock->getOperations().splice(
      insBlock->begin(), maskedOp->getBlock()->getOperations(), maskedOp);
  MaskYieldOp::create(builder, maskedOp->getLoc(), maskedOp->getResults());
}

Operation *maskOperation(OpBuilder &builder, Location loc, Operation *maskedOp,
                         Value mask, Value maskedoff) {
  OpBuilder::InsertionGuard guard(builder);

  if (!mask) {
    return maskedOp;
  }
  TypeRange resultTypes{};
  if (maskedOp) {
    // XXX: If set before, the masked op's iterator will be invalidated.
    builder.setInsertionPointAfter(maskedOp);
    resultTypes = maskedOp->getResultTypes();
  }
  return MaskOp::create(
      builder, loc, resultTypes, mask,
      [&](OpBuilder &b, Location loc) { createMaskOpRegion(b, loc, maskedOp); },
      maskedoff);
}

Operation *MaskOp::getMaskedOp() {
  Block *bodyBlock = &getBody().front();
  assert(bodyBlock->mightHaveTerminator() && "expected a terminator");
  auto terminator = cast<MaskYieldOp>(bodyBlock->getTerminator());
  auto maskedOp = &bodyBlock->front();
  if (maskedOp == terminator) {
    return nullptr;
  }
  return maskedOp;
}

void MaskOp::build(OpBuilder &builder, OperationState &result,
                   TypeRange resultTypes, Value mask,
                   function_ref<void(OpBuilder &, Location)> bodyBuilder,
                   Value maskedoff) {
  build(builder, result, mask, bodyBuilder, maskedoff);
  result.addTypes(resultTypes);
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
  if ((*this)->getParentOfType<MaskOp>()) {
    return emitOpError("nested `dyno.mask` is not allowed");
  }
  if (getMask().getType().getElementType() !=
      IntegerType::get(getContext(), 1)) {
    return emitOpError("expected the mask tile element type to be i1");
  }
  if (auto maskedoff = getMaskedoff()) {
    if (getNumResults() == 0) {
      return emitOpError("expected maskedoff only on value-producing masks");
    }
    for (Type resultType : getResultTypes()) {
      if (maskedoff.getType() != resultType) {
        return emitOpError(
            "expected maskedoff tile type to match each result type exactly");
      }
    }
  }
  Block *bodyBlock = &getBody().front();
  if (bodyBlock->empty()) {
    return emitOpError("expected the masked region to have an operation");
  }
  if (bodyBlock->getOperations().size() > 2) {
    return emitOpError(
        "expected only one operation and terminator inside the masked region");
  }
  auto terminator = dyn_cast<MaskYieldOp>(bodyBlock->getTerminator());
  if (!terminator) {
    return emitOpError("expected a `dyno.mask_yield` terminator");
  }
  if (getNumResults() == 0) {
    if (auto maskedOp = getMaskedOp();
        maskedOp && maskedOp->getNumResults() != 0) {
      return emitOpError("expected effect-only masks to contain a masked "
                         "operation with no results");
    }
  }
  if (terminator.getNumOperands() != getNumResults()) {
    return emitOpError(
        "expected the masked region to yield one value per result");
  }
  for (auto [yielded, result] :
       llvm::zip(terminator.getOperands(), getResults())) {
    if (cast<TileType>(result.getType()).getShape() !=
        getMask().getType().getShape()) {
      return emitOpError(
          "expected the mask shape to match each result shape exactly");
    }
    if (yielded.getType() != result.getType()) {
      return emitOpError(
          "expected yielded tile types to match mask result types exactly");
    }
  }

  return success();
}

struct ElideEmptyMaskOp : OpRewritePattern<MaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskOp op,
                                PatternRewriter &rewriter) const override {
    Block *block = &op.getBody().front();

    if (block->getOperations().size() > 1) {
      return rewriter.notifyMatchFailure(op, "expected a single mask yield op");
    }

    auto terminator = cast<MaskYieldOp>(block->getTerminator());
    if (!terminator.getTiles().empty()) {
      rewriter.replaceAllUsesWith(op.getResults(), terminator.getTiles());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void MaskOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<ElideEmptyMaskOp>(context);
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
  if (failed(verifyDistinctInRangeDims((*this), getTile().getType().getRank(),
                                       getDims(), "reduction dims"))) {
    return failure();
  }
  if (auto init = getInit()) {
    if (!TileType::isCompatible(init.getType(), getResult().getType())) {
      return emitOpError(
          "expected init tile type to match the reduction result type");
    }
  }
  return success();
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

LogicalResult LoadOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::build(OpBuilder &builder, OperationState &result, Value tile,
                  int64_t dim) {
  build(builder, result, tile, builder.getI64IntegerAttr(dim));
}

LogicalResult DimOp::verify() {
  auto tileType = getTile().getType();
  int64_t dim = getDim();
  if (dim < 0 || dim >= static_cast<int64_t>(tileType.getRank())) {
    return emitOpError() << "expected dim to be in range [0, "
                         << tileType.getRank() << ")";
  }
  return success();
}

OpFoldResult DimOp::fold(FoldAdaptor adaptor) {
  int64_t staticDim = getTile().getType().getShape()[getDim()];
  if (!ShapedType::isDynamic(staticDim)) {
    return IntegerAttr::get(IndexType::get(getContext()), staticDim);
  }
  return {};
}

namespace {

struct CanonicalizeDynoDim : OpRewritePattern<DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp op,
                                PatternRewriter &rewriter) const override {
    auto reified = reifyDynoDim(rewriter, op.getTile(), op.getDim());
    if (failed(reified)) {
      return failure();
    }
    rewriter.replaceOp(op,
                       getOrCreateIndexValue(rewriter, *reified, op.getLoc()));
    return success();
  }
};

} // namespace

void DimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<CanonicalizeDynoDim>(context);
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> location,
                                          Adaptor adaptor,
                                          SmallVectorImpl<Type> &inferred) {
  inferred.push_back(
      cast<TileType>(adaptor.getTile().getType()).getElementType());
  return success();
}

LogicalResult ExtractOp::verify() {
  if (getTile().getType().getRank() != 0) {
    return emitOpError("expected a rank-0 tile operand");
  }
  return success();
}

OpFoldResult ExtractOp::fold(FoldAdaptor adaptor) {
  if (auto splatOp = getTile().getDefiningOp<SplatOp>()) {
    if (splatOp.getResult().getType().getRank() == 0 &&
        !isa<TileType>(splatOp.getValue().getType())) {
      return splatOp.getValue();
    }
  }
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
  if (failed(verifyDynamicStaticIndexList((*this), getStaticDims(), getDims(),
                                          "dimension"))) {
    return failure();
  }

  auto resultType = getResult().getType();
  unsigned operandRank = 0;
  if (auto operandType = dyn_cast<TileType>(getValue().getType())) {
    operandRank = operandType.getRank();
    if (resultType.getRank() < operandRank) {
      return emitOpError(
          "expected result rank to include the operand tile rank as a suffix");
    }
    if (failed(verifyExactTileShape(
            (*this), resultType.getShape().take_back(operandRank),
            operandType.getShape(),
            "result suffix shape to match the operand tile shape exactly"))) {
      return failure();
    }
  }

  unsigned prefixRank = getStaticDims().size();
  if (resultType.getRank() != prefixRank + operandRank) {
    return emitOpError() << "expected result rank " << resultType.getRank()
                         << " to equal prefix rank " << prefixRank
                         << " plus operand rank " << operandRank;
  }

  for (auto [dim, staticDim] : llvm::enumerate(getStaticDims())) {
    int64_t resultDim = resultType.getShape()[dim];
    if (!TileType::isDimCompatible(resultDim, staticDim)) {
      return emitOpError() << "expected result prefix dimension " << dim
                           << " to match the specified broadcast size";
    }
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

LogicalResult StepOp::verify() {
  if (failed(verifyDynamicStaticIndexList((*this), getStaticSizes(), getSizes(),
                                          "size"))) {
    return failure();
  }

  auto resultType = getResult().getType();
  if (getStaticSizes().size() != resultType.getRank()) {
    return emitOpError("expected one size entry per result dimension");
  }
  int64_t dim = getDim().getSExtValue();
  if (dim < 0 || dim >= static_cast<int64_t>(resultType.getRank())) {
    return emitOpError() << "expected dim to be in range [0, "
                         << resultType.getRank() << ")";
  }
  if (getStart().getType() != resultType.getElementType()) {
    return emitOpError("expected result element type to match the start type");
  }
  if (!isSignlessIntegerOrIndex(resultType.getElementType())) {
    return emitOpError(
        "expected result element type to be signless integer or index");
  }
  for (auto [idx, staticSize] : llvm::enumerate(getStaticSizes())) {
    if (!TileType::isDimCompatible(resultType.getShape()[idx], staticSize)) {
      return emitOpError() << "expected result size " << idx
                           << " to match the specified mixed size list";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto sourceType = getTile().getType();
  auto resultType = getResult().getType();
  if (failed(verifyExactTileShape(
          (*this), sourceType.getShape(), resultType.getShape(),
          "input and result tile shapes to be equal"))) {
    return failure();
  }
  return verifyCastKind((*this), getKind(), sourceType.getElementType(),
                        resultType.getElementType());
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  auto condType = getCond().getType();
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto resultType = getResult().getType();
  if (condType.getElementType() != IntegerType::get(getContext(), 1)) {
    return emitOpError("expected the condition tile element type to be i1");
  }
  if (condType.getShape() != lhsType.getShape() ||
      lhsType.getShape() != rhsType.getShape() ||
      rhsType.getShape() != resultType.getShape()) {
    return emitOpError(
        "expected condition, lhs, rhs, and result shapes to match exactly");
  }
  if (lhsType.getElementType() != rhsType.getElementType() ||
      rhsType.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "expected lhs, rhs, and result element types to match exactly");
  }
  return success();
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
  return verifyGatherScatterIndexShapes((*this), indices,
                                        getResult().getType());
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
  return verifyGatherScatterIndexShapes((*this), indices, getValue().getType());
}

//===----------------------------------------------------------------------===//
// MaskYieldOp
//===----------------------------------------------------------------------===//

} // namespace dyno
} // namespace mlir
