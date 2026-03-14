//===- ZuanOps.cpp - Zuan Operations ----------------------------*- C++ -*-===//
//
// This file implements the Zuan operations.
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

static LogicalResult verifyDistinctInRangeDims(Operation *op, int64_t rank,
                                               ArrayRef<int64_t> dims,
                                               StringRef desc) {
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= rank) {
      return op->emitOpError() << "expected " << desc
                               << " to be in range [0, " << rank << ")";
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
  return MaskOp::create(builder, 
      loc, resultTypes, mask,
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
    return emitOpError("nested `zuan.mask` is not allowed");
  }
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
  // At most one operation inside.
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
    return emitOpError("expected a `zuan.mask_yield` terminator");
  }
  if (terminator.getNumOperands() != getNumResults()) {
    return emitOpError("expected the masked region to yield one value per result");
  }
  for (auto [yielded, result] : llvm::zip(terminator.getOperands(), getResults())) {
    if (!TileType::isCompatible(yielded.getType(), result.getType())) {
      return emitOpError("expected yielded tile types to match mask result types");
    }
  }

  return success();
}

Operation *MaskOp::unroll(OpBuilder &builder, UnrollOptions options,
                          UnrollState &state) {
  Value mask = getUnrolledValue(builder, getMask(), options, state);
  Value maskedoff = getMaskedoff();
  if (maskedoff) {
    maskedoff = getUnrolledValue(builder, maskedoff, options, state);
  }
  auto maskedOp = getMaskedOp();
  if (maskedOp) {
    maskedOp = unrollOp(builder, maskedOp, options, state);
    return maskOperation(builder, getLoc(), maskedOp, mask, maskedoff);
  } else {
    // Just clone the yield op.
    auto bodyBlock = &getBody().front();
    auto terminator = cast<MaskYieldOp>(bodyBlock->getTerminator());
    auto newOperands =
        llvm::map_to_vector(terminator.getOperands(), [&](Value op) {
          return getUnrolledValue(builder, op, options, state);
        });
    auto newResultTypes = llvm::map_to_vector(
        newOperands, [&](Value opd) { return opd.getType(); });
    return MaskOp::create(builder, 
        getLoc(), newResultTypes, mask,
        [&](OpBuilder &b, Location loc) {
          MaskYieldOp::create(b, loc, newOperands);
        },
        maskedoff);
  }
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

  auto lhsRank = lhsType.getRank();
  auto rhsRank = rhsType.getRank();

  size_t leadingSize;

  if (lhsRank < rhsRank) {
    assert(rhsRank >= 2 && "expected rank >= 2");
    // (1) * k @ k * n => (1) * n
    leadingSize = lhsRank - 1;
  } else if (lhsRank > rhsRank) {
    assert(lhsRank >= 2 && "expected rank >= 2");
    // m * k @ k * (1) => m * (1)
    leadingSize = rhsRank - 1;
  } else {
    assert(lhsRank >= 2 && "expected rank >= 2");
    leadingSize = lhsRank - 2;
  }

  auto lhsK = lhsShape.back();
  auto rhsK = rhsShape[leadingSize];

  if (!TileType::isDimCompatible(lhsK, rhsK)) {
    return emitOptionalError(
        location,
        "expected the inner dimensions of the lhs and rhs to be compatible");
  }

  ArrayRef<int64_t> lhsLeadingDims(lhsShape.begin(), leadingSize);
  ArrayRef<int64_t> rhsLeadingDims(rhsShape.begin(), leadingSize);
  auto resultShape =
      TileType::selectStaticShape(lhsLeadingDims, rhsLeadingDims);
  if (lhsRank >= rhsRank) {
    // M, if exists.
    resultShape.push_back(lhsShape[lhsShape.size() - 2]);
  }
  if (rhsRank >= lhsRank) {
    // N, if exists
    resultShape.push_back(rhsShape.back());
  }

  auto tileType = TileType::get(resultShape, lhsType.getElementType());
  inferred.push_back(tileType);
  return success();
}

LogicalResult MatmulOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  if (lhsType.getRank() < 1 || rhsType.getRank() < 1) {
    return emitOpError("expected lhs and rhs to be at least rank-1 tiles");
  }
  if (std::max(lhsType.getRank(), rhsType.getRank()) < 2) {
    return emitOpError("expected matmul operands to include an inner reduction dimension");
  }
  auto leadingSize = getLeadingSize();
  if (!TileType::isShapeCompatible(lhsType.getShape().take_front(leadingSize),
                                   rhsType.getShape().take_front(leadingSize))) {
    return emitOpError(
        "expected lhs and rhs leading dimensions to be compatible");
  }
  if (!TileType::isDimCompatible(lhsType.getShape().back(),
                                 rhsType.getShape()[leadingSize])) {
    return emitOpError("expected lhs/rhs contraction dimensions to be compatible");
  }
  return success();
}

unsigned MatmulOp::getLeadingSize() {
  auto lhsRank = getLhs().getType().getRank();
  auto rhsRank = getRhs().getType().getRank();

  size_t leadingSize;
  if (lhsRank < rhsRank) {
    assert(rhsRank >= 2 && "expected rank >= 2");
    // (1) * k @ k * n => (1) * n
    leadingSize = lhsRank - 1;
  } else if (lhsRank > rhsRank) {
    assert(lhsRank >= 2 && "expected rank >= 2");
    // m * k @ k * (1) => m * (1)
    leadingSize = rhsRank - 1;
  } else {
    assert(lhsRank >= 2 && "expected rank >= 2");
    leadingSize = lhsRank - 2;
  }
  return leadingSize;
}

Operation *MatmulOp::unroll(OpBuilder &builder, UnrollOptions options,
                            UnrollState &state) {
  UnrollOptions lhsOptions = options;
  UnrollOptions rhsOptions = options;

  size_t leadingSize = getLeadingSize();
  auto lhsRank = getLhs().getType().getRank();
  auto rhsRank = getRhs().getType().getRank();

  auto unrollIdx = options.getUnrollIdx();
  /// If use elementwise mul & reduction to compute the result.
  bool useDot = false;

  if (unrollIdx == leadingSize) {
    if (lhsRank < rhsRank) {
      // (1) * k @ k * n => (1) * n, unrolling on `n`.
      lhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
      rhsOptions.overrideUnrollIdx(rhsRank - 1);
      if (options.shouldReduce()) {
        useDot = true;
      }
    } else if (lhsRank > rhsRank) {
      // m * k @ k * (1) => m * (1), unrolling on `m`.
      lhsOptions.overrideUnrollIdx(lhsRank - 2);
      rhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
      if (options.shouldReduce()) {
        useDot = true;
      }
    } else {
      // result: m * n, unrolling on `m`
      lhsOptions.overrideUnrollIdx(lhsRank - 2);
      rhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
    }
  } else if (unrollIdx == leadingSize + 1) {
    // Unrolling on `n`.
    assert(lhsRank == rhsRank && lhsRank >= 2 &&
           "expected lhs and rhs to have the same rank >= 2");
    lhsOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
    rhsOptions.overrideUnrollIdx(rhsRank - 1);
  } else if (unrollIdx < leadingSize) {
    // Unrolling the shared leading dims.
  } else {
    // Canonicalize the unroll index.
    options.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  }

  auto lhs = getUnrolledValue(builder, getLhs(), lhsOptions, state);
  auto rhs = getUnrolledValue(builder, getRhs(), rhsOptions, state);

  if (useDot) {
    Value mul;
    if (isa<FloatType>(getLhs().getType().getElementType())) {
      mul = arith::MulFOp::create(builder, getLoc(), lhs, rhs);
    } else {
      mul = arith::MulIOp::create(builder, getLoc(), lhs, rhs);
    }
    SmallVector<int64_t> dims{static_cast<int64_t>(leadingSize)};
    auto reduction = ReductionOp::create(builder, getLoc(), CombiningKind::ADD,
                                                 mul, dims, /*init=*/nullptr);
    return reduction;
  } else {
    auto matmulOp = MatmulOp::create(builder, getLoc(), lhs, rhs);
    return matmulOp;
  }
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
      ReductionOp::create(builder, getLoc(), getKind(), tile, newDims, init);
  return reductionOp;
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
  return success();
}

Operation *LoadOp::unroll(OpBuilder &builder, UnrollOptions options,
                          UnrollState &state) {
  auto memref = getUnrolledMemref(builder, getBase(), options, state);
  auto loadOp = LoadOp::create(builder, getLoc(), memref);
  return loadOp;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

Operation *StoreOp::unroll(OpBuilder &builder, UnrollOptions options,
                           UnrollState &state) {
  auto memref = getUnrolledMemref(builder, getBase(), options, state);
  auto value = getUnrolledValue(builder, getValue(), options, state);
  return StoreOp::create(builder, getLoc(), value, memref);
}

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

struct CanonicalizeZuanDim : OpRewritePattern<DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp op,
                                PatternRewriter &rewriter) const override {
    auto reified = reifyZuanDim(rewriter, op.getTile(), op.getDim());
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
  results.add<CanonicalizeZuanDim>(context);
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> location,
                                          Adaptor adaptor,
                                          SmallVectorImpl<Type> &inferred) {
  inferred.push_back(cast<TileType>(adaptor.getTile().getType()).getElementType());
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
        if (auto dimVal = dims[i].dyn_cast<Value>()) {
          newDims.push_back(getUnrolledValue(builder, dimVal, options, state));
        } else {
          newDims.push_back(dims[i]);
        }
      }
    }
  }

  if (unrollIdx < dims.size()) {
    options.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);
  }

  Value unrolledValue = getUnrolledValue(builder, value, options, state);
  auto splatOp = SplatOp::create(builder, getLoc(), unrolledValue, newDims);
  return splatOp;
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
  if (lhsRank == 0 && rhsRank == 0) {
    // Row-splitting before VP can legitimately reduce an outer product all the
    // way to scalar x scalar. Treat that as a rank-0 tile result.
    inferred.push_back(TileType::get({}, lhsType.getElementType()));
    return success();
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
  if (lhsRank == 0 && rhsRank == 0) {
    // See `inferReturnTypes`: scalar x scalar is a valid fully unrolled outer.
    return success();
  }
  unsigned leadingRank = lhsRank;
  if (lhsRank >= rhsRank) {
    leadingRank -= 1;
  }
  if (!TileType::isShapeCompatible(getLhs().getType().getShape().take_front(leadingRank),
                                   getRhs().getType().getShape().take_front(leadingRank))) {
    return emitOpError(
        "expected lhs and rhs leading dimensions to be compatible");
  }
  return success();
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

  auto outerOp = OuterOp::create(builder, getLoc(), getKind(), lhs, rhs);
  return outerOp;
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
          offset = arith::IndexCastOp::create(builder, getLoc(), start.getType(),
                                                      offset);
        } else if (offset.getType().getIntOrFloatBitWidth() >
                   start.getType().getIntOrFloatBitWidth()) {
          offset = arith::TruncIOp::create(builder, getLoc(), start.getType(),
                                                   offset);
        } else {
          offset =
              arith::ExtUIOp::create(builder, getLoc(), start.getType(), offset);
        }
      }
      increment = arith::AddIOp::create(builder, getLoc(), start, offset);
    } else {
      auto offsetInt =
          cast<IntegerAttr>(options.getOffset().dyn_cast<Attribute>()).getInt();
      auto offsetValue = arith::ConstantOp::create(builder, 
          getLoc(), start.getType(),
          builder.getIntegerAttr(start.getType(), offsetInt));
      increment = arith::AddIOp::create(builder, getLoc(), start, offsetValue);
    }

    if (options.shouldReduce()) {
      auto splatOp = SplatOp::create(builder, getLoc(), increment, newSizes);
      return splatOp;
    } else {
      auto stepOp = StepOp::create(builder, getLoc(), increment, dim, newSizes);
      return stepOp;
    }
  } else {
    if (dim > unrollIdx && options.shouldReduce()) {
      dim -= 1;
    }
    auto stepOp = StepOp::create(builder, getLoc(), start, dim, newSizes);
    return stepOp;
  }
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  // TODO: Verify the types.
  return success();
}

Operation *CastOp::unroll(OpBuilder &builder, UnrollOptions options,
                          UnrollState &state) {
  auto tile = getUnrolledValue(builder, getTile(), options, state);
  auto targetType = getUnrolledTileType(getResult().getType(), options);
  auto castOp = CastOp::create(builder, getLoc(), targetType, getKind(), tile);
  return castOp;
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  return success();
}

Operation *SelectOp::unroll(OpBuilder &builder, UnrollOptions options,
                            UnrollState &state) {
  auto cond = getUnrolledValue(builder, getCond(), options, state);
  auto lhs = getUnrolledValue(builder, getLhs(), options, state);
  auto rhs = getUnrolledValue(builder, getRhs(), options, state);
  auto selectOp = SelectOp::create(builder, getLoc(), cond, lhs, rhs);
  return selectOp;
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
  return verifyGatherScatterIndexShapes((*this), indices, getResult().getType());
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
  auto gatherOp = GatherOp::create(builder, getLoc(), memref, unrolledIndices);
  return gatherOp;
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

Operation *ScatterOp::unroll(OpBuilder &builder, UnrollOptions options,
                             UnrollState &state) {
  auto memrefOptions = options;
  memrefOptions.overrideUnrollIdx(UnrollOptions::kNoUnrollIdx);

  auto memref = getUnrolledMemref(builder, getBase(), memrefOptions, state);
  auto value = getUnrolledValue(builder, getValue(), options, state);
  auto indices = getIndices();
  SmallVector<Value> unrolledIndices;
  for (auto index : indices) {
    unrolledIndices.push_back(getUnrolledValue(builder, index, options, state));
  }
  return ScatterOp::create(builder, getLoc(), value, memref, unrolledIndices);
}

//===----------------------------------------------------------------------===//
// MaskYieldOp
//===----------------------------------------------------------------------===//

} // namespace zuan
} // namespace mlir
