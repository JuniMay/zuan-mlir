//===- Slicing.cpp - Dyno structural slicing utilities ---------*- C++ -*-===//
//
// This file implements structural result-domain slicing helpers for Dyno.
//
//===----------------------------------------------------------------------===//

#include "Dyno/Utils/Slicing.h"

#include "Dyno/IR/Dyno.h"
#include "Dyno/Utils/ReductionSemantics.h"
#include "Dyno/Utils/ShapeInference.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace dyno {

namespace {

static FailureOr<SmallVector<OpFoldResult>>
reifyShapedValueShape(OpBuilder &builder, Value value) {
  if (isa<TileType>(value.getType())) {
    return reifyDynoShape(builder, value);
  }
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType) {
    return failure();
  }

  SmallVector<OpFoldResult> shape;
  shape.reserve(memrefType.getRank());
  Location loc = value.getLoc();
  for (auto [idx, dim] : llvm::enumerate(memrefType.getShape())) {
    if (!ShapedType::isDynamic(dim)) {
      shape.push_back(builder.getIndexAttr(dim));
      continue;
    }
    shape.push_back(
        memref::DimOp::create(builder, loc, value, idx).getResult());
  }
  return shape;
}

static Value materializeOffsetAs(Value offset, Type targetType, Location loc,
                                 OpBuilder &builder) {
  if (offset.getType() == targetType) {
    return offset;
  }
  if (isa<IndexType>(offset.getType()) || isa<IndexType>(targetType)) {
    return arith::IndexCastOp::create(builder, loc, targetType, offset);
  }
  if (offset.getType().getIntOrFloatBitWidth() >
      targetType.getIntOrFloatBitWidth()) {
    return arith::TruncIOp::create(builder, loc, targetType, offset);
  }
  return arith::ExtUIOp::create(builder, loc, targetType, offset);
}

static Operation *mapResults(Operation *oldOp, Operation *newOp,
                             SliceState &state) {
  for (auto [oldResult, newResult] :
       llvm::zip(oldOp->getResults(), newOp->getResults())) {
    state.valueMap.map(oldResult, newResult);
  }
  return newOp;
}

static FailureOr<Value>
cloneScalarOrMemrefValue(OpBuilder &builder, Value value, SliceState &state);
static FailureOr<Value> sliceOrCloneNonTileValue(OpBuilder &builder,
                                                 Value value,
                                                 const SliceSpec &spec,
                                                 SliceState &state);
static FailureOr<Operation *> sliceOperation(OpBuilder &builder, Operation *op,
                                             const SliceSpec &spec,
                                             SliceState &state);

static FailureOr<Value> getCurrentMemrefValue(OpBuilder &builder, Value memref,
                                              SliceState &state) {
  if (state.valueMap.contains(memref)) {
    return state.valueMap.lookup(memref);
  }
  if (!memref.getDefiningOp()) {
    return memref;
  }
  return cloneScalarOrMemrefValue(builder, memref, state);
}

static FailureOr<Value>
cloneScalarOrMemrefValue(OpBuilder &builder, Value value, SliceState &state) {
  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }
  if (!value.getDefiningOp()) {
    return value;
  }

  Operation *op = value.getDefiningOp();
  if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
    auto source = cloneScalarOrMemrefValue(builder, dimOp.getSource(), state);
    auto index = cloneScalarOrMemrefValue(builder, dimOp.getIndex(), state);
    if (failed(source) || failed(index)) {
      return failure();
    }
    auto cloned = memref::DimOp::create(builder, op->getLoc(), *source, *index);
    state.valueMap.map(dimOp.getResult(), cloned.getResult());
    return cloned.getResult();
  }

  if (op->getNumRegions() != 0 || !isMemoryEffectFree(op)) {
    return failure();
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    auto mapped = isa<TileType>(operand.getType())
                      ? FailureOr<Value>(failure())
                      : cloneScalarOrMemrefValue(builder, operand, state);
    if (failed(mapped)) {
      return failure();
    }
    newOperands.push_back(*mapped);
  }

  Operation *cloned =
      builder.create(op->getLoc(), op->getName().getIdentifier(), newOperands,
                     op->getResultTypes(), op->getAttrs());
  mapResults(op, cloned, state);
  return cloned->getResult(cast<OpResult>(value).getResultNumber());
}

// Scalar values inside a slice may still depend on sliced memrefs via
// `memref.dim` or arithmetic on those sizes, so they need spec-aware cloning.
static FailureOr<Value> sliceOrCloneNonTileValue(OpBuilder &builder,
                                                 Value value,
                                                 const SliceSpec &spec,
                                                 SliceState &state) {
  if (isa<TileType, MemRefType>(value.getType())) {
    return sliceValue(builder, value, spec, state);
  }
  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }
  if (!value.getDefiningOp()) {
    return value;
  }

  Operation *op = value.getDefiningOp();
  if (isa<memref::DimOp>(op)) {
    auto slicedOp = sliceOperation(builder, op, spec, state);
    if (failed(slicedOp)) {
      return failure();
    }
    if (state.valueMap.contains(value)) {
      return state.valueMap.lookup(value);
    }
    return (*slicedOp)->getResult(cast<OpResult>(value).getResultNumber());
  }

  if (op->getNumRegions() != 0 || !isMemoryEffectFree(op)) {
    return failure();
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    auto mapped = isa<TileType, MemRefType>(operand.getType())
                      ? sliceValue(builder, operand, spec, state)
                      : sliceOrCloneNonTileValue(builder, operand, spec, state);
    if (failed(mapped)) {
      return failure();
    }
    newOperands.push_back(*mapped);
  }

  Operation *cloned =
      builder.create(op->getLoc(), op->getName().getIdentifier(), newOperands,
                     op->getResultTypes(), op->getAttrs());
  mapResults(op, cloned, state);
  return cloned->getResult(cast<OpResult>(value).getResultNumber());
}

static SmallVector<OpFoldResult> getZeroOffsets(OpBuilder &builder,
                                                unsigned rank) {
  return SmallVector<OpFoldResult>(rank, builder.getIndexAttr(0));
}

static FailureOr<Value> slicePointwiseOperand(OpBuilder &builder, Value operand,
                                              const SliceSpec &spec,
                                              SliceState &state) {
  if (isa<TileType>(operand.getType())) {
    return sliceValue(builder, operand, spec, state);
  }
  return sliceOrCloneNonTileValue(builder, operand, spec, state);
}

static FailureOr<Operation *> sliceElementwiseLike(OpBuilder &builder,
                                                   Operation *op,
                                                   const SliceSpec &spec,
                                                   SliceState &state) {
  SmallVector<Value> newOperands;
  for (Value operand : op->getOperands()) {
    auto mapped = slicePointwiseOperand(builder, operand, spec, state);
    if (failed(mapped)) {
      return failure();
    }
    newOperands.push_back(*mapped);
  }

  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(op->getNumResults());
  for (Type resultType : op->getResultTypes()) {
    if (auto tileType = dyn_cast<TileType>(resultType)) {
      newResultTypes.push_back(spec.getSlicedTileType(tileType));
    } else {
      newResultTypes.push_back(resultType);
    }
  }

  Operation *newOp =
      builder.create(op->getLoc(), op->getName().getIdentifier(), newOperands,
                     newResultTypes, op->getAttrs());
  return mapResults(op, newOp, state);
}

static FailureOr<Operation *> sliceSCFForOp(OpBuilder &builder,
                                            scf::ForOp forOp,
                                            const SliceSpec &spec,
                                            SliceState &state) {
  SmallVector<Value> newInits;
  newInits.reserve(forOp.getInitArgs().size());
  for (Value init : forOp.getInitArgs()) {
    auto mapped = isa<TileType>(init.getType())
                      ? sliceValue(builder, init, spec, state)
                      : sliceOrCloneNonTileValue(builder, init, spec, state);
    if (failed(mapped)) {
      return failure();
    }
    newInits.push_back(*mapped);
  }

  auto lb = cloneOrReuseValue(builder, forOp.getLowerBound(), state);
  auto ub = cloneOrReuseValue(builder, forOp.getUpperBound(), state);
  auto step = cloneOrReuseValue(builder, forOp.getStep(), state);
  if (failed(lb) || failed(ub) || failed(step)) {
    return failure();
  }

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), *lb, *ub, *step, newInits);
  if (!newForOp.getBody()->mightHaveTerminator() ||
      newForOp.getBody()->empty()) {
    // Keep the fresh loop body structurally valid before querying its
    // terminator below.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(newForOp.getBody());
    scf::YieldOp::create(builder, forOp.getLoc(), ValueRange{});
  }

  SliceState bodyState;
  bodyState.valueMap = IRMapping(state.valueMap);
  bodyState.valueMap.map(forOp.getInductionVar(), newForOp.getInductionVar());
  bodyState.valueMap.map(forOp.getRegionIterArgs(),
                         newForOp.getRegionIterArgs());

  OpBuilder::InsertionGuard guard(builder);
  auto *oldYield = newForOp.getBody()->getTerminator();
  builder.setInsertionPoint(oldYield);
  SmallVector<Value> newYields;
  for (Value yielded : forOp.getYieldedValues()) {
    auto mapped =
        isa<TileType>(yielded.getType())
            ? sliceValue(builder, yielded, spec, bodyState)
            : sliceOrCloneNonTileValue(builder, yielded, spec, bodyState);
    if (failed(mapped)) {
      newForOp.erase();
      return failure();
    }
    newYields.push_back(*mapped);
  }
  // Materialize the replacement yield before erasing the placeholder so the
  // loop body never transiently lacks an SCF terminator.
  scf::YieldOp::create(builder, forOp.getLoc(), newYields);
  oldYield->erase();
  return mapResults(forOp, newForOp, state);
}

static FailureOr<Operation *> sliceReductionOp(OpBuilder &builder,
                                               ReductionOp reductionOp,
                                               const SliceSpec &resultSpec,
                                               SliceState &state) {
  auto tileType = reductionOp.getTile().getType();
  llvm::SmallDenseSet<int64_t> reducedDims(reductionOp.getDims().begin(),
                                           reductionOp.getDims().end());

  if (resultSpec.getSourceRank() !=
      reductionOp.getResult().getType().getRank()) {
    return failure();
  }

  auto sourceShape = reifyDynoShape(builder, reductionOp.getTile());
  if (failed(sourceShape)) {
    return failure();
  }

  SliceSpec sourceSpec;
  sourceSpec.offsets.reserve(tileType.getRank());
  sourceSpec.sizes.reserve(tileType.getRank());
  sourceSpec.droppedDims.reserve(tileType.getRank());

  unsigned resultDim = 0;
  for (unsigned sourceDim = 0; sourceDim < tileType.getRank(); ++sourceDim) {
    if (reducedDims.contains(sourceDim)) {
      sourceSpec.offsets.push_back(builder.getIndexAttr(0));
      sourceSpec.sizes.push_back((*sourceShape)[sourceDim]);
      sourceSpec.droppedDims.push_back(false);
      continue;
    }
    sourceSpec.offsets.push_back(resultSpec.offsets[resultDim]);
    sourceSpec.sizes.push_back(resultSpec.sizes[resultDim]);
    sourceSpec.droppedDims.push_back(resultSpec.droppedDims[resultDim]);
    ++resultDim;
  }

  SmallVector<int64_t> newDims;
  for (int64_t dim : reductionOp.getDims()) {
    int64_t shift = 0;
    for (int64_t prior = 0; prior < dim; ++prior) {
      shift += sourceSpec.droppedDims[prior] ? 1 : 0;
    }
    newDims.push_back(dim - shift);
  }

  auto tile = sliceValue(builder, reductionOp.getTile(), sourceSpec, state);
  if (failed(tile)) {
    return failure();
  }

  Value init = reductionOp.getInit();
  if (init) {
    auto slicedInit = sliceValue(builder, init, resultSpec, state);
    if (failed(slicedInit)) {
      return failure();
    }
    init = *slicedInit;
  }

  auto newOp = ReductionOp::create(builder, reductionOp.getLoc(),
                                   reductionOp.getKind(), *tile, newDims, init);
  copyReductionFloatingPointPolicy(reductionOp, newOp);
  return mapResults(reductionOp, newOp, state);
}

static FailureOr<SmallVector<Value>>
sliceMaskedInnerOperands(OpBuilder &builder, Operation *maskedOp,
                         const SliceSpec &spec, SliceState &state) {
  SmallVector<Value> operands;
  operands.reserve(maskedOp->getNumOperands());

  for (Value operand : maskedOp->getOperands()) {
    FailureOr<Value> mapped = failure();
    if (auto loadOp = dyn_cast<LoadOp>(maskedOp)) {
      if (operand == loadOp.getBase()) {
        mapped = sliceMemrefView(builder, operand, spec, state);
      }
    } else if (auto storeOp = dyn_cast<StoreOp>(maskedOp)) {
      if (operand == storeOp.getBase()) {
        mapped = sliceMemrefView(builder, operand, spec, state);
      }
    } else if (auto gatherOp = dyn_cast<GatherOp>(maskedOp)) {
      if (operand == gatherOp.getBase()) {
        mapped = cloneOrReuseValue(builder, operand, state);
      }
    } else if (auto scatterOp = dyn_cast<ScatterOp>(maskedOp)) {
      if (operand == scatterOp.getBase()) {
        mapped = cloneOrReuseValue(builder, operand, state);
      }
    }

    if (failed(mapped)) {
      mapped = isa<TileType>(operand.getType())
                   ? sliceValue(builder, operand, spec, state)
                   : sliceOrCloneNonTileValue(builder, operand, spec, state);
    }
    if (failed(mapped)) {
      return failure();
    }
    operands.push_back(*mapped);
  }
  return operands;
}

static Operation *buildMaskedInnerOp(OpBuilder &builder, Operation *maskedOp,
                                     ValueRange operands,
                                     const SliceSpec &spec) {
  if (auto loadOp = dyn_cast<LoadOp>(maskedOp)) {
    return LoadOp::create(builder, loadOp.getLoc(), operands.front());
  }
  if (auto storeOp = dyn_cast<StoreOp>(maskedOp)) {
    return StoreOp::create(builder, storeOp.getLoc(), operands[0], operands[1]);
  }
  if (auto gatherOp = dyn_cast<GatherOp>(maskedOp)) {
    return GatherOp::create(builder, gatherOp.getLoc(), operands.front(),
                            operands.drop_front());
  }
  if (auto scatterOp = dyn_cast<ScatterOp>(maskedOp)) {
    return ScatterOp::create(builder, scatterOp.getLoc(), operands.front(),
                             operands[1], operands.drop_front(2));
  }
  if (auto reductionOp = dyn_cast<ReductionOp>(maskedOp)) {
    llvm::SmallDenseSet<int64_t> reducedDims(reductionOp.getDims().begin(),
                                             reductionOp.getDims().end());
    SmallVector<bool> sourceDropped;
    sourceDropped.reserve(reductionOp.getTile().getType().getRank());
    unsigned resultDim = 0;
    for (unsigned sourceDim = 0;
         sourceDim < reductionOp.getTile().getType().getRank(); ++sourceDim) {
      if (reducedDims.contains(sourceDim)) {
        sourceDropped.push_back(false);
        continue;
      }
      sourceDropped.push_back(spec.droppedDims[resultDim++]);
    }
    SmallVector<int64_t> dims;
    for (int64_t dim : reductionOp.getDims()) {
      int64_t shift = 0;
      for (int64_t prior = 0; prior < dim; ++prior) {
        shift += sourceDropped[prior] ? 1 : 0;
      }
      dims.push_back(dim - shift);
    }
    auto newOp = ReductionOp::create(
        builder, reductionOp.getLoc(), reductionOp.getKind(), operands.front(),
        dims, operands.size() == 2 ? operands[1] : Value());
    copyReductionFloatingPointPolicy(reductionOp, newOp);
    return newOp;
  }
  if (auto splatOp = dyn_cast<SplatOp>(maskedOp)) {
    auto resultType = spec.getSlicedTileType(splatOp.getResult().getType());
    unsigned operandRank = 0;
    if (auto operandType = dyn_cast<TileType>(operands.front().getType())) {
      operandRank = operandType.getRank();
    }
    unsigned prefixRank = resultType.getRank() - operandRank;
    SmallVector<OpFoldResult> prefixDims;
    prefixDims.reserve(prefixRank);
    for (unsigned dim = 0; dim < prefixRank; ++dim) {
      prefixDims.push_back(spec.sizes[dim]);
    }
    return SplatOp::create(builder, splatOp.getLoc(), operands.front(),
                           prefixDims);
  }
  if (auto stepOp = dyn_cast<StepOp>(maskedOp)) {
    Value start = operands.front();
    unsigned dim = stepOp.getDim().getZExtValue();
    if (auto offsetValue = spec.offsets[dim].dyn_cast<Value>()) {
      start = arith::AddIOp::create(
          builder, stepOp.getLoc(), start,
          materializeOffsetAs(offsetValue, start.getType(), stepOp.getLoc(),
                              builder));
    } else if (auto offsetAttr = spec.offsets[dim].dyn_cast<Attribute>()) {
      auto offset = cast<IntegerAttr>(offsetAttr);
      if (offset.getInt() != 0) {
        Value offsetValue = arith::ConstantOp::create(
            builder, stepOp.getLoc(), start.getType(),
            builder.getIntegerAttr(start.getType(), offset.getInt()));
        start =
            arith::AddIOp::create(builder, stepOp.getLoc(), start, offsetValue);
      }
    } else {
      return nullptr;
    }

    SmallVector<OpFoldResult> newSizes;
    for (unsigned idx = 0; idx < spec.getSourceRank(); ++idx) {
      if (!spec.dropsDim(idx)) {
        newSizes.push_back(spec.sizes[idx]);
      }
    }

    if (spec.dropsDim(dim)) {
      return SplatOp::create(builder, stepOp.getLoc(), start, newSizes);
    }

    unsigned newDim = dim;
    for (unsigned idx = 0; idx < dim; ++idx) {
      newDim -= spec.dropsDim(idx) ? 1 : 0;
    }
    return StepOp::create(builder, stepOp.getLoc(), start, newDim, newSizes);
  }
  if (auto castOp = dyn_cast<CastOp>(maskedOp)) {
    return CastOp::create(builder, castOp.getLoc(),
                          spec.getSlicedTileType(castOp.getResult().getType()),
                          castOp.getKind(), operands.front());
  }
  if (auto selectOp = dyn_cast<SelectOp>(maskedOp)) {
    return SelectOp::create(builder, selectOp.getLoc(), operands[0],
                            operands[1], operands[2]);
  }
  if (isa<arith::CmpIOp, arith::CmpFOp>(maskedOp) ||
      maskedOp->hasTrait<OpTrait::Elementwise>()) {
    SmallVector<Type> resultTypes;
    for (Type resultType : maskedOp->getResultTypes()) {
      resultTypes.push_back(
          isa<TileType>(resultType)
              ? Type(spec.getSlicedTileType(cast<TileType>(resultType)))
              : resultType);
    }
    return builder.create(maskedOp->getLoc(),
                          maskedOp->getName().getIdentifier(), operands,
                          resultTypes, maskedOp->getAttrs());
  }
  return nullptr;
}

static bool isSupportedMaskedInnerOp(Operation *maskedOp) {
  return isa<LoadOp, StoreOp, GatherOp, ScatterOp, ReductionOp, SplatOp, StepOp,
             CastOp, SelectOp, arith::CmpIOp, arith::CmpFOp>(maskedOp) ||
         maskedOp->hasTrait<OpTrait::Elementwise>();
}

static FailureOr<Operation *> sliceMaskOp(OpBuilder &builder, MaskOp maskOp,
                                          const SliceSpec &spec,
                                          SliceState &state) {
  auto mask = sliceValue(builder, maskOp.getMask(), spec, state);
  if (failed(mask)) {
    return failure();
  }

  Value maskedoff = maskOp.getMaskedoff();
  if (maskedoff) {
    auto slicedMaskedoff = sliceValue(builder, maskedoff, spec, state);
    if (failed(slicedMaskedoff)) {
      return failure();
    }
    maskedoff = *slicedMaskedoff;
  }

  SmallVector<Type> resultTypes;
  for (Type type : maskOp.getResultTypes()) {
    resultTypes.push_back(spec.getSlicedTileType(cast<TileType>(type)));
  }

  Operation *maskedOp = maskOp.getMaskedOp();
  if (maskedOp && !isSupportedMaskedInnerOp(maskedOp)) {
    return failure();
  }
  SmallVector<Value> slicedMaskedOperands;
  SmallVector<Value> slicedYieldOperands;
  if (maskedOp) {
    auto operands = sliceMaskedInnerOperands(builder, maskedOp, spec, state);
    if (failed(operands)) {
      return failure();
    }
    slicedMaskedOperands = std::move(*operands);
  } else {
    auto yieldOp = cast<MaskYieldOp>(maskOp.getBody().front().getTerminator());
    for (Value operand : yieldOp.getOperands()) {
      auto mapped = sliceValue(builder, operand, spec, state);
      if (failed(mapped)) {
        return failure();
      }
      slicedYieldOperands.push_back(*mapped);
    }
  }

  auto newMask = MaskOp::create(
      builder, maskOp.getLoc(), resultTypes, *mask,
      [&](OpBuilder &bodyBuilder, Location loc) {
        if (maskedOp) {
          Operation *newMaskedOp = buildMaskedInnerOp(
              bodyBuilder, maskedOp, slicedMaskedOperands, spec);
          assert(newMaskedOp &&
                 "pre-validated masked inner op must be sliceable");
          MaskYieldOp::create(bodyBuilder, loc, newMaskedOp->getResults());
          return;
        }

        MaskYieldOp::create(bodyBuilder, loc, slicedYieldOperands);
      },
      maskedoff);
  return mapResults(maskOp, newMask, state);
}

static FailureOr<Operation *> sliceSplatOp(OpBuilder &builder, SplatOp splatOp,
                                           const SliceSpec &spec,
                                           SliceState &state) {
  Value value = splatOp.getValue();
  unsigned operandRank = 0;
  if (auto operandType = dyn_cast<TileType>(value.getType())) {
    operandRank = operandType.getRank();
  }
  if (spec.getSourceRank() != splatOp.getResult().getType().getRank()) {
    return failure();
  }
  unsigned prefixRank = splatOp.getResult().getType().getRank() - operandRank;

  SmallVector<OpFoldResult> newPrefixDims;
  newPrefixDims.reserve(prefixRank);
  for (unsigned dim = 0; dim < prefixRank; ++dim) {
    if (!spec.dropsDim(dim)) {
      newPrefixDims.push_back(spec.sizes[dim]);
    }
  }

  if (operandRank != 0) {
    SliceSpec operandSpec;
    operandSpec.offsets.reserve(operandRank);
    operandSpec.sizes.reserve(operandRank);
    operandSpec.droppedDims.reserve(operandRank);
    for (unsigned dim = 0; dim < operandRank; ++dim) {
      operandSpec.offsets.push_back(spec.offsets[prefixRank + dim]);
      operandSpec.sizes.push_back(spec.sizes[prefixRank + dim]);
      operandSpec.droppedDims.push_back(spec.droppedDims[prefixRank + dim]);
    }
    auto slicedOperand = sliceValue(builder, value, operandSpec, state);
    if (failed(slicedOperand)) {
      return failure();
    }
    value = *slicedOperand;
  } else {
    auto scalar = sliceOrCloneNonTileValue(builder, value, spec, state);
    if (failed(scalar)) {
      return failure();
    }
    value = *scalar;
  }

  auto newOp = SplatOp::create(builder, splatOp.getLoc(), value, newPrefixDims);
  return mapResults(splatOp, newOp, state);
}

static FailureOr<Operation *> sliceStepOp(OpBuilder &builder, StepOp stepOp,
                                          const SliceSpec &spec,
                                          SliceState &state) {
  if (spec.getSourceRank() != stepOp.getResult().getType().getRank()) {
    return failure();
  }
  auto start =
      sliceOrCloneNonTileValue(builder, stepOp.getStart(), spec, state);
  if (failed(start)) {
    return failure();
  }

  unsigned dim = stepOp.getDim().getZExtValue();
  if (auto offsetValue = spec.offsets[dim].dyn_cast<Value>()) {
    auto mappedOffset = cloneOrReuseValue(builder, offsetValue, state);
    if (failed(mappedOffset)) {
      return failure();
    }
    *start = arith::AddIOp::create(
        builder, stepOp.getLoc(), *start,
        materializeOffsetAs(*mappedOffset, (*start).getType(), stepOp.getLoc(),
                            builder));
  } else if (auto offsetAttr = spec.offsets[dim].dyn_cast<Attribute>()) {
    auto offset = cast<IntegerAttr>(offsetAttr);
    if (offset.getInt() != 0) {
      Value offsetValue = arith::ConstantOp::create(
          builder, stepOp.getLoc(), (*start).getType(),
          builder.getIntegerAttr((*start).getType(), offset.getInt()));
      *start =
          arith::AddIOp::create(builder, stepOp.getLoc(), *start, offsetValue);
    }
  } else {
    return failure();
  }

  SmallVector<OpFoldResult> newSizes;
  for (unsigned idx = 0; idx < spec.getSourceRank(); ++idx) {
    if (!spec.dropsDim(idx)) {
      newSizes.push_back(spec.sizes[idx]);
    }
  }

  if (spec.dropsDim(dim)) {
    auto newOp = SplatOp::create(builder, stepOp.getLoc(), *start, newSizes);
    return mapResults(stepOp, newOp, state);
  }

  unsigned newDim = dim;
  for (unsigned idx = 0; idx < dim; ++idx) {
    newDim -= spec.dropsDim(idx) ? 1 : 0;
  }
  auto newOp =
      StepOp::create(builder, stepOp.getLoc(), *start, newDim, newSizes);
  return mapResults(stepOp, newOp, state);
}

static FailureOr<Operation *>
sliceMemrefEffectOp(OpBuilder &builder, Operation *op, Value value, Value base,
                    ValueRange indices, const SliceSpec &spec,
                    SliceState &state) {
  auto slicedBase = sliceMemrefView(builder, base, spec, state);
  if (failed(slicedBase)) {
    return failure();
  }

  auto slicedValue = sliceValue(builder, value, spec, state);
  if (failed(slicedValue)) {
    return failure();
  }

  if (auto storeOp = dyn_cast<StoreOp>(op)) {
    auto newOp =
        StoreOp::create(builder, storeOp.getLoc(), *slicedValue, *slicedBase);
    return mapResults(storeOp, newOp, state);
  }

  SmallVector<Value> slicedIndices;
  for (Value index : indices) {
    auto slicedIndex = sliceValue(builder, index, spec, state);
    if (failed(slicedIndex)) {
      return failure();
    }
    slicedIndices.push_back(*slicedIndex);
  }

  auto newOp = ScatterOp::create(builder, op->getLoc(), *slicedValue,
                                 *slicedBase, slicedIndices);
  return mapResults(op, newOp, state);
}

static FailureOr<Operation *> sliceOperation(OpBuilder &builder, Operation *op,
                                             const SliceSpec &spec,
                                             SliceState &state) {
  if (op->getNumResults() != 0 && state.valueMap.contains(op->getResult(0))) {
    Value mapped = state.valueMap.lookup(op->getResult(0));
    if (auto mappedResult = dyn_cast<OpResult>(mapped)) {
      return mappedResult.getOwner();
    }
    return failure();
  }

  if (auto loadOp = dyn_cast<LoadOp>(op)) {
    auto base = sliceMemrefView(builder, loadOp.getBase(), spec, state);
    if (failed(base)) {
      return failure();
    }
    auto newOp = LoadOp::create(builder, loadOp.getLoc(), *base);
    return mapResults(loadOp, newOp, state);
  }

  if (auto storeOp = dyn_cast<StoreOp>(op)) {
    return sliceMemrefEffectOp(builder, storeOp, storeOp.getValue(),
                               storeOp.getBase(), {}, spec, state);
  }

  if (auto gatherOp = dyn_cast<GatherOp>(op)) {
    auto base = cloneOrReuseValue(builder, gatherOp.getBase(), state);
    if (failed(base)) {
      return failure();
    }
    SmallVector<Value> indices;
    for (Value index : gatherOp.getIndices()) {
      auto slicedIndex = sliceValue(builder, index, spec, state);
      if (failed(slicedIndex)) {
        return failure();
      }
      indices.push_back(*slicedIndex);
    }
    auto newOp = GatherOp::create(builder, gatherOp.getLoc(), *base, indices);
    return mapResults(gatherOp, newOp, state);
  }

  if (auto scatterOp = dyn_cast<ScatterOp>(op)) {
    auto base = cloneOrReuseValue(builder, scatterOp.getBase(), state);
    if (failed(base)) {
      return failure();
    }
    auto value = sliceValue(builder, scatterOp.getValue(), spec, state);
    if (failed(value)) {
      return failure();
    }
    SmallVector<Value> indices;
    for (Value index : scatterOp.getIndices()) {
      auto slicedIndex = sliceValue(builder, index, spec, state);
      if (failed(slicedIndex)) {
        return failure();
      }
      indices.push_back(*slicedIndex);
    }
    auto newOp =
        ScatterOp::create(builder, scatterOp.getLoc(), *value, *base, indices);
    return mapResults(scatterOp, newOp, state);
  }

  if (auto reductionOp = dyn_cast<ReductionOp>(op)) {
    return sliceReductionOp(builder, reductionOp, spec, state);
  }
  if (auto maskOp = dyn_cast<MaskOp>(op)) {
    return sliceMaskOp(builder, maskOp, spec, state);
  }
  if (auto splatOp = dyn_cast<SplatOp>(op)) {
    return sliceSplatOp(builder, splatOp, spec, state);
  }
  if (auto stepOp = dyn_cast<StepOp>(op)) {
    return sliceStepOp(builder, stepOp, spec, state);
  }
  if (auto castOp = dyn_cast<CastOp>(op)) {
    auto tile = sliceValue(builder, castOp.getTile(), spec, state);
    if (failed(tile)) {
      return failure();
    }
    auto resultType = spec.getSlicedTileType(castOp.getResult().getType());
    auto newOp = CastOp::create(builder, castOp.getLoc(), resultType,
                                castOp.getKind(), *tile);
    return mapResults(castOp, newOp, state);
  }
  if (auto selectOp = dyn_cast<SelectOp>(op)) {
    auto cond = sliceValue(builder, selectOp.getCond(), spec, state);
    auto lhs = sliceValue(builder, selectOp.getLhs(), spec, state);
    auto rhs = sliceValue(builder, selectOp.getRhs(), spec, state);
    if (failed(cond) || failed(lhs) || failed(rhs)) {
      return failure();
    }
    auto newOp =
        SelectOp::create(builder, selectOp.getLoc(), *cond, *lhs, *rhs);
    return mapResults(selectOp, newOp, state);
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return sliceSCFForOp(builder, forOp, spec, state);
  }
  if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
    auto source = sliceValue(builder, dimOp.getSource(), spec, state);
    auto index = cloneOrReuseValue(builder, dimOp.getIndex(), state);
    if (failed(source) || failed(index)) {
      return failure();
    }
    auto constantIndex =
        dyn_cast_or_null<arith::ConstantIndexOp>((*index).getDefiningOp());
    if (!constantIndex) {
      auto newOp =
          memref::DimOp::create(builder, dimOp.getLoc(), *source, *index);
      return mapResults(dimOp, newOp, state);
    }
    int64_t dim = constantIndex.value();
    if (spec.dropsDim(dim)) {
      Value newValue =
          getOrCreateIndexValue(builder, spec.sizes[dim], dimOp.getLoc());
      // Keep the remapped value and the returned SSA value identical so the
      // first sliced user observes the dropped-dimension size immediately.
      if (!newValue.getDefiningOp()) {
        Value zero = arith::ConstantIndexOp::create(builder, dimOp.getLoc(), 0);
        newValue =
            arith::AddIOp::create(builder, dimOp.getLoc(), newValue, zero);
      }
      state.valueMap.map(dimOp.getResult(), newValue);
      return newValue.getDefiningOp();
    }
    for (unsigned prior = 0; prior < static_cast<unsigned>(dim); ++prior) {
      dim -= spec.dropsDim(prior) ? 1 : 0;
    }
    auto newOp = memref::DimOp::create(builder, dimOp.getLoc(), *source, dim);
    return mapResults(dimOp, newOp, state);
  }
  if (isa<arith::CmpIOp, arith::CmpFOp>(op) ||
      op->hasTrait<OpTrait::Elementwise>()) {
    return sliceElementwiseLike(builder, op, spec, state);
  }

  if (!isMemoryEffectFree(op)) {
    return failure();
  }

  return sliceElementwiseLike(builder, op, spec, state);
}

} // namespace

unsigned SliceSpec::getResultRank() const {
  return llvm::count(droppedDims, false);
}

FailureOr<SliceSpec> SliceSpec::getIdentity(OpBuilder &builder, Value value) {
  auto shape = reifyShapedValueShape(builder, value);
  if (failed(shape)) {
    return failure();
  }
  SliceSpec spec;
  spec.offsets = getZeroOffsets(builder, shape->size());
  spec.sizes.assign(shape->begin(), shape->end());
  spec.droppedDims.assign(shape->size(), false);
  return spec;
}

FailureOr<SliceSpec> SliceSpec::getSingleDimSlice(OpBuilder &builder,
                                                  Value value, unsigned dim,
                                                  OpFoldResult offset,
                                                  OpFoldResult size,
                                                  bool dropUnitDim) {
  auto spec = getIdentity(builder, value);
  if (failed(spec) || dim >= spec->getSourceRank()) {
    return failure();
  }
  spec->offsets[dim] = offset;
  spec->sizes[dim] = size;
  spec->droppedDims[dim] = dropUnitDim;
  return spec;
}

FailureOr<SliceSpec> SliceSpec::getPrefixSlice(OpBuilder &builder, Value value,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes,
                                               ArrayRef<bool> droppedDims) {
  auto spec = getIdentity(builder, value);
  if (failed(spec) || offsets.size() != sizes.size() ||
      offsets.size() > spec->getSourceRank() ||
      (!droppedDims.empty() && droppedDims.size() != offsets.size())) {
    return failure();
  }

  for (unsigned dim = 0; dim < offsets.size(); ++dim) {
    spec->offsets[dim] = offsets[dim];
    spec->sizes[dim] = sizes[dim];
    spec->droppedDims[dim] = droppedDims.empty() ? false : droppedDims[dim];
  }
  return spec;
}

SmallVector<int64_t> SliceSpec::getSlicedShape(ArrayRef<int64_t> shape) const {
  SmallVector<int64_t> result;
  result.reserve(shape.size());
  for (auto [dim, staticSize] : llvm::enumerate(shape)) {
    if (dropsDim(dim)) {
      continue;
    }
    if (auto attr = sizes[dim].dyn_cast<Attribute>()) {
      result.push_back(cast<IntegerAttr>(attr).getInt());
    } else {
      result.push_back(ShapedType::kDynamic);
    }
  }
  return result;
}

TileType SliceSpec::getSlicedTileType(TileType type) const {
  return TileType::get(getSlicedShape(type.getShape()), type.getElementType());
}

void SliceState::initialize(Operation *root) {
  BackwardSliceOptions options;
  options.inclusive = true;

  SetVector<Operation *> slice;
  (void)getBackwardSlice(root, &slice, options);
  llvm::SmallDenseSet<Operation *> inSlice(slice.begin(), slice.end());
  DominanceInfo dominance(root);

  auto reuseDominatingResult = [&](Value result) {
    if (isa<MemRefType>(result.getType())) {
      valueMap.map(result, result);
      return;
    }
    if (auto tileType = dyn_cast<TileType>(result.getType());
        tileType && tileType.getRank() == 0) {
      valueMap.map(result, result);
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

  scope->walk([&](Operation *op) {
    if (op == root || !dominance.properlyDominates(op, root)) {
      return;
    }
    for (Value result : op->getResults()) {
      reuseDominatingResult(result);
    }
  });
}

FailureOr<Value> cloneOrReuseValue(OpBuilder &builder, Value value,
                                   SliceState &state) {
  return cloneScalarOrMemrefValue(builder, value, state);
}

FailureOr<Value> sliceMemrefView(OpBuilder &builder, Value memref,
                                 const SliceSpec &spec, SliceState &state) {
  auto currentMemref = getCurrentMemrefValue(builder, memref, state);
  if (failed(currentMemref)) {
    return failure();
  }

  auto memrefType = cast<MemRefType>((*currentMemref).getType());
  SmallVector<OpFoldResult> strides(spec.getSourceRank(),
                                    builder.getIndexAttr(1));

  Type unreducedType = memref::SubViewOp::inferResultType(
      memrefType, spec.offsets, spec.sizes, strides);
  auto unreducedMemrefType = cast<MemRefType>(unreducedType);
  auto [unreducedStrides, unreducedOffset] =
      unreducedMemrefType.getStridesAndOffset();

  SmallVector<int64_t> resultShape;
  SmallVector<int64_t> resultStrides;
  for (unsigned dim = 0; dim < spec.getSourceRank(); ++dim) {
    if (spec.dropsDim(dim)) {
      continue;
    }
    resultShape.push_back(unreducedMemrefType.getShape()[dim]);
    resultStrides.push_back(unreducedStrides[dim]);
  }

  auto layout = StridedLayoutAttr::get(builder.getContext(), unreducedOffset,
                                       resultStrides);
  auto resultType = MemRefType::get(resultShape, memrefType.getElementType(),
                                    layout, memrefType.getMemorySpace());
  return memref::SubViewOp::create(builder, memref.getLoc(), resultType,
                                   *currentMemref, spec.offsets, spec.sizes,
                                   strides)
      .getResult();
}

FailureOr<Value> sliceValue(OpBuilder &builder, Value value,
                            const SliceSpec &spec, SliceState &state) {
  // Structural slicing must materialize a sliced memref view even when the
  // current slice state seeds the base memref for reuse.
  if (isa<MemRefType>(value.getType())) {
    return sliceMemrefView(builder, value, spec, state);
  }

  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }

  if (!isa<TileType>(value.getType())) {
    return cloneOrReuseValue(builder, value, state);
  }

  auto opResult = dyn_cast<OpResult>(value);
  if (!opResult) {
    return failure();
  }
  auto newOp = sliceOperation(builder, opResult.getOwner(), spec, state);
  if (failed(newOp)) {
    return failure();
  }
  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }
  return (*newOp)->getResult(opResult.getResultNumber());
}

FailureOr<Operation *> sliceRootOperation(OpBuilder &builder, Operation *root,
                                          const SliceSpec &spec,
                                          SliceState &state) {
  return sliceOperation(builder, root, spec, state);
}

} // namespace dyno
} // namespace mlir
