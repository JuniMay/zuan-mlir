//===- LinalgToDyno.cpp - Linalg to Dyno conversion pass --------*- C++ -*-===//
//
// This file implements the Linalg to Dyno dialect conversion pass.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LinalgToDyno.h"
#include "Dyno/IR/Dyno.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "linalg-to-dyno"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "] ")

using namespace mlir;

namespace {

static AffineMap reindexIndexingMap(AffineMap map) {
  assert(map.isProjectedPermutation(/*allowZeroInResults=*/true) &&
         "expected projected permutation");
  auto res = compressUnusedDims(map);
  assert(res.getNumDims() ==
             (res.getNumResults() - res.getNumOfZeroResults()) &&
         "expected reindexed map with same number of dims and results");
  return res;
}

static Value getOrSplat(OpBuilder &builder, Value value,
                        dyno::LinalgConversionState &state) {
  auto lookup = state.valueMap.lookup(value);
  if (!isa<dyno::TileType>(lookup.getType())) {
    // We do not overwrite the value map here, because we are not sure if this
    // value will be used for other purposes, e.g., as an index in a
    // non-broadcasted op, if such op exists. The most conservative approach is
    // adopted here, which is to create a splat op each time we need a
    // broadcasted value and rely on CSE to clean up.
    return dyno::SplatOp::create(builder, value.getLoc(), lookup,
                                         state.ofrShape);
  }
  return lookup;
}

static LogicalResult convertOneOpToDyno(OpBuilder &builder, Operation *op,
                                        dyno::LinalgConversionState &state) {
  auto loc = op->getLoc();

  SmallVector<Type> resultTypes;
  for (auto resultType : op->getResultTypes()) {
    resultTypes.push_back(dyno::TileType::get(state.staticShape, resultType));
  }

  if (isa<arith::ConstantOp>(op)) {
    // Just clone, not sure if it will be used as scalar in the future.
    builder.clone(*op);
    return success();
  }

  if (auto linalgYieldOp = dyn_cast<linalg::YieldOp>(op)) {
    for (auto [i, yieldedOperand] :
         llvm::enumerate(linalgYieldOp.getOperands())) {
      LLVM_DEBUG(DBGS() << "Yielded operand: " << yieldedOperand << "\n");
      auto mappedOpd = getOrSplat(builder, yieldedOperand, state);
      auto initOpOperand = state.linalgOp.getDpsInitOperand(i);
      auto indexingMap = state.linalgOp.getMatchingIndexingMap(initOpOperand);
      Value linalgInitMemref = initOpOperand->get();
      if (indexingMap.isProjectedPermutation()) {
        // Flat Dyno performs the destination write exactly where the linalg
        // yield happens instead of staging it in a hidden `dyno.yield` region.
        Value transformedInitMemref = state.transformedMemRefs[linalgInitMemref];
        dyno::StoreOp::create(builder, loc, mappedOpd, transformedInitMemref);
      } else {
        // Handle the non-projected permutation operands.
        auto indices = state.nonProjectedPermutationIndices[initOpOperand];
        dyno::ScatterOp::create(builder, loc, mappedOpd, linalgInitMemref,
                                        indices);
      }
    }
    return success();
  }

  //----------------------------------------------------------------------
  // Check Cast Operations.
  //----------------------------------------------------------------------

  if (isa<CastOpInterface>(op)) {
    dyno::CastKind castKind;
    TypeSwitch<Operation *, void>(op)
        .Case([&](arith::IndexCastOp) { castKind = dyno::CastKind::INDEXCAST; })
        .Case([&](arith::IndexCastUIOp) {
          castKind = dyno::CastKind::INDEXCASTUI;
        })
        .Case([&](arith::TruncIOp) { castKind = dyno::CastKind::TRUNCI; })
        .Case([&](arith::TruncFOp) { castKind = dyno::CastKind::TRUNCF; })
        .Case([&](arith::ExtSIOp) { castKind = dyno::CastKind::EXTSI; })
        .Case([&](arith::ExtUIOp) { castKind = dyno::CastKind::EXTUI; })
        .Case([&](arith::FPToSIOp) { castKind = dyno::CastKind::FPTOSI; })
        .Case([&](arith::FPToUIOp) { castKind = dyno::CastKind::FPTOUI; })
        .Case([&](arith::SIToFPOp) { castKind = dyno::CastKind::SITOFP; })
        .Case([&](arith::UIToFPOp) { castKind = dyno::CastKind::UITOFP; })
        .Case([&](arith::BitcastOp) { castKind = dyno::CastKind::BITCAST; })
        .Case([&](arith::ExtFOp) { castKind = dyno::CastKind::EXTF; })
        .Default([&](Operation *) {
          llvm_unreachable("unsupported cast operation");
        });

    auto source = getOrSplat(builder, op->getOperand(0), state);
    auto resultElementType = op->getResult(0).getType();
    auto resultType = dyno::TileType::get(state.staticShape, resultElementType);

    Operation *cast =
        dyno::CastOp::create(builder, loc, resultType, castKind, source);
    if (auto mask = state.getMask()) {
      cast = dyno::maskOperation(builder, loc, cast, *mask);
    }
    state.valueMap.map(op->getResult(0), cast->getResult(0));
    return success();
  }

  //----------------------------------------------------------------------
  // linalg.index operation
  //----------------------------------------------------------------------

  if (auto indexOp = dyn_cast<linalg::IndexOp>(op)) {
    // Index op corresponds to a step op on the given index, with start value 0.
    auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
    int64_t dim = indexOp.getDim();
    Operation *steps =
        dyno::StepOp::create(builder, loc, zero, dim, state.ofrShape);
    if (auto mask = state.getMask()) {
      steps = dyno::maskOperation(builder, loc, steps, *mask);
    }
    state.valueMap.map(op->getResult(0), steps->getResult(0));
    return success();
  }

  //----------------------------------------------------------------------
  // Compare operations.
  //----------------------------------------------------------------------

  if (auto cmpfOp = dyn_cast<arith::CmpFOp>(op)) {
    auto lhs = getOrSplat(builder, op->getOperand(0), state);
    auto rhs = getOrSplat(builder, op->getOperand(1), state);
    Operation *newOp =
        arith::CmpFOp::create(builder, loc, cmpfOp.getPredicate(), lhs, rhs);
    if (auto mask = state.getMask()) {
      newOp = dyno::maskOperation(builder, loc, newOp, *mask);
    }
    state.valueMap.map(op->getResult(0), newOp->getResult(0));
    return success();
  }

  if (auto cmpiOp = dyn_cast<arith::CmpIOp>(op)) {
    auto lhs = getOrSplat(builder, op->getOperand(0), state);
    auto rhs = getOrSplat(builder, op->getOperand(1), state);
    Operation *newOp =
        arith::CmpIOp::create(builder, loc, cmpiOp.getPredicate(), lhs, rhs);
    if (auto mask = state.getMask()) {
      newOp = dyno::maskOperation(builder, loc, newOp, *mask);
    }
    state.valueMap.map(op->getResult(0), newOp->getResult(0));
    return success();
  }

  //----------------------------------------------------------------------
  // Select operation.
  //----------------------------------------------------------------------

  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    auto cond = getOrSplat(builder, op->getOperand(0), state);
    auto lhs = getOrSplat(builder, op->getOperand(1), state);
    auto rhs = getOrSplat(builder, op->getOperand(2), state);
    Operation *newOp = dyno::SelectOp::create(builder, loc, cond, lhs, rhs);
    if (auto mask = state.getMask()) {
      newOp = dyno::maskOperation(builder, loc, newOp, *mask);
    }
    state.valueMap.map(op->getResult(0), newOp->getResult(0));
    return success();
  }

  //----------------------------------------------------------------------
  // Load and store operations.
  //----------------------------------------------------------------------

  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    auto base = state.valueMap.lookup(loadOp.getMemRef());
    auto indices = loadOp.getIndices();
    SmallVector<Value> newIndices;
    for (auto index : indices) {
      newIndices.push_back(getOrSplat(builder, index, state));
    }
    Operation *newOp = dyno::GatherOp::create(builder, loc, base, newIndices);
    if (auto mask = state.getMask()) {
      newOp = dyno::maskOperation(builder, loc, newOp, *mask);
    }
    state.valueMap.map(op->getResult(0), newOp->getResult(0));
    return success();
  }

  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    auto value = getOrSplat(builder, storeOp.getValue(), state);
    auto memref = state.valueMap.lookup(storeOp.getMemRef());
    auto indices = storeOp.getIndices();
    SmallVector<Value> newIndices;
    for (auto index : indices) {
      newIndices.push_back(getOrSplat(builder, index, state));
    }
    // No longer need to reset the insertion point because the ops are flat now.
    Operation *newOp =
        dyno::ScatterOp::create(builder, loc, value, memref, newIndices);
    if (auto mask = state.getMask()) {
      dyno::maskOperation(builder, loc, newOp, *mask);
    }
    return success();
  }

  //----------------------------------------------------------------------
  // If operation.
  //----------------------------------------------------------------------

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // Recursively convert the regions, and merge the yielded values.
    auto cond = getOrSplat(builder, ifOp.getCondition(), state);
    // Negate the condition for the else branch.
    Value trueElem =
        arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                          builder.getBoolAttr(true));
    Value trueValue =
        dyno::SplatOp::create(builder, loc, trueElem, state.ofrShape);
    Value negCond = arith::XOrIOp::create(builder, loc, cond, trueValue);

    if (auto mask = state.getMask()) {
      // Elementwise and the mask.
      cond = arith::AndIOp::create(builder, loc, cond, *mask);
      negCond = arith::AndIOp::create(builder, loc, negCond, *mask);
    }

    state.pushMask(cond);
    for (auto &op : ifOp.getThenRegion().getOps()) {
      if (isa<scf::YieldOp>(op)) {
        continue;
      }
      if (failed(convertOneOpToDyno(builder, &op, state))) {
        return failure();
      }
    }
    state.popMask();

    if (ifOp.getNumRegions() == 2) {
      state.pushMask(negCond);
      for (auto &op : ifOp.getElseRegion().getOps()) {
        if (isa<scf::YieldOp>(op)) {
          continue;
        }
        if (failed(convertOneOpToDyno(builder, &op, state))) {
          return failure();
        }
      }
      state.popMask();
    }

    auto thenYield = ifOp.thenYield();
    auto elseYield = ifOp.elseYield();

    if (ifOp->getResults().empty()) {
      return success();
    }

    for (auto [then, else_, result] :
         llvm::zip(thenYield.getOperands(), elseYield.getOperands(),
                   ifOp->getResults())) {
      auto thenTile = getOrSplat(builder, then, state);
      auto elseTile = getOrSplat(builder, else_, state);
      Operation *op =
          dyno::SelectOp::create(builder, loc, cond, thenTile, elseTile);
      if (auto mask = state.getMask()) {
        op = dyno::maskOperation(builder, loc, op, *mask);
      }
      state.valueMap.map(result, op->getResult(0));
    }

    return success();
  }

  // Otherwise, only elementwise operations are supported.
  if (!OpTrait::hasElementwiseMappableTraits(op)) {
    return failure();
  }

  //----------------------------------------------------------------------
  // Check Reductions.
  //----------------------------------------------------------------------

  SmallVector<std::pair<Value, Value>> reductionOperands;
  for (Value opd : op->getOperands()) {
    auto blockArg = dyn_cast<BlockArgument>(opd);
    if (!blockArg || blockArg.getOwner() != state.linalgOp.getBlock() ||
        blockArg.getArgNumber() < state.linalgOp.getNumDpsInputs()) {
      // The operand is not a reduction operand if it is not a block argument
      // or it is not the argument corresponding to dps inits.
      continue;
    }
    SmallVector<Operation *> reductionOps;
    Value reduceOpd = matchReduction(state.linalgOp.getRegionOutputArgs(),
                                     blockArg.getArgNumber() -
                                         state.linalgOp.getNumDpsInputs(),
                                     reductionOps);
    if (!reduceOpd) {
      continue;
    }
    reductionOperands.push_back({reduceOpd, opd});
  }

  if (!reductionOperands.empty()) {
    assert(reductionOperands.size() == 1);
    auto [reduceOpd, initialOpd] = reductionOperands.front();
    auto reduceTile = getOrSplat(builder, reduceOpd, state);
    auto initialTile = getOrSplat(builder, initialOpd, state);
    SmallVector<int64_t> dimsToReduce;
    for (auto [i, iterType] :
         llvm::enumerate(state.linalgOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iterType)) {
        dimsToReduce.push_back(i);
      }
    }
    auto vectorCombiningKind = linalg::getCombinerOpKind(op);
    if (!vectorCombiningKind.has_value()) {
      return failure();
    }
    auto combiningKind = static_cast<dyno::CombiningKind>(*vectorCombiningKind);
    auto multiReduction = dyno::ReductionOp::create(builder, 
        loc, combiningKind, reduceTile, dimsToReduce, initialTile);
    state.valueMap.map(op->getResult(0), multiReduction.getResult());
    return success();
  }

  //----------------------------------------------------------------------
  // Fallback to elementwise operations.
  //----------------------------------------------------------------------

  SmallVector<Value> tileOperands;
  for (Value opd : op->getOperands()) {
    Value tileOpd = getOrSplat(builder, opd, state);
    tileOperands.push_back(tileOpd);
  }
  // All the operands should share the common shape, so it is ok to build the
  // elementwise operations.
  auto newOp = builder.create(loc, op->getName().getIdentifier(), tileOperands,
                              resultTypes, op->getAttrs());
  if (auto mask = state.getMask()) {
    newOp = dyno::maskOperation(builder, loc, newOp, *mask);
  }
  state.valueMap.map(op->getResults(), newOp->getResults());
  return success();
}

/// Compute a indexing tile used in gather/scatter operations for the dps
/// inputs/inits.
static Value convertIndexingDim(OpBuilder &builder, AffineExpr currExpr,
                                AffineMap indexingMap, unsigned dim,
                                SmallVector<OpFoldResult> ofrShape) {
  auto loc = builder.getUnknownLoc();
  if (auto constExpr = dyn_cast<AffineConstantExpr>(currExpr)) {
    auto cst =
        arith::ConstantIndexOp::create(builder, loc, constExpr.getValue());
    return dyno::SplatOp::create(builder, loc, cst, ofrShape);
  }
  if (auto dimExpr = dyn_cast<AffineDimExpr>(currExpr)) {
    int64_t pos = dimExpr.getPosition();
    auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value step = dyno::StepOp::create(builder, loc, zero, pos, ofrShape);
    return step;
  }
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(currExpr)) {
    auto kind = binExpr.getKind();
    auto lhs = convertIndexingDim(builder, binExpr.getLHS(), indexingMap, dim,
                                  ofrShape);
    auto rhs = convertIndexingDim(builder, binExpr.getRHS(), indexingMap, dim,
                                  ofrShape);

    if (!lhs || !rhs) {
      return nullptr;
    }

    switch (kind) {
    case mlir::AffineExprKind::Add:
      return arith::AddIOp::create(builder, loc, lhs, rhs);
    case mlir::AffineExprKind::Mul:
      return arith::MulIOp::create(builder, loc, lhs, rhs);
    case mlir::AffineExprKind::Mod:
      return arith::RemUIOp::create(builder, loc, lhs, rhs);
    case mlir::AffineExprKind::FloorDiv:
      return arith::FloorDivSIOp::create(builder, loc, lhs, rhs);
    case mlir::AffineExprKind::CeilDiv:
      return arith::CeilDivUIOp::create(builder, loc, lhs, rhs);
    default:
      llvm_unreachable("unexpected affine binary op");
    }
  }

  return nullptr;
}

struct LinalgGenericToDynoPattern : RewritePattern {
  LinalgGenericToDynoPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    //----------------------------------------------------------------------
    // 0. Pre-conditions.
    //----------------------------------------------------------------------

    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(op, "expected linalg op");
    }
    if (!linalgOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op, "expected pure buffer semantics");
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp);
    auto loc = linalgOp.getLoc();

    //----------------------------------------------------------------------
    // 1. Compute the common shape of the linalg op.
    //----------------------------------------------------------------------

    // The static loop ranges of the linalg op.
    SmallVector<int64_t> staticSizes = linalgOp.getStaticLoopRanges();
    // The dynamic loop ranges of the linalg op.
    SmallVector<OpFoldResult> commonShape;
    // The shape of the dps init operand, which is reduced according to the
    // iterator types.
    SmallVector<OpFoldResult> reducedShape;

    for (size_t dimPos = 0; dimPos < linalgOp.getNumLoops(); ++dimPos) {
      auto staticSize = staticSizes[dimPos];
      auto iteratorType = linalgOp.getIteratorTypesArray()[dimPos];
      if (ShapedType::isDynamic(staticSize)) {
        Value operand;
        unsigned operandDimPos;
        if (failed(linalgOp.mapIterationSpaceDimToOperandDim(dimPos, operand,
                                                             operandDimPos))) {
          // TODO: Are there any alternative approaches to handle the iteration
          // space mapping?
          return rewriter.notifyMatchFailure(op,
                                             "unable to get the loop range");
        }
        Value dim = memref::DimOp::create(rewriter, loc, operand, operandDimPos);
        commonShape.push_back(dim);
      } else {
        commonShape.push_back(rewriter.getIndexAttr(staticSize));
      }

      if (!linalg::isReductionIterator(iteratorType)) {
        // This dim is not reduced, so we keep it in the reduced shape.
        reducedShape.push_back(commonShape.back());
      }
    }

    //----------------------------------------------------------------------
    // 2. Transform the input and init memrefs.
    //----------------------------------------------------------------------

    auto linalgOpOperands = linalgOp.getOpOperandsMatchingBBargs();
    // Three things will be mapped here:
    // 1. The values defined above the linalg op.
    // 2. The values defined inside the linalg op.
    IRMapping valueMap;
    // Map the values defined above the linalg op.
    SetVector<Value> valuesDefinedAbove;
    mlir::getUsedValuesDefinedAbove(linalgOp->getRegion(0), valuesDefinedAbove);
    valueMap.map(valuesDefinedAbove.getArrayRef(),
                 valuesDefinedAbove.getArrayRef());
    // The map of shape-transformed memrefs.
    DenseMap<Value, Value> transformedMemRefs;
    /// The operands that are not using a projected permutation.
    SmallVector<OpOperand *> nonProjectedPermutationOperands;

    for (OpOperand *opOperand : linalgOpOperands) {
      auto bbArg = linalgOp.getMatchingBlockArgument(opOperand);
      if (linalgOp.isScalar(opOperand)) {
        valueMap.map(bbArg, opOperand->get());
        continue;
      }
      auto indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
      if (!indexingMap.isProjectedPermutation()) {
        // We need to compute the indices and use `gather` or `scatter` to
        // load/store the values. And this will be handled later in the dynamic
        // op.
        nonProjectedPermutationOperands.push_back(opOperand);
        continue;
      }

      // Handle the projected permutation with broadcasting.
      if (linalgOp.isDpsInput(opOperand)) {
        auto memrefType = cast<MemRefType>(opOperand->get().getType());
        auto memrefShape = memrefType.getShape();

        //----------------------------------------------------------------------
        // 1) Compute the readand permutation maps of this operand.
        //----------------------------------------------------------------------

        // Map from the memref shape into the common shape.
        // e.g. (a, b, c) -> (c, b)
        // ==>  (d0, d1) -> (0, d1, d0)
        //        c   b      a   b   c
        auto readMap = inverseAndBroadcastProjectedPermutation(indexingMap);
        // Get the permuted dims. e.g. above, we want (d0, d1) -> (0, d1, d0)
        // the result is (0, d0, d1) with [0, 2, 1]
        SmallVector<unsigned> permutedDims;
        auto isPermBroadcast =
            readMap.isPermutationOfMinorIdentityWithBroadcasting(permutedDims);
        // The inverse operation should make it minor identity.
        assert(isPermBroadcast && "expected permutation with broadcast");
        auto permutationMap =
            AffineMap::getPermutationMap(permutedDims, rewriter.getContext());

        //----------------------------------------------------------------------
        // 2) Extend the memref shape to the rank of the common shape.
        //----------------------------------------------------------------------

        // We can expand the memref shape in the front and then permute with
        // `transpose`.
        SmallVector<ReassociationIndices> reassociation;
        SmallVector<OpFoldResult> outputShape;
        SmallVector<int64_t> resultShape;

        auto numDimsToExpand =
            linalgOp.getNumLoops() - indexingMap.getNumResults();
        // Build outputShape and resultShape
        for (unsigned i = 0; i < numDimsToExpand; ++i) {
          outputShape.push_back(rewriter.getIndexAttr(1));
          resultShape.push_back(1);
        }
        for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
          int64_t shapeVal = memrefShape[i];
          resultShape.push_back(shapeVal);
          if (ShapedType::isDynamic(shapeVal)) {
            Value dim =
                memref::DimOp::create(rewriter, loc, opOperand->get(), i);
            outputShape.push_back(dim);
          } else {
            outputShape.push_back(rewriter.getIndexAttr(shapeVal));
          }
        }
        // Build reassociation
        ReassociationIndices expandReassociation;
        for (unsigned i = 0; i <= numDimsToExpand; ++i) {
          expandReassociation.push_back(i);
        }
        reassociation.push_back(expandReassociation);
        for (unsigned i = 1; i < indexingMap.getNumResults(); ++i) {
          reassociation.push_back({static_cast<int64_t>(numDimsToExpand + i)});
        }
        Value expanded = memref::ExpandShapeOp::create(rewriter, 
            loc, resultShape, opOperand->get(), reassociation, outputShape);

        //----------------------------------------------------------------------
        // 3) Transpose it to make sure all the dims are at their right place.
        //----------------------------------------------------------------------
        Value transposed = memref::TransposeOp::create(rewriter, 
            loc, expanded, AffineMapAttr::get(permutationMap));

        //----------------------------------------------------------------------
        // 4) Use `subview` to broadcast the dimensions that are not present in
        //    the original memref shape.
        //----------------------------------------------------------------------

        // subview it with all the dynamic dims and 0 stride on broadcast dims.
        SmallVector<OpFoldResult> strides;
        SmallVector<OpFoldResult> offsets(linalgOp.getNumLoops(),
                                          rewriter.getIndexAttr(0));
        // Compute strides. All indexingMap inputs not present in the result are
        // 0, others are 1.
        for (unsigned i = 0; i < linalgOp.getNumLoops(); ++i) {
          auto dimExpr = rewriter.getAffineDimExpr(i);
          if (indexingMap.getResultPosition(dimExpr).has_value()) {
            strides.push_back(rewriter.getIndexAttr(1));
          } else {
            // This dimension will be broadcasted.
            strides.push_back(rewriter.getIndexAttr(0));
          }
        }
        Value subview = memref::SubViewOp::create(rewriter, 
            loc, transposed, offsets, commonShape, strides);
        // Map the original memref to the valueMap.
        transformedMemRefs[opOperand->get()] = subview;
        // bbArgs are not mapped here, dyno.loads will be created inside the
        // dynamic region.
      } else {
        // For output, just transpose it into the source dimension order.
        auto permutationMap =
            inversePermutation(reindexIndexingMap(indexingMap));
        // All the initial reads will be performed on this transposed memref.
        auto transposed = memref::TransposeOp::create(rewriter, 
            loc, opOperand->get(), AffineMapAttr::get(permutationMap));
        // Map the original and transposed memrefs.
        transformedMemRefs[opOperand->get()] = transposed;
      }
    }

    //----------------------------------------------------------------------
    // 3. Materialize tile loads and map the block arguments.
    //----------------------------------------------------------------------
    // The old region wrapper used block arguments as implicit tiles for both
    // inputs and destination-style outputs. In the flat SSA form we replace
    // that protocol with explicit `dyno.load`/`dyno.gather` values up front.
    for (OpOperand *opOperand : linalgOpOperands) {
      if (linalgOp.isScalar(opOperand)) {
        continue;
      }
      auto indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
      if (!indexingMap.isProjectedPermutation()) {
        // Non-projected permutation operands will be handled later.
        continue;
      }

      auto bbArg = linalgOp.getMatchingBlockArgument(opOperand);
      Value source = transformedMemRefs[opOperand->get()];
      Value loaded = dyno::LoadOp::create(rewriter, loc, source);
      valueMap.map(bbArg, loaded);
    }

    dyno::LinalgConversionState state(commonShape, linalgOp);

    // Gather the non-projected permutation operands.
    for (auto opOperand : nonProjectedPermutationOperands) {
      auto indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
      auto bbArg = linalgOp.getMatchingBlockArgument(opOperand);
      SmallVector<Value> indices;
      // Iterate the rhs of the indexing map
      for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
        auto expr = indexingMap.getResult(i);
        Value dim;
        if (linalgOp.isDpsInput(opOperand)) {
          dim = convertIndexingDim(rewriter, expr, indexingMap, i, commonShape);
        } else {
          dim =
              convertIndexingDim(rewriter, expr, indexingMap, i, reducedShape);
        }
        if (!dim) {
          return rewriter.notifyMatchFailure(op, "unable to compute indexing");
        }
        indices.push_back(dim);
      }
      state.nonProjectedPermutationIndices[opOperand] = indices;
      // Gather the values from the memref.
      Value gathered =
          dyno::GatherOp::create(rewriter, loc, opOperand->get(), indices);
      // Map the dps init bbarg to the gathered value.
      valueMap.map(bbArg, gathered);
    }

    state.valueMap = valueMap;
    state.transformedMemRefs = transformedMemRefs;

    //----------------------------------------------------------------------
    // 4. Convert the operations.
    //----------------------------------------------------------------------

    for (auto &op : linalgOp.getBlock()->getOperations()) {
      LLVM_DEBUG(DBGS() << "Converting: " << op.getName() << "\n");
      if (failed(convertOneOpToDyno(rewriter, &op, state))) {
        return failure();
      }
    }
    rewriter.eraseOp(linalgOp);

    return success();
  }
};

} // namespace

namespace mlir {
namespace dyno {

LinalgConversionState::LinalgConversionState(SmallVector<OpFoldResult> ofrShape,
                                             linalg::LinalgOp linalgOp)
    : ofrShape(ofrShape), linalgOp(linalgOp) {
  this->staticShape = llvm::map_to_vector(ofrShape, [](OpFoldResult ofr) {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      return cast<IntegerAttr>(attr).getInt();
    } else {
      return ShapedType::kDynamic;
    }
  });
}

void populateLinalgToDynoConversionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns) {
  patterns.insert<LinalgGenericToDynoPattern>(context);
}

void ConvertLinalgToDynoPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLinalgToDynoConversionPatterns(&getContext(), patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ConvertLinalgToDynoPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<dyno::DynoDialect, arith::ArithDialect, math::MathDialect,
              scf::SCFDialect, memref::MemRefDialect, linalg::LinalgDialect>();
}

void registerConvertLinalgToDynoPass() {
  PassRegistration<ConvertLinalgToDynoPass>();
}

} // namespace dyno
} // namespace mlir
