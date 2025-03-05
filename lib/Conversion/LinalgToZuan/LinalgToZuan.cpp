//===- LinalgToZuan.cpp - Linalg to Zuan conversion pass --------*- C++ -*-===//
//
// This file implements the Linalg to Zuan dialect conversion pass.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LinalgToZuan.h"
#include "Zuan/IR/Zuan.h"
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

#define DEBUG_TYPE "linalg-to-zuan"
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
                        zuan::LinalgConversionState &state) {
  auto lookup = state.valueMap.lookup(value);
  if (!isa<zuan::TileType>(lookup.getType())) {
    // We do not overwrite the value map here, because we are not sure if this
    // value will be used for other purposes, e.g., as an index in a
    // non-broadcasted op, if such op exists. The most conservative approach is
    // adopted here, which is to create a splat op each time we need a
    // broadcasted value and rely on CSE to clean up.
    return builder.create<zuan::SplatOp>(value.getLoc(), lookup,
                                         state.ofrShape);
  }
  return lookup;
}

static LogicalResult convertOneOpToZuan(OpBuilder &builder, Operation *op,
                                        zuan::LinalgConversionState &state) {
  auto loc = op->getLoc();

  SmallVector<Type> resultTypes;
  for (auto resultType : op->getResultTypes()) {
    resultTypes.push_back(zuan::TileType::get(state.staticShape, resultType));
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
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(state.yieldBlock);
      Value linalgInitMemref = state.linalgOp.getDpsInitOperand(i)->get();
      Value dynamicInitMemref = state.valueMap.lookup(linalgInitMemref);
      builder.create<zuan::StoreOp>(loc, mappedOpd, dynamicInitMemref);
    }
    return success();
  }

  //----------------------------------------------------------------------
  // Check Cast Operations.
  //----------------------------------------------------------------------

  if (isa<CastOpInterface>(op)) {
    zuan::CastKind castKind;
    TypeSwitch<Operation *, void>(op)
        .Case([&](arith::IndexCastOp) { castKind = zuan::CastKind::INDEXCAST; })
        .Case([&](arith::IndexCastUIOp) {
          castKind = zuan::CastKind::INDEXCASTUI;
        })
        .Case([&](arith::TruncIOp) { castKind = zuan::CastKind::TRUNCI; })
        .Case([&](arith::TruncFOp) { castKind = zuan::CastKind::TRUNCF; })
        .Case([&](arith::ExtSIOp) { castKind = zuan::CastKind::EXTSI; })
        .Case([&](arith::ExtUIOp) { castKind = zuan::CastKind::EXTUI; })
        .Case([&](arith::FPToSIOp) { castKind = zuan::CastKind::FPTOSI; })
        .Case([&](arith::FPToUIOp) { castKind = zuan::CastKind::FPTOUI; })
        .Case([&](arith::SIToFPOp) { castKind = zuan::CastKind::SITOFP; })
        .Case([&](arith::UIToFPOp) { castKind = zuan::CastKind::UITOFP; })
        .Case([&](arith::BitcastOp) { castKind = zuan::CastKind::BITCAST; })
        .Case([&](arith::ExtFOp) { castKind = zuan::CastKind::EXTF; })
        .Default([&](Operation *) {
          llvm_unreachable("unsupported cast operation");
        });

    auto source = state.valueMap.lookup(op->getOperand(0));
    auto resultElementType = op->getResult(0).getType();
    auto resultType = zuan::TileType::get(state.staticShape, resultElementType);

    Operation *cast =
        builder.create<zuan::CastOp>(loc, resultType, castKind, source);
    if (auto mask = state.getMask()) {
      cast = zuan::maskOperation(builder, loc, cast, *mask);
    }
    state.valueMap.map(op->getResult(0), cast->getResult(0));
    return success();
  }

  //----------------------------------------------------------------------
  // linalg.index operation
  //----------------------------------------------------------------------

  if (auto indexOp = dyn_cast<linalg::IndexOp>(op)) {
    // Index op corresponds to a step op on the given index, with start value 0.
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    int64_t dim = indexOp.getDim();
    Operation *steps =
        builder.create<zuan::StepOp>(loc, zero, dim, state.ofrShape);
    if (auto mask = state.getMask()) {
      steps = zuan::maskOperation(builder, loc, steps, *mask);
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
        builder.create<arith::CmpFOp>(loc, cmpfOp.getPredicate(), lhs, rhs);
    if (auto mask = state.getMask()) {
      newOp = zuan::maskOperation(builder, loc, newOp, *mask);
    }
    state.valueMap.map(op->getResult(0), newOp->getResult(0));
    return success();
  }

  if (auto cmpiOp = dyn_cast<arith::CmpIOp>(op)) {
    auto lhs = getOrSplat(builder, op->getOperand(0), state);
    auto rhs = getOrSplat(builder, op->getOperand(1), state);
    Operation *newOp =
        builder.create<arith::CmpIOp>(loc, cmpiOp.getPredicate(), lhs, rhs);
    if (auto mask = state.getMask()) {
      newOp = zuan::maskOperation(builder, loc, newOp, *mask);
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
    Operation *newOp = builder.create<zuan::SelectOp>(loc, cond, lhs, rhs);
    if (auto mask = state.getMask()) {
      newOp = zuan::maskOperation(builder, loc, newOp, *mask);
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
    Operation *newOp = builder.create<zuan::GatherOp>(loc, base, newIndices);
    if (auto mask = state.getMask()) {
      newOp = zuan::maskOperation(builder, loc, newOp, *mask);
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
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(state.yieldBlock);
    Operation *newOp =
        builder.create<zuan::ScatterOp>(loc, value, memref, newIndices);
    if (auto mask = state.getMask()) {
      zuan::maskOperation(builder, loc, newOp, *mask);
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
        builder.create<arith::ConstantIntOp>(loc, 1, builder.getIntegerType(1));
    Value trueValue =
        builder.create<zuan::SplatOp>(loc, trueElem, state.ofrShape);
    Value negCond = builder.create<arith::XOrIOp>(loc, cond, trueValue);

    if (auto mask = state.getMask()) {
      // Elementwise and the mask.
      cond = builder.create<arith::AndIOp>(loc, cond, *mask);
      negCond = builder.create<arith::AndIOp>(loc, negCond, *mask);
    }

    state.pushMask(cond);
    for (auto &op : ifOp.getThenRegion().getOps()) {
      if (isa<scf::YieldOp>(op)) {
        continue;
      }
      if (failed(convertOneOpToZuan(builder, &op, state))) {
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
        if (failed(convertOneOpToZuan(builder, &op, state))) {
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
          builder.create<zuan::SelectOp>(loc, cond, thenTile, elseTile);
      if (auto mask = state.getMask()) {
        op = zuan::maskOperation(builder, loc, op, *mask);
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
    auto combiningKind = static_cast<zuan::CombiningKind>(*vectorCombiningKind);
    auto multiReduction = builder.create<zuan::ReductionOp>(
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
    newOp = zuan::maskOperation(builder, loc, newOp, *mask);
  }
  state.valueMap.map(op->getResults(), newOp->getResults());
  return success();
}

struct LinalgGenericToZuanPattern : RewritePattern {
  LinalgGenericToZuanPattern(MLIRContext *context)
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

    for (auto map : linalgOp.getIndexingMapsArray()) {
      if (!map.isProjectedPermutation(true)) {
        return rewriter.notifyMatchFailure(op,
                                           "expected projected permutation");
      }
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp);
    auto loc = linalgOp.getLoc();

    //----------------------------------------------------------------------
    // 1. Compute the common shape of the linalg op.
    //----------------------------------------------------------------------

    SmallVector<int64_t> staticSizes = linalgOp.getStaticLoopRanges();
    SmallVector<OpFoldResult> commonShape;
    for (size_t dimPos = 0; dimPos < linalgOp.getNumLoops(); ++dimPos) {
      auto staticSize = staticSizes[dimPos];
      if (ShapedType::isDynamic(staticSize)) {
        Value operand;
        unsigned operandDimPos;
        if (failed(linalgOp.mapIterationSpaceDimToOperandDim(dimPos, operand,
                                                             operandDimPos))) {
          llvm_unreachable("should successfully map iteration space wth "
                           "projected permutation");
        }
        Value dim = rewriter.create<memref::DimOp>(loc, operand, operandDimPos);
        commonShape.push_back(dim);
      } else {
        commonShape.push_back(rewriter.getIndexAttr(staticSize));
      }
    }

    //----------------------------------------------------------------------
    // 2. Transform the input and init memrefs.
    //----------------------------------------------------------------------

    auto linalgOpOperands = linalgOp.getOpOperandsMatchingBBargs();
    // Three things will be mapped here:
    // 1. The values defined above the linalg op.
    // 2. The values defined inside the linalg op.
    // 3. The transformed memrefs for the linalg op.
    IRMapping valueMap;
    // Map the values defined above the linalg op.
    SetVector<Value> valuesDefinedAbove;
    mlir::getUsedValuesDefinedAbove(linalgOp->getRegion(0), valuesDefinedAbove);
    valueMap.map(valuesDefinedAbove.getArrayRef(),
                 valuesDefinedAbove.getArrayRef());
    /// The memrefs for zuan.dynamic op init operands.
    SmallVector<Value> initMemrefs;

    for (OpOperand *opOperand : linalgOpOperands) {
      auto bbArg = linalgOp.getMatchingBlockArgument(opOperand);
      if (linalgOp.isScalar(opOperand)) {
        valueMap.map(bbArg, opOperand->get());
        continue;
      }
      auto indexingMap = linalgOp.getMatchingIndexingMap(opOperand);
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
                rewriter.create<memref::DimOp>(loc, opOperand->get(), i);
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
        Value expanded = rewriter.create<memref::ExpandShapeOp>(
            loc, resultShape, opOperand->get(), reassociation, outputShape);

        //----------------------------------------------------------------------
        // 3) Transpose it to make sure all the dims are at their right place.
        //----------------------------------------------------------------------
        Value transposed = rewriter.create<memref::TransposeOp>(
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
        Value subview = rewriter.create<memref::SubViewOp>(
            loc, transposed, offsets, commonShape, strides);
        // Map the original memref to the valueMap.
        valueMap.map(opOperand->get(), subview);
        // bbArgs are not mapped here, zuan.loads will be created inside the
        // dynamic region.
      } else {
        // For output, just transpose it into the source dimension order.
        auto permutationMap =
            inversePermutation(reindexIndexingMap(indexingMap));
        // All the initial reads will be performed on this transposed memref.
        auto transposed = rewriter.create<memref::TransposeOp>(
            loc, opOperand->get(), AffineMapAttr::get(permutationMap));
        // Map the original and transposed memrefs to the valueMap.
        valueMap.map(opOperand->get(), transposed);
        initMemrefs.push_back(transposed);
        // bbArgs are not mapped here, because zuan.dynamic will build
        // corresponding block arguments.
      }
    }

    //----------------------------------------------------------------------
    // 3. Create the dynamic op and map the bbArgs.
    //----------------------------------------------------------------------

    // The memrefs that are explicitly written to inside the linalg op.
    SetVector<Value> storeMemrefs;
    linalgOp->walk([&](memref::StoreOp storeOp) {
      storeMemrefs.insert(storeOp.getMemRef());
    });
    initMemrefs.append(storeMemrefs.begin(), storeMemrefs.end());

    zuan::YieldOp yieldOp;

    auto dynamicOp = rewriter.create<zuan::DynamicOp>(
        loc, initMemrefs, [&](OpBuilder &b, Location loc, ValueRange inits) {
          // The index into the bbArgs of the dynamic region.
          unsigned initIdx = 0;
          // Map the dps init bbargs with dynamic region inits.
          for (OpOperand *opOperand : linalgOpOperands) {
            if (linalgOp.isScalar(opOperand)) {
              continue;
            }
            auto bbArg = linalgOp.getMatchingBlockArgument(opOperand);
            if (linalgOp.isDpsInput(opOperand)) {
              // Get the mapped subview for the dps input operand.
              Value subview = valueMap.lookup(opOperand->get());
              // Create a load op for the subview.
              Value loaded = b.create<zuan::LoadOp>(loc, subview);
              // Map the dps input bbarg to the loaded value.
              valueMap.map(bbArg, loaded);
            } else {
              // Map the dps init bbarg to the dynamic region init.
              valueMap.map(bbArg, inits[initIdx++]);
            }
          }
          yieldOp = b.create<zuan::YieldOp>(loc);
        });

    // This is the block to insert converted operations inside linalg op.
    auto dynamicBlock = &dynamicOp.getBody().front();
    auto yieldBlock = &yieldOp.getBody().front();

    zuan::LinalgConversionState state(commonShape, dynamicBlock, yieldBlock,
                                      linalgOp);
    state.valueMap = valueMap;

    //----------------------------------------------------------------------
    // 4. Convert the operations.
    //----------------------------------------------------------------------

    // Insert new operations before the yield op.
    rewriter.setInsertionPoint(yieldOp);
    for (auto &op : linalgOp.getBlock()->getOperations()) {
      LLVM_DEBUG(DBGS() << "Converting: " << op.getName() << "\n");
      if (failed(convertOneOpToZuan(rewriter, &op, state))) {
        return failure();
      }
    }
    rewriter.eraseOp(linalgOp);

    return success();
  }
};

} // namespace

namespace mlir {
namespace zuan {

LinalgConversionState::LinalgConversionState(SmallVector<OpFoldResult> ofrShape,
                                             Block *dynamicBlock,
                                             Block *yieldBlock,
                                             linalg::LinalgOp linalgOp)
    : ofrShape(ofrShape), dynamicBlock(dynamicBlock), yieldBlock(yieldBlock),
      linalgOp(linalgOp) {
  this->staticShape = llvm::map_to_vector(ofrShape, [](OpFoldResult ofr) {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      return cast<IntegerAttr>(attr).getInt();
    } else {
      return ShapedType::kDynamic;
    }
  });
}

void populateLinalgToZuanConversionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns) {
  patterns.insert<LinalgGenericToZuanPattern>(context);
}

void ConvertLinalgToZuanPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLinalgToZuanConversionPatterns(&getContext(), patterns);

  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ConvertLinalgToZuanPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<zuan::ZuanDialect, arith::ArithDialect, math::MathDialect,
              scf::SCFDialect, memref::MemRefDialect, linalg::LinalgDialect>();
}

void registerConvertLinalgToZuanPass() {
  PassRegistration<ConvertLinalgToZuanPass>();
}

} // namespace zuan
} // namespace mlir