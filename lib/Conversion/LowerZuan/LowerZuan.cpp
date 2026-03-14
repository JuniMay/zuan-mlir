#include "Conversion/LowerZuan.h"

#include "Zuan/IR/Zuan.h"
#include "Zuan/Interfaces/ZuanUnrollingInterface.h"
#include "Zuan/Utils/ShapeInference.h"
#include "Zuan/Utils/Unrolling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace zuan {

namespace {

static FailureOr<Value> materializeTileDim(OpBuilder &builder, Location loc,
                                           Value tile, unsigned dim) {
  auto reified = reifyZuanDim(builder, tile, dim);
  if (failed(reified)) {
    return failure();
  }
  return getOrCreateIndexValue(builder, *reified, loc);
}

static FailureOr<SmallVector<OpFoldResult>>
reifyRequiredShape(OpBuilder &builder, Value value) {
  auto shape = reifyZuanShape(builder, value);
  if (failed(shape)) {
    return failure();
  }
  return *shape;
}

template <typename RootOp, typename GetTileFn>
struct ZuanUnrollLeadingDimPattern : OpRewritePattern<RootOp> {
  ZuanUnrollLeadingDimPattern(MLIRContext *context, unsigned targetRank,
                              GetTileFn getTile)
      : OpRewritePattern<RootOp>(context), targetRank(targetRank),
        getTile(getTile) {}

  LogicalResult matchAndRewrite(RootOp op,
                                PatternRewriter &rewriter) const final {
    Value tile = getTile(op);
    auto tileType = dyn_cast<TileType>(tile.getType());
    if (!tileType || tileType.getRank() <= targetRank) {
      return rewriter.notifyMatchFailure(op, "tile root already small enough");
    }

    auto loc = op.getLoc();
    auto dim = materializeTileDim(rewriter, loc, tile, 0);
    if (failed(dim)) {
      return rewriter.notifyMatchFailure(op, "failed to reify leading dim");
    }

    UnrollState state;
    state.initialize(op);
    Value ub = getUnrolledValue(rewriter, *dim, getCloneOptions(), state);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    scf::ForOp::create(
        rewriter, loc, zero, ub, one, ValueRange{},
        [&](OpBuilder &b, Location loopLoc, Value iv, ValueRange) {
          UnrollOptions options(iv, b.getIndexAttr(1), 0, true);
          op.unroll(b, options, state);
          scf::YieldOp::create(b, loopLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned targetRank;
  GetTileFn getTile;
};

static Value getStoreTile(StoreOp op) { return op.getValue(); }
static Value getScatterTile(ScatterOp op) { return op.getValue(); }

struct ZuanLowerMatmulPattern : OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const final {
    OpBuilder::InsertionGuard guard(rewriter);

    std::optional<std::pair<Value, Value>> masks;
    if (auto maskOp = dyn_cast<MaskOp>(op->getParentOp())) {
      masks = std::make_pair(maskOp.getMask(), maskOp.getMaskedoff());
      rewriter.setInsertionPoint(maskOp);
    } else {
      rewriter.setInsertionPoint(op);
    }

    auto lhsShape = reifyRequiredShape(rewriter, op.getLhs());
    auto rhsShape = reifyRequiredShape(rewriter, op.getRhs());
    auto resultShape = reifyRequiredShape(rewriter, op.getResult());
    if (failed(lhsShape) || failed(rhsShape) || failed(resultShape)) {
      return rewriter.notifyMatchFailure(op, "failed to reify operand shapes");
    }

    auto loc = op.getLoc();
    auto lhsRank = op.getLhs().getType().getRank();
    auto rhsRank = op.getRhs().getType().getRank();
    auto elementType = op.getLhs().getType().getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                           rewriter.getZeroAttr(elementType));

    Value acc = zuan::SplatOp::create(rewriter, loc, zero, *resultShape);
    Value ub = getOrCreateIndexValue(rewriter, lhsShape->back(), loc);
    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto forOp = scf::ForOp::create(
        rewriter, loc, lb, ub, step, acc,
        [&](OpBuilder &b, Location loopLoc, Value iv, ValueRange iterArgs) {
          Value accIter = iterArgs[0];

          UnrollState state;
          state.initialize(op);
          UnrollOptions options(iv, b.getIndexAttr(1), lhsShape->size() - 1,
                                true);
          Value unrolledLhs = getUnrolledValue(b, op.getLhs(), options, state);

          if (rhsRank < lhsRank) {
            options.overrideUnrollIdx(rhsShape->size() - 1);
          } else {
            options.overrideUnrollIdx(rhsShape->size() - 2);
          }
          Value unrolledRhs = getUnrolledValue(b, op.getRhs(), options, state);

          Value res;
          if (masks) {
            auto [mask, maskedoff] = *masks;
            auto type = op.getResult().getType();

            auto outerOp =
                MaskOp::create(b, loopLoc, type, mask,
                               [&](OpBuilder &inner, Location innerLoc) {
                                 Value outer = OuterOp::create(
                                     inner, innerLoc, CombiningKind::MUL,
                                     unrolledLhs, unrolledRhs);
                                 MaskYieldOp::create(inner, innerLoc, outer);
                               });
            Value outer = outerOp.getResult(0);
            auto maskOp = MaskOp::create(
                b, loopLoc, type, mask,
                [&](OpBuilder &inner, Location innerLoc) {
                  Value add = isa<FloatType>(elementType)
                                  ? Value(arith::AddFOp::create(inner, innerLoc,
                                                                accIter, outer))
                                  : Value(arith::AddIOp::create(
                                        inner, innerLoc, accIter, outer));
                  MaskYieldOp::create(inner, innerLoc, add);
                },
                maskedoff);
            res = maskOp.getResult(0);
          } else {
            res = OuterOp::create(b, loopLoc, CombiningKind::MUL, unrolledLhs,
                                  unrolledRhs);
            res = isa<FloatType>(elementType)
                      ? Value(arith::AddFOp::create(b, loopLoc, accIter, res))
                      : Value(arith::AddIOp::create(b, loopLoc, accIter, res));
          }

          scf::YieldOp::create(b, loopLoc, res);
        });

    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

struct ZuanLowerReductionPattern : OpRewritePattern<ReductionOp> {
  using OpRewritePattern<ReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const final {
    SmallVector<int64_t> reductionDims(op.getDims().begin(),
                                       op.getDims().end());

    if (op.getTile().getType().getRank() == 0) {
      return rewriter.notifyMatchFailure(op,
                                         "rank-0 reduction is already flat");
    }
    if (reductionDims.empty()) {
      return rewriter.notifyMatchFailure(op, "no reduction dims");
    }
    if (reductionDims.size() == 1 && reductionDims[0] == 0 &&
        op.getTile().getType().getRank() == 1) {
      return rewriter.notifyMatchFailure(op, "1-D reduction stays for VP path");
    }

    OpBuilder::InsertionGuard guard(rewriter);

    Value mask = nullptr;
    if (auto maskOp = dyn_cast<MaskOp>(op->getParentOp())) {
      mask = maskOp.getMask();
      rewriter.setInsertionPoint(maskOp);
    } else {
      rewriter.setInsertionPoint(op);
    }

    auto sourceShape = reifyRequiredShape(rewriter, op.getTile());
    auto resultShape = reifyRequiredShape(rewriter, op.getResult());
    if (failed(sourceShape) || failed(resultShape)) {
      return rewriter.notifyMatchFailure(op, "failed to reify reduction shape");
    }

    auto loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto elementType = op.getTile().getType().getElementType();
    Value acc = op.getInit();
    if (!acc) {
      Value zeroValue = arith::ConstantOp::create(
          rewriter, loc, elementType, rewriter.getZeroAttr(elementType));
      acc = zuan::SplatOp::create(rewriter, loc, zeroValue, *resultShape);
    }

    llvm::sort(reductionDims.begin(), reductionDims.end(),
               std::greater<int64_t>());

    Value currAcc = acc;
    Value currSource = op.getTile();
    Value currMask = mask;

    SmallVector<scf::ForOp> loops;
    SmallVector<Value> yieldedValues;

    UnrollState state;
    state.initialize(op);

    for (int64_t dim : reductionDims) {
      Value ub = getOrCreateIndexValue(rewriter, (*sourceShape)[dim], loc);
      auto loop = scf::ForOp::create(rewriter, loc, zero, ub, one, currAcc);

      rewriter.setInsertionPointToStart(loop.getBody());
      UnrollOptions options(loop.getInductionVar(), rewriter.getIndexAttr(1),
                            dim, true);
      currSource = getUnrolledValue(rewriter, currSource, options, state);
      if (currMask) {
        currMask = getUnrolledValue(rewriter, currMask, options, state);
      }
      currAcc = loop.getRegionIterArg(0);

      loops.push_back(loop);
      yieldedValues.push_back(loop.getResult(0));
    }

    Value partialReduced =
        createCombiningOp(rewriter, loc, op.getKind(), currAcc, currSource);
    if (currMask) {
      partialReduced =
          maskOperation(rewriter, loc, partialReduced.getDefiningOp(), currMask,
                        currAcc)
              ->getResult(0);
    }
    scf::YieldOp::create(rewriter, loc, partialReduced);
    for (size_t i = 0; i + 1 < loops.size(); ++i) {
      rewriter.setInsertionPointToEnd(loops[i].getBody());
      scf::YieldOp::create(rewriter, loc, yieldedValues[i + 1]);
    }

    rewriter.replaceOp(op, loops.front().getResults());
    return success();
  }
};

struct ZuanUnrollEffectMaskPattern : OpRewritePattern<MaskOp> {
  explicit ZuanUnrollEffectMaskPattern(MLIRContext *context,
                                       unsigned targetRank)
      : OpRewritePattern<MaskOp>(context), targetRank(targetRank) {}

  LogicalResult matchAndRewrite(MaskOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getNumResults() != 0 ||
        !isa_and_nonnull<StoreOp, ScatterOp>(op.getMaskedOp())) {
      return rewriter.notifyMatchFailure(op, "not an effect-only mask root");
    }

    auto maskType = op.getMask().getType();
    if (maskType.getRank() <= targetRank) {
      return rewriter.notifyMatchFailure(op, "mask root already small enough");
    }

    auto loc = op.getLoc();
    auto dim = materializeTileDim(rewriter, loc, op.getMask(), 0);
    if (failed(dim)) {
      return rewriter.notifyMatchFailure(op, "failed to reify mask dim");
    }

    UnrollState state;
    state.initialize(op);
    Value ub = getUnrolledValue(rewriter, *dim, getCloneOptions(), state);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    scf::ForOp::create(
        rewriter, loc, zero, ub, one, ValueRange{},
        [&](OpBuilder &b, Location loopLoc, Value iv, ValueRange) {
          UnrollOptions options(iv, b.getIndexAttr(1), 0, true);
          op.unroll(b, options, state);
          scf::YieldOp::create(b, loopLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned targetRank;
};

} // namespace

void LowerZuanPass::runOnOperation() {
  RewritePatternSet preLoweringPatterns(&getContext());
  preLoweringPatterns.add<ZuanLowerMatmulPattern, ZuanLowerReductionPattern>(
      &getContext());
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(preLoweringPatterns)))) {
    signalPassFailure();
    return;
  }

  RewritePatternSet expansionPatterns(&getContext());
  expansionPatterns
      .add<ZuanUnrollLeadingDimPattern<ScatterOp, Value (*)(ScatterOp)>>(
          &getContext(), targetRank, &getScatterTile);
  expansionPatterns
      .add<ZuanUnrollLeadingDimPattern<StoreOp, Value (*)(StoreOp)>>(
          &getContext(), targetRank, &getStoreTile);
  expansionPatterns.add<ZuanUnrollEffectMaskPattern>(&getContext(), targetRank);

  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(expansionPatterns)))) {
    signalPassFailure();
  }
}

void LowerZuanPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<zuan::ZuanDialect, memref::MemRefDialect, arith::ArithDialect,
                  scf::SCFDialect>();
}

void registerLowerZuanPass() { PassRegistration<LowerZuanPass>(); }

} // namespace zuan
} // namespace mlir
