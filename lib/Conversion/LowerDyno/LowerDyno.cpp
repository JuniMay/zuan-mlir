#include "Conversion/LowerDyno.h"

#include "Dyno/IR/Dyno.h"
#include "Dyno/Utils/ShapeInference.h"
#include "Dyno/Utils/Slicing.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace dyno {

namespace {

static FailureOr<Value> materializeTileDim(OpBuilder &builder, Location loc,
                                           Value tile, unsigned dim) {
  auto reified = reifyDynoDim(builder, tile, dim);
  if (failed(reified)) {
    return failure();
  }
  return getOrCreateIndexValue(builder, *reified, loc);
}

static bool isEffectOnlyMaskRoot(MaskOp op) {
  return op.getNumResults() == 0 &&
         isa_and_nonnull<StoreOp, ScatterOp>(op.getMaskedOp());
}

static FailureOr<Operation *> sliceRoot(PatternRewriter &rewriter, Operation *op,
                                        Value tile, unsigned dim,
                                        OpFoldResult offset,
                                        OpFoldResult size, bool dropUnitDim) {
  auto spec = SliceSpec::getSingleDimSlice(rewriter, tile, dim, offset, size,
                                           dropUnitDim);
  if (failed(spec)) {
    return failure();
  }

  SliceState state;
  state.initialize(op);
  return sliceRootOperation(rewriter, op, *spec, state);
}

template <typename RootOp, typename GetTileFn>
struct DynoSliceLeadingDimPattern : OpRewritePattern<RootOp> {
  DynoSliceLeadingDimPattern(MLIRContext *context, unsigned targetRank,
                             GetTileFn getTile)
      : OpRewritePattern<RootOp>(context), targetRank(targetRank),
        getTile(getTile) {}

  LogicalResult matchAndRewrite(RootOp op,
                                PatternRewriter &rewriter) const final {
    Value tile = getTile(op);
    auto tileType = dyn_cast<TileType>(tile.getType());
    if (!tileType || tileType.getRank() <= targetRank) {
      return rewriter.notifyMatchFailure(op, "root already at target rank");
    }

    auto loc = op.getLoc();
    auto leadingDim = materializeTileDim(rewriter, loc, tile, 0);
    if (failed(leadingDim)) {
      return rewriter.notifyMatchFailure(op, "failed to reify leading dimension");
    }

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto loop = scf::ForOp::create(rewriter, loc, zero, *leadingDim, one,
                                   ValueRange{});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      if (failed(sliceRoot(rewriter, op.getOperation(), tile, 0,
                           loop.getInductionVar(), rewriter.getIndexAttr(1),
                           /*dropUnitDim=*/true))) {
        rewriter.eraseOp(loop);
        return rewriter.notifyMatchFailure(op,
                                           "failed to materialize sliced root");
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned targetRank;
  GetTileFn getTile;
};

static Value getStoreTile(StoreOp op) { return op.getValue(); }
static Value getScatterTile(ScatterOp op) { return op.getValue(); }
static Value getMaskTile(MaskOp op) { return op.getMask(); }

struct DynoSliceEffectMaskPattern : OpRewritePattern<MaskOp> {
  DynoSliceEffectMaskPattern(MLIRContext *context, unsigned targetRank)
      : OpRewritePattern<MaskOp>(context), targetRank(targetRank) {}

  LogicalResult matchAndRewrite(MaskOp op,
                                PatternRewriter &rewriter) const final {
    if (!isEffectOnlyMaskRoot(op)) {
      return rewriter.notifyMatchFailure(op, "not an effect-only mask root");
    }
    return DynoSliceLeadingDimPattern<MaskOp, Value (*)(MaskOp)>(
               this->getContext(), targetRank, &getMaskTile)
        .matchAndRewrite(op, rewriter);
  }

private:
  unsigned targetRank;
};

} // namespace

void LowerDynoPass::runOnOperation() {
  RewritePatternSet slicingPatterns(&getContext());
  slicingPatterns
      .add<DynoSliceLeadingDimPattern<StoreOp, Value (*)(StoreOp)>>(
          &getContext(), targetRank, &getStoreTile);
  slicingPatterns
      .add<DynoSliceLeadingDimPattern<ScatterOp, Value (*)(ScatterOp)>>(
          &getContext(), targetRank, &getScatterTile);
  slicingPatterns.add<DynoSliceEffectMaskPattern>(&getContext(), targetRank);

  if (failed(applyPatternsGreedily(getOperation(), std::move(slicingPatterns)))) {
    signalPassFailure();
  }
}

void LowerDynoPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<dyno::DynoDialect, arith::ArithDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
}

void registerLowerDynoPass() { PassRegistration<LowerDynoPass>(); }

} // namespace dyno
} // namespace mlir
