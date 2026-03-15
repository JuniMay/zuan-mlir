#include "Conversion/DynoToVP.h"

#include "VP/IR/VP.h"
#include "Dyno/IR/Dyno.h"
#include "Dyno/Utils/ConvertToVP.h"
#include "Dyno/Utils/ShapeInference.h"
#include "Dyno/Utils/Unrolling.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace dyno {

namespace {

enum class FloatingPointPolicy {
  Strict,
  Relaxed,
};

enum class ReductionModeKind {
  Auto,
  Sequential,
  Parallel,
};

static std::optional<FloatingPointPolicy>
parseFloatingPointPolicy(StringRef policy) {
  return llvm::StringSwitch<std::optional<FloatingPointPolicy>>(policy)
      .Case("strict", FloatingPointPolicy::Strict)
      .Case("relaxed", FloatingPointPolicy::Relaxed)
      .Default(std::nullopt);
}

static std::optional<ReductionModeKind> parseReductionModeKind(StringRef mode) {
  return llvm::StringSwitch<std::optional<ReductionModeKind>>(mode)
      .Case("auto", ReductionModeKind::Auto)
      .Case("sequential", ReductionModeKind::Sequential)
      .Case("parallel", ReductionModeKind::Parallel)
      .Default(std::nullopt);
}

struct DynoStripminingReduction1DPattern : OpRewritePattern<ReductionOp> {
  DynoStripminingReduction1DPattern(MLIRContext *context, unsigned vf,
                                    bool scalable,
                                    ReductionModeKind reductionMode,
                                    FloatingPointPolicy fpPolicy)
      : OpRewritePattern<ReductionOp>(context, 3), vf(vf), scalable(scalable),
        reductionMode(reductionMode), fpPolicy(fpPolicy) {}

  static LogicalResult isStripminableReduction(ReductionOp op,
                                               RewriterBase &rewriter) {
    if (op->hasAttr("dyno.stripmined")) {
      return failure();
    }
    auto tile = op.getTile();
    if (op.getDims().size() != 1 || op.getDims()[0] != 0 ||
        tile.getType().getRank() != 1) {
      return failure();
    }

    auto dim = reifyDynoDim(rewriter, tile, 0);
    if (failed(dim)) {
      return failure();
    }
    if (auto value = dim->dyn_cast<Value>()) {
      if (isa<vp::GetVLOp>(value.getDefiningOp())) {
        return failure();
      }
    }
    return success();
  }

  static LogicalResult rewriteParallelReduction(ReductionOp op,
                                                RewriterBase &rewriter,
                                                unsigned vf, bool scalable) {
    if (failed(isStripminableReduction(op, rewriter))) {
      return failure();
    }

    auto tile = op.getTile();
    auto dim = reifyDynoDim(rewriter, tile, 0);
    if (failed(dim)) {
      return failure();
    }

    auto loc = op.getLoc();
    Type type = tile.getType().getElementType();
    Value dimValue = getOrCreateIndexValue(rewriter, *dim, loc);
    Value vlmax =
        vp::GetVLOp::create(rewriter, loc, dimValue, vf, type, scalable);

    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value zeroElem = arith::ConstantOp::create(rewriter, loc, type,
                                               rewriter.getZeroAttr(type));
    Value initAcc = dyno::SplatOp::create(rewriter, loc, zeroElem,
                                          SmallVector<OpFoldResult>{vlmax});

    SmallVector<Value> inits = {dimValue, zeroIdx, initAcc};
    SmallVector<Type> resultTypes = {dimValue.getType(), zeroIdx.getType(),
                                     initAcc.getType()};

    auto whileOp = scf::WhileOp::create(
        rewriter, loc, resultTypes, inits,
        [&](OpBuilder &b, Location loopLoc, ValueRange args) {
          Value avl = args[0];
          Value cond = arith::CmpIOp::create(
              b, loopLoc, arith::CmpIPredicate::sgt, avl, zeroIdx);
          scf::ConditionOp::create(b, loopLoc, cond, args);
        },
        [&](OpBuilder &b, Location loopLoc, ValueRange args) {
          Value avl = args[0];
          Value idx = args[1];
          Value acc = args[2];

          Value vl = vp::GetVLOp::create(b, loopLoc, avl, vf, type, scalable);
          UnrollState state;
          state.initialize(op);
          UnrollOptions options(idx, vl, 0, true);
          Value newSourceTile = getUnrolledValue(b, tile, options, state);
          Value newAcc =
              createCombiningOp(b, loopLoc, op.getKind(), acc, newSourceTile);
          if (auto *newAccOp = newAcc.getDefiningOp()) {
            newAccOp->setAttr("dyno_passthru_operand", b.getIndexAttr(0));
          }

          Value newAvl = arith::SubIOp::create(b, loopLoc, avl, vl);
          Value newIdx = arith::AddIOp::create(b, loopLoc, idx, vl);
          scf::YieldOp::create(b, loopLoc, ValueRange{newAvl, newIdx, newAcc});
        });

    Value init = op.getInit();
    if (init) {
      UnrollState state;
      state.initialize(op);
      init = getUnrolledValue(rewriter, init, getCloneOptions(), state);
    }

    Value finalRed = dyno::ReductionOp::create(
        rewriter, loc, op.getKind(), whileOp->getResult(2), op.getDims(), init);
    finalRed.getDefiningOp()->setAttr("dyno.stripmined",
                                      rewriter.getUnitAttr());
    rewriter.replaceOp(op, finalRed);
    return success();
  }

  static LogicalResult rewriteReduction(ReductionOp op, RewriterBase &rewriter,
                                        unsigned vf, bool scalable,
                                        ReductionModeKind reductionMode,
                                        FloatingPointPolicy fpPolicy) {
    (void)reductionMode;
    (void)fpPolicy;
    return rewriteParallelReduction(op, rewriter, vf, scalable);
  }

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const final {
    if (failed(
            rewriteReduction(op, rewriter, vf, scalable, reductionMode,
                             fpPolicy))) {
      return rewriter.notifyMatchFailure(op,
                                         "not a stripminable 1-D reduction");
    }
    return success();
  }

private:
  unsigned vf;
  bool scalable;
  ReductionModeKind reductionMode;
  FloatingPointPolicy fpPolicy;
};

static bool isEffectOnlyMaskRoot(MaskOp op) {
  return op.getNumResults() == 0 &&
         isa_and_nonnull<StoreOp, ScatterOp>(op.getMaskedOp());
}

static Value getRootTileValue(Operation *op);
static void cleanupConvertedSlice(ArrayRef<Operation *> slice,
                                  PatternRewriter &rewriter);

static bool isSupportedEffectRoot(StoreOp) { return true; }
static bool isSupportedEffectRoot(ScatterOp) { return true; }
static bool isSupportedEffectRoot(MaskOp op) {
  return isEffectOnlyMaskRoot(op);
}

static bool isSupportedVPRoot(StoreOp) { return true; }
static bool isSupportedVPRoot(ScatterOp) { return true; }
static bool isSupportedVPRoot(ExtractOp) { return true; }
static bool isSupportedVPRoot(MaskOp op) { return isEffectOnlyMaskRoot(op); }

static SetVector<Operation *> collectBackwardSlice(Operation *root) {
  BackwardSliceOptions options;
  options.inclusive = true;

  SetVector<Operation *> slice;
  (void)getBackwardSlice(root, &slice, options);
  return slice;
}

static bool needsLoopStripmining(Operation *root) {
  // Last-dimension stripmining is the hardware-adaptation step for concrete
  // effect roots. Rank-0 roots never need it; rank-1 and rank-2+ roots both do.
  Value tile = getRootTileValue(root);
  auto tileType = tile ? dyn_cast<TileType>(tile.getType()) : nullptr;
  if (!tileType) {
    return false;
  }
  return TypeSwitch<Operation *, bool>(root)
      .Case<StoreOp, ScatterOp>([&](auto) { return tileType.getRank() >= 1; })
      .Case<MaskOp>([&](MaskOp maskOp) {
        return isEffectOnlyMaskRoot(maskOp) && tileType.getRank() >= 1;
      })
      .Default([](Operation *) { return false; });
}

static Value getRootTileValue(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<StoreOp>([](StoreOp storeOp) { return storeOp.getValue(); })
      .Case<ScatterOp>([](ScatterOp scatterOp) { return scatterOp.getValue(); })
      .Case<MaskOp>([](MaskOp maskOp) { return maskOp.getMask(); })
      .Default([](Operation *) { return Value(); });
}

static Type getRootProcessingType(Operation *op) {
  return TypeSwitch<Operation *, Type>(op)
      .Case<StoreOp>([](StoreOp storeOp) {
        return storeOp.getValue().getType().getElementType();
      })
      .Case<ScatterOp>([](ScatterOp scatterOp) {
        return scatterOp.getValue().getType().getElementType();
      })
      .Case<MaskOp>([](MaskOp maskOp) -> Type {
        return TypeSwitch<Operation *, Type>(maskOp.getMaskedOp())
            .Case<StoreOp>([](StoreOp storeOp) {
              return storeOp.getValue().getType().getElementType();
            })
            .Case<ScatterOp>([](ScatterOp scatterOp) {
              return scatterOp.getValue().getType().getElementType();
            })
            .Default([](Operation *) { return Type(); });
      })
      .Default([](Operation *) { return Type(); });
}

static FailureOr<SmallVector<OpFoldResult>> reifyRootShape(OpBuilder &builder,
                                                           Operation *op) {
  Value tile = getRootTileValue(op);
  if (!tile) {
    return failure();
  }
  return reifyDynoShape(builder, tile);
}

static Operation *unrollRoot(OpBuilder &builder, Operation *op,
                             UnrollOptions options, UnrollState &state) {
  return TypeSwitch<Operation *, Operation *>(op)
      .Case<StoreOp>([&](StoreOp storeOp) {
        return storeOp.unroll(builder, options, state);
      })
      .Case<ScatterOp>([&](ScatterOp scatterOp) {
        return scatterOp.unroll(builder, options, state);
      })
      .Case<MaskOp>(
          [&](MaskOp maskOp) { return maskOp.unroll(builder, options, state); })
      .Default([](Operation *) -> Operation * { return nullptr; });
}

static bool isAlreadyStripminedRoot(Operation *op,
                                    ArrayRef<OpFoldResult> shape) {
  if (op->hasAttr("dyno.stripmined")) {
    return true;
  }
  if (shape.empty()) {
    return false;
  }
  if (auto dimValue = shape.back().dyn_cast<Value>()) {
    return isa<vp::GetVLOp>(dimValue.getDefiningOp());
  }
  return false;
}

template <typename RootOp>
struct DynoStaticRowNormalizationPattern : OpRewritePattern<RootOp> {
  DynoStaticRowNormalizationPattern(MLIRContext *context)
      : OpRewritePattern<RootOp>(context, 3) {}

  LogicalResult matchAndRewrite(RootOp op,
                                PatternRewriter &rewriter) const final {
    if (!isSupportedEffectRoot(op)) {
      return rewriter.notifyMatchFailure(op, "not a supported effect root");
    }

    Value rootTile = getRootTileValue(op);
    auto rootTileType =
        rootTile ? dyn_cast<TileType>(rootTile.getType()) : nullptr;
    if (!rootTileType || rootTileType.getRank() < 2) {
      return rewriter.notifyMatchFailure(op, "root is not at least 2-D");
    }
    int64_t staticRows = rootTileType.getShape().front();
    if (ShapedType::isDynamic(staticRows) || staticRows < 1) {
      return rewriter.notifyMatchFailure(op,
                                         "root leading dimension is not a constant row pack");
    }

    auto loc = op.getLoc();
    Value dim = arith::ConstantIndexOp::create(rewriter, loc, staticRows);

    UnrollState state;
    state.initialize(op);
    dim = getUnrolledValue(rewriter, dim, getCloneOptions(), state);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Normalize any static `N x ...` effect root into `N` rank-reduced row
    // roots before VP lowering. This keeps the VP path focused on true 1-D
    // vector tiles instead of the more fragile row-pack representation.
    scf::ForOp::create(
        rewriter, loc, zero, dim, one, ValueRange{},
        [&](OpBuilder &b, Location loopLoc, Value iv, ValueRange) {
          // Rank-reduce the unit row so the carried slice becomes `?xf32`
          // instead of `1x?xf32`, and the corresponding outer lhs becomes a
          // rank-0 tile that VP lowering can treat as a scalar.
          UnrollOptions options(iv, b.getIndexAttr(1), 0, true);
          unrollRoot(b, op, options, state);
          scf::YieldOp::create(b, loopLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

static LogicalResult rewriteRootStripmining(Operation *op,
                                            RewriterBase &rewriter, unsigned vf,
                                            bool scalable) {
  if (op->hasAttr("dyno.stripmined")) {
    return failure();
  }
  if (auto maskOp = dyn_cast<MaskOp>(op)) {
    if (!isEffectOnlyMaskRoot(maskOp)) {
      return failure();
    }
  } else if (!isa<StoreOp, ScatterOp>(op)) {
    return failure();
  }

  auto shape = reifyRootShape(rewriter, op);
  if (failed(shape) || shape->empty()) {
    return failure();
  }
  if (isAlreadyStripminedRoot(op, *shape)) {
    return failure();
  }

  auto loc = op->getLoc();
  Type type = getRootProcessingType(op);
  // The shared stripmining rewrite is rank-agnostic: once any leading
  // dimensions are tiled away, both 1-D and 2-D+ effect roots just chunk the
  // trailing dimension with `vp.getvl`.
  Value dim = getOrCreateIndexValue(rewriter, shape->back(), loc);

  UnrollState state;
  state.initialize(op);
  dim = getUnrolledValue(rewriter, dim, getCloneOptions(), state);

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> inits = {dim, zero};
  SmallVector<Type> resultTypes = {dim.getType(), zero.getType()};

  scf::WhileOp::create(
      rewriter, loc, resultTypes, inits,
      [&](OpBuilder &b, Location loopLoc, ValueRange args) {
        Value avl = args[0];
        Value cond = arith::CmpIOp::create(
            b, loopLoc, arith::CmpIPredicate::sgt, avl, zero);
        scf::ConditionOp::create(b, loopLoc, cond, args);
      },
      [&](OpBuilder &b, Location loopLoc, ValueRange args) {
        Value avl = args[0];
        Value idx = args[1];

        Value vl = vp::GetVLOp::create(b, loopLoc, avl, vf, type, scalable);
        UnrollOptions options(idx, vl, shape->size() - 1, false);
        Operation *newRoot = unrollRoot(b, op, options, state);
        if (!newRoot) {
          return;
        }
        // Mark the materialized inner root so a later greedy visit does not try
        // to stripmine the same trailing-dimension chunk again.
        newRoot->setAttr("dyno.stripmined", b.getUnitAttr());

        Value newAvl = arith::SubIOp::create(b, loopLoc, avl, vl);
        Value newIdx = arith::AddIOp::create(b, loopLoc, idx, vl);
        scf::YieldOp::create(b, loopLoc, ValueRange{newAvl, newIdx});
      });

  rewriter.eraseOp(op);
  return success();
}

template <typename RootOp, typename OnSuccess>
static LogicalResult convertSliceToVP(RootOp op, PatternRewriter &rewriter,
                                      unsigned vf, bool scalable,
                                      OnSuccess onSuccess) {
  auto slice = collectBackwardSlice(op);
  VPConversionState state;
  state.vf = vf;
  state.scalable = scalable;
  state.initialize(op);
  if (failed(convertToVP(rewriter, op, state))) {
    return failure();
  }
  onSuccess(state);
  slice.remove(op.getOperation());
  cleanupConvertedSlice(slice.getArrayRef(), rewriter);
  return success();
}

static void finalizeConvertedRoot(StoreOp op, PatternRewriter &rewriter,
                                  VPConversionState &) {
  rewriter.eraseOp(op);
}

static void finalizeConvertedRoot(ScatterOp op, PatternRewriter &rewriter,
                                  VPConversionState &) {
  rewriter.eraseOp(op);
}

static void finalizeConvertedRoot(MaskOp op, PatternRewriter &rewriter,
                                  VPConversionState &) {
  rewriter.eraseOp(op);
}

static void finalizeConvertedRoot(ExtractOp op, PatternRewriter &rewriter,
                                  VPConversionState &state) {
  rewriter.replaceOp(op, state.valueMap.lookup(op.getResult()));
}

static void cleanupConvertedSlice(ArrayRef<Operation *> slice,
                                  PatternRewriter &rewriter) {
  for (Operation *op : llvm::reverse(slice)) {
    if (!op || op->hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    // Destructive VP-root rewrites erase the old tile slice after lowering the
    // concrete effect/value root. Resolve direct `dyno.dim` users first so the
    // producer can be dropped without losing its shape semantics.
    if (failed(resolveDimUsersOfOp(op, rewriter))) {
      continue;
    }
    if (!op->use_empty()) {
      continue;
    }
    if (isMemoryEffectFree(op)) {
      rewriter.eraseOp(op);
    }
  }
}

template <typename RootOp>
struct DynoStripminingPattern : OpRewritePattern<RootOp> {
  DynoStripminingPattern(MLIRContext *context, unsigned vf, bool scalable)
      : OpRewritePattern<RootOp>(context), vf(vf), scalable(scalable) {}

  LogicalResult matchAndRewrite(RootOp op,
                                PatternRewriter &rewriter) const final {
    if (!isSupportedEffectRoot(op)) {
      return rewriter.notifyMatchFailure(op, "not a supported effect root");
    }

    auto shape = reifyRootShape(rewriter, op);
    if (failed(shape) || shape->empty()) {
      return rewriter.notifyMatchFailure(op, "not a stripminable tile root");
    }
    if (failed(rewriteRootStripmining(op, rewriter, vf, scalable))) {
      return rewriter.notifyMatchFailure(op, "failed to stripmine root");
    }
    return success();
  }

private:
  unsigned vf;
  bool scalable;
};

template <typename RootOp> struct DynoTilingPattern : OpRewritePattern<RootOp> {
  DynoTilingPattern(MLIRContext *context, unsigned uf)
      : OpRewritePattern<RootOp>(context, 2), uf(uf) {}

  LogicalResult matchAndRewrite(RootOp op,
                                PatternRewriter &rewriter) const final {
    if (!isSupportedEffectRoot(op)) {
      return rewriter.notifyMatchFailure(op, "not a supported effect root");
    }

    Value rootTile = getRootTileValue(op);
    auto rootTileType =
        rootTile ? dyn_cast<TileType>(rootTile.getType()) : nullptr;
    if (!rootTileType || rootTileType.getRank() < 2) {
      return rewriter.notifyMatchFailure(op, "root is not at least 2-D");
    }

    auto shape = reifyRootShape(rewriter, op);
    if (failed(shape) || shape->size() < 2) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to reify tiled root shape");
    }

    if (auto rows = getConstantDynoIntValue((*shape)[0])) {
      if (*rows <= uf) {
        return rewriter.notifyMatchFailure(op, "already tiled");
      }
    }

    auto loc = op.getLoc();
    Value dim = getOrCreateIndexValue(rewriter, (*shape)[0], loc);

    UnrollState state;
    state.initialize(op);
    dim = getUnrolledValue(rewriter, dim, getCloneOptions(), state);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, uf);
    Value size = arith::ConstantIndexOp::create(rewriter, loc, uf);
    Value div = arith::DivUIOp::create(rewriter, loc, dim, size);
    Value ub = arith::MulIOp::create(rewriter, loc, div, size);

    scf::ForOp::create(
        rewriter, loc, zero, ub, step, ValueRange{},
        [&](OpBuilder &b, Location loopLoc, Value iv, ValueRange) {
          UnrollOptions options(iv, b.getIndexAttr(uf), 0, false);
          unrollRoot(b, op, options, state);
          scf::YieldOp::create(b, loopLoc);
        });

    if (uf != 1) {
      UnrollState tailState;
      tailState.initialize(op);
      scf::ForOp::create(
          rewriter, loc, ub, dim, one, ValueRange{},
          [&](OpBuilder &b, Location loopLoc, Value iv, ValueRange) {
            UnrollOptions options(iv, b.getIndexAttr(1), 0, false);
            unrollRoot(b, op, options, tailState);
            scf::YieldOp::create(b, loopLoc);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned uf;
};

template <typename RootOp>
struct ConvertRootToVPPattern : OpRewritePattern<RootOp> {
  ConvertRootToVPPattern(MLIRContext *context, unsigned vf, bool scalable)
      : OpRewritePattern<RootOp>(context), vf(vf), scalable(scalable) {}

  LogicalResult matchAndRewrite(RootOp op,
                                PatternRewriter &rewriter) const final {
    if (!isSupportedVPRoot(op)) {
      return rewriter.notifyMatchFailure(op, "not a supported VP root");
    }
    return convertSliceToVP(op, rewriter, vf, scalable,
                            [&](VPConversionState &state) {
                              finalizeConvertedRoot(op, rewriter, state);
                            });
  }

private:
  unsigned vf;
  bool scalable;
};

} // namespace

void DynoStripminingPass::runOnOperation() {
  auto parsedReductionMode = parseReductionModeKind(reductionMode);
  if (!parsedReductionMode) {
    getOperation()->emitError()
        << "unsupported dyno-stripmining reduction-mode `" << reductionMode
        << "`; expected one of: auto, sequential, parallel";
    signalPassFailure();
    return;
  }

  auto parsedFpPolicy = parseFloatingPointPolicy(fpPolicy);
  if (!parsedFpPolicy) {
    getOperation()->emitError()
        << "unsupported dyno-stripmining fp-policy `" << fpPolicy
        << "`; expected one of: strict, relaxed";
    signalPassFailure();
    return;
  }

  auto applyOuterNormalization = [&]() -> LogicalResult {
    RewritePatternSet outerNormalizationPatterns(&getContext());
    outerNormalizationPatterns
        .add<DynoStaticRowNormalizationPattern<StoreOp>,
             DynoStaticRowNormalizationPattern<ScatterOp>,
             DynoStaticRowNormalizationPattern<MaskOp>>(&getContext());
    return applyPatternsGreedily(getOperation(),
                                 std::move(outerNormalizationPatterns));
  };

  IRRewriter rewriter(&getContext());
  SmallVector<Operation *> reductions;
  getOperation()->walk([&](ReductionOp op) { reductions.push_back(op); });
  for (Operation *op : reductions) {
    auto reductionOp = dyn_cast<ReductionOp>(op);
    if (!reductionOp || !op->getBlock()) {
      continue;
    }
    rewriter.setInsertionPoint(reductionOp);
    (void)DynoStripminingReduction1DPattern::rewriteReduction(
        reductionOp, rewriter, vf, scalable, *parsedReductionMode,
        *parsedFpPolicy);
  }

  if (failed(applyOuterNormalization())) {
    signalPassFailure();
    return;
  }

  // Some roots already have a constant leading row-pack before any tiling, so
  // normalize those first.
  // Leading-dimension tiling is only meaningful once a root still has at least
  // two dimensions. Rank-1 roots skip tiling and go straight to the shared
  // last-dimension stripmining pattern below.
  bool hasLoopStripminingRoot = false;
  getOperation()->walk([&](Operation *op) {
    hasLoopStripminingRoot |= needsLoopStripmining(op);
  });
  if (!hasLoopStripminingRoot) {
    return;
  }

  RewritePatternSet tilingPatterns(&getContext());
  tilingPatterns.add<DynoTilingPattern<StoreOp>, DynoTilingPattern<ScatterOp>,
                     DynoTilingPattern<MaskOp>>(&getContext(), uf);
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(tilingPatterns)))) {
    signalPassFailure();
    return;
  }

  // Tiling can introduce fresh constant leading row-packs, e.g. a `2 x ?`
  // store slice in matmul or a `1 x ?` elementwise row slice. Normalize those
  // newly created roots before the trailing-dimension stripmining and VP
  // conversion phases.
  if (failed(applyOuterNormalization())) {
    signalPassFailure();
    return;
  }

  RewritePatternSet stripminingPatterns(&getContext());
  stripminingPatterns
      .add<DynoStripminingPattern<StoreOp>, DynoStripminingPattern<ScatterOp>,
           DynoStripminingPattern<MaskOp>>(&getContext(), vf, scalable);
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(stripminingPatterns)))) {
    signalPassFailure();
  }
}

void DynoStripminingPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<dyno::DynoDialect, vp::VPDialect, scf::SCFDialect,
                  arith::ArithDialect, memref::MemRefDialect>();
}

void ConvertDynoToVPPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns
      .add<ConvertRootToVPPattern<StoreOp>, ConvertRootToVPPattern<ScatterOp>,
           ConvertRootToVPPattern<ExtractOp>, ConvertRootToVPPattern<MaskOp>>(
          &getContext(), vf, scalable);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ConvertDynoToVPPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<dyno::DynoDialect, vp::VPDialect, scf::SCFDialect,
                  arith::ArithDialect, memref::MemRefDialect>();
}

void registerDynoStripminingPass() { PassRegistration<DynoStripminingPass>(); }
void registerConvertDynoToVPPass() { PassRegistration<ConvertDynoToVPPass>(); }

} // namespace dyno
} // namespace mlir
