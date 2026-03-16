#include "Conversion/DynoToVP.h"
#include "Dyno/IR/Dyno.h"
#include "Dyno/Utils/Builders.h"
#include "Dyno/Utils/ConvertToVP.h"
#include "Dyno/Utils/ReductionAttrs.h"
#include "Dyno/Utils/ReductionSemantics.h"
#include "Dyno/Utils/ShapeInference.h"
#include "Dyno/Utils/Slicing.h"
#include "VP/IR/VP.h"
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

static std::optional<ReductionModeKind> parseReductionModeKind(StringRef mode) {
  return llvm::StringSwitch<std::optional<ReductionModeKind>>(mode)
      .Case("auto", ReductionModeKind::Auto)
      .Case("sequential", ReductionModeKind::Sequential)
      .Case("parallel", ReductionModeKind::Parallel)
      .Default(std::nullopt);
}

// Only relaxed floating add/mul may carry the reassociation marker on the
// final register reduction emitted by the parallel strip-mined path.
static bool usesRelaxedParallelReassociation(CombiningKind kind,
                                             Type elementType) {
  if (!isa<FloatType>(elementType)) {
    return false;
  }
  return kind == CombiningKind::ADD || kind == CombiningKind::MUL;
}

// Build the carried reduction init as a tile-shaped value, materializing the
// implicit identity when the source op omits an explicit init.
static FailureOr<Value> buildReductionInitTile(OpBuilder &builder,
                                               ReductionOp op) {
  if (Value init = op.getInit()) {
    return init;
  }

  auto identity =
      buildReductionIdentity(builder, op.getLoc(), op.getKind(),
                             op.getResult().getType().getElementType());
  if (failed(identity)) {
    return failure();
  }

  auto resultShape = reifyDynoShape(builder, op.getResult());
  if (failed(resultShape)) {
    return failure();
  }
  return Value(dyno::SplatOp::create(builder, op.getLoc(), *identity,
                                     ArrayRef<OpFoldResult>(*resultShape)));
}

// Slice one lexicographic reduction coordinate out of the source tile and
// rank-reduce the reduced dimensions to scalars.
static FailureOr<Value> sliceReductionCoordinate(OpBuilder &builder,
                                                 ReductionOp op,
                                                 ArrayRef<int64_t> reducedDims,
                                                 ValueRange coordinates) {
  auto sourceShape = reifyDynoShape(builder, op.getTile());
  if (failed(sourceShape) || reducedDims.size() != coordinates.size()) {
    return failure();
  }

  SliceSpec spec;
  spec.offsets.reserve(sourceShape->size());
  spec.sizes.reserve(sourceShape->size());
  spec.droppedDims.reserve(sourceShape->size());

  unsigned reducedIndex = 0;
  for (auto [sourceDim, dimSize] : llvm::enumerate(*sourceShape)) {
    if (reducedIndex < reducedDims.size() &&
        static_cast<int64_t>(sourceDim) == reducedDims[reducedIndex]) {
      spec.offsets.push_back(coordinates[reducedIndex]);
      spec.sizes.push_back(builder.getIndexAttr(1));
      spec.droppedDims.push_back(true);
      ++reducedIndex;
      continue;
    }
    spec.offsets.push_back(builder.getIndexAttr(0));
    spec.sizes.push_back(dimSize);
    spec.droppedDims.push_back(false);
  }

  SliceState state;
  state.initialize(op);
  return sliceValue(builder, op.getTile(), spec, state);
}

// Materialize the exact lexicographic traversal required for ordered
// higher-dimensional reductions.
static FailureOr<Value> buildOrderedReductionTraversal(
    RewriterBase &rewriter, ReductionOp op, ArrayRef<int64_t> reducedDims,
    unsigned depth, SmallVectorImpl<Value> &coordinates, Value initAcc) {
  auto loc = op.getLoc();
  if (depth == reducedDims.size()) {
    auto frontier =
        sliceReductionCoordinate(rewriter, op, reducedDims, coordinates);
    if (failed(frontier)) {
      return failure();
    }
    return createCombiningOp(rewriter, loc, op.getKind(), initAcc, *frontier);
  }

  auto sourceShape = reifyDynoShape(rewriter, op.getTile());
  if (failed(sourceShape)) {
    return failure();
  }

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value upper =
      getOrCreateIndexValue(rewriter, (*sourceShape)[reducedDims[depth]], loc);
  auto loop =
      scf::ForOp::create(rewriter, loc, zero, upper, one, ValueRange{initAcc});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    coordinates.push_back(loop.getInductionVar());
    auto nested = buildOrderedReductionTraversal(
        rewriter, op, reducedDims, depth + 1, coordinates,
        loop.getRegionIterArgs().front());
    coordinates.pop_back();
    if (failed(nested)) {
      rewriter.eraseOp(loop);
      return failure();
    }
    scf::YieldOp::create(rewriter, loc, *nested);
  }
  return loop.getResult(0);
}

// Rewrite a reduction into the explicit ordered traversal form from the
// formalism, preserving one carried accumulator through the whole traversal.
static LogicalResult rewriteOrderedReduction(ReductionOp op,
                                             RewriterBase &rewriter) {
  auto init = buildReductionInitTile(rewriter, op);
  if (failed(init)) {
    return failure();
  }

  SmallVector<Value> coordinates;
  auto reduced = buildOrderedReductionTraversal(rewriter, op, op.getDims(), 0,
                                                coordinates, *init);
  if (failed(reduced)) {
    return failure();
  }
  rewriter.replaceOp(op, *reduced);
  return success();
}

// Restrict strip-mining to unprocessed rank-1 reductions over source dim 0 so
// the VP boundary only sees the normalized 1-D forms.
static LogicalResult isRank1StripmineCandidate(ReductionOp op,
                                               RewriterBase &rewriter) {
  if (op->hasAttr(kDynoStripminedAttr) ||
      op->hasAttr(kDynoParallelReductionAttr)) {
    return failure();
  }
  auto tile = op.getTile();
  if (op.getDims().size() != 1 || op.getDims().front() != 0 ||
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

// Emit the formal parallel strip-mined reduction: lane-wise vector
// accumulation, masked tail preservation, and one final register reduction.
static LogicalResult
rewriteParallelRank1Reduction(ReductionOp op, RewriterBase &rewriter,
                              unsigned vf, bool scalable,
                              FloatingPointPolicy fpPolicy) {
  if (failed(isRank1StripmineCandidate(op, rewriter))) {
    return failure();
  }

  auto identity =
      buildReductionIdentity(rewriter, op.getLoc(), op.getKind(),
                             op.getTile().getType().getElementType());
  if (failed(identity)) {
    return failure();
  }

  auto tile = op.getTile();
  auto dim = reifyDynoDim(rewriter, tile, 0);
  if (failed(dim)) {
    return failure();
  }

  auto loc = op.getLoc();
  Type elementType = tile.getType().getElementType();
  Value dimValue = getOrCreateIndexValue(rewriter, *dim, loc);
  Value vlmax =
      vp::GetVLOp::create(rewriter, loc, dimValue, vf, elementType, scalable);

  Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value initAcc = dyno::SplatOp::create(rewriter, loc, *identity,
                                        SmallVector<OpFoldResult>{vlmax});

  SmallVector<Value> inits = {dimValue, zeroIdx, initAcc};
  SmallVector<Type> resultTypes = {dimValue.getType(), zeroIdx.getType(),
                                   initAcc.getType()};
  auto whileOp = scf::WhileOp::create(
      rewriter, loc, resultTypes, inits,
      [&](OpBuilder &, Location, ValueRange) {},
      [&](OpBuilder &, Location, ValueRange) {});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(whileOp.getBeforeBody());
    Value avl = whileOp.getBeforeArguments()[0];
    Value cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                       avl, zeroIdx);
    scf::ConditionOp::create(rewriter, loc, cond, whileOp.getBeforeArguments());
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(whileOp.getAfterBody());
    Value avl = whileOp.getAfterArguments()[0];
    Value idx = whileOp.getAfterArguments()[1];
    Value acc = whileOp.getAfterArguments()[2];

    Value vl =
        vp::GetVLOp::create(rewriter, loc, avl, vf, elementType, scalable);
    auto spec = SliceSpec::getSingleDimSlice(rewriter, tile, 0, idx, vl,
                                             /*dropUnitDim=*/false);
    if (failed(spec)) {
      rewriter.eraseOp(whileOp);
      return failure();
    }

    SliceState state;
    state.initialize(op);
    auto chunk = sliceValue(rewriter, tile, *spec, state);
    if (failed(chunk)) {
      rewriter.eraseOp(whileOp);
      return failure();
    }

    // TODO: Investigate if mask is really necessary here or if the VP lowering
    // eliminates it. The problem is that for Dyno dialect, the elementwise ops
    // enforces same shaped operands. An old solution was using
    // `passthru-operand` as a marker attribute to annotate which operand can be
    // "longer" (is the accumulator). But this is a quite ad-hoc solution, and
    // should be treated carefully. Mask here is definitely a performance issue.
    Value laneIndices = dyno::StepOp::create(rewriter, loc, zeroIdx, 0,
                                             SmallVector<OpFoldResult>{vlmax});
    Value activeVL = dyno::SplatOp::create(rewriter, loc, vl,
                                           SmallVector<OpFoldResult>{vlmax});
    Value activeMask = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, laneIndices, activeVL);
    auto maskedAcc = dyno::MaskOp::create(
        rewriter, loc, TypeRange{acc.getType()}, activeMask,
        [&](OpBuilder &b, Location bodyLoc) {
          Value combined =
              createCombiningOp(b, bodyLoc, op.getKind(), acc, *chunk);
          dyno::MaskYieldOp::create(b, bodyLoc, ValueRange{combined});
        },
        acc);
    Value newAcc = maskedAcc.getResult(0);

    Value newAvl = arith::SubIOp::create(rewriter, loc, avl, vl);
    Value newIdx = arith::AddIOp::create(rewriter, loc, idx, vl);
    scf::YieldOp::create(rewriter, loc, ValueRange{newAvl, newIdx, newAcc});
  }

  Value finalReduction = dyno::ReductionOp::create(rewriter, loc, op.getKind(),
                                                   whileOp->getResult(2),
                                                   op.getDims(), op.getInit());
  auto finalReductionOp = cast<ReductionOp>(finalReduction.getDefiningOp());
  copyReductionFloatingPointPolicy(op, finalReductionOp);
  finalReductionOp->setAttr(kDynoStripminedAttr, rewriter.getUnitAttr());
  finalReductionOp->setAttr(kDynoParallelReductionAttr, rewriter.getUnitAttr());
  if (fpPolicy == FloatingPointPolicy::Relaxed &&
      usesRelaxedParallelReassociation(op.getKind(), elementType)) {
    finalReductionOp->setAttr(kDynoParallelReassocAttr, rewriter.getUnitAttr());
  }
  rewriter.replaceOp(op, finalReduction);
  return success();
}

// Emit the exact ordered rank-1 strip-mined reduction: source-order chunking
// with one chunk-local ordered reduction and no final register reduction.
static LogicalResult rewriteSequentialRank1Reduction(ReductionOp op,
                                                     RewriterBase &rewriter,
                                                     unsigned vf,
                                                     bool scalable) {
  if (failed(isRank1StripmineCandidate(op, rewriter))) {
    return failure();
  }

  auto loc = op.getLoc();
  Type elementType = op.getTile().getType().getElementType();
  auto dim = reifyDynoDim(rewriter, op.getTile(), 0);
  if (failed(dim)) {
    return failure();
  }

  auto initTile = buildReductionInitTile(rewriter, op);
  if (failed(initTile)) {
    return failure();
  }

  Value dimValue = getOrCreateIndexValue(rewriter, *dim, loc);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

  SmallVector<Value> inits = {dimValue, zero, *initTile};
  SmallVector<Type> resultTypes = {dimValue.getType(), zero.getType(),
                                   initTile->getType()};
  auto whileOp = scf::WhileOp::create(
      rewriter, loc, resultTypes, inits,
      [&](OpBuilder &, Location, ValueRange) {},
      [&](OpBuilder &, Location, ValueRange) {});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(whileOp.getBeforeBody());
    Value avl = whileOp.getBeforeArguments()[0];
    Value cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                       avl, zero);
    scf::ConditionOp::create(rewriter, loc, cond, whileOp.getBeforeArguments());
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(whileOp.getAfterBody());
    Value avl = whileOp.getAfterArguments()[0];
    Value idx = whileOp.getAfterArguments()[1];
    Value acc = whileOp.getAfterArguments()[2];
    Value vl =
        vp::GetVLOp::create(rewriter, loc, avl, vf, elementType, scalable);

    auto chunkSpec =
        SliceSpec::getSingleDimSlice(rewriter, op.getTile(), 0, idx, vl,
                                     /*dropUnitDim=*/false);
    if (failed(chunkSpec)) {
      rewriter.eraseOp(whileOp);
      return failure();
    }

    SliceState state;
    state.initialize(op);
    auto chunk = sliceValue(rewriter, op.getTile(), *chunkSpec, state);
    if (failed(chunk)) {
      rewriter.eraseOp(whileOp);
      return failure();
    }

    // Keep each chunk as an ordered 1-D reduction so the VP boundary can lower
    // it to an ordered vector reduction without lane scalarization here.
    Value newAcc = dyno::ReductionOp::create(rewriter, loc, op.getKind(),
                                             *chunk, op.getDims(), acc);
    auto newAccOp = cast<ReductionOp>(newAcc.getDefiningOp());
    copyReductionFloatingPointPolicy(op, newAccOp);
    newAccOp->setAttr(kDynoStripminedAttr, rewriter.getUnitAttr());
    newAccOp->setAttr(kDynoSequentialReductionAttr, rewriter.getUnitAttr());

    Value newAvl = arith::SubIOp::create(rewriter, loc, avl, vl);
    Value newIdx = arith::AddIOp::create(rewriter, loc, idx, vl);
    scf::YieldOp::create(rewriter, loc, ValueRange{newAvl, newIdx, newAcc});
  }

  rewriter.replaceOp(op, whileOp.getResult(2));
  return success();
}

// Normalize higher-dimensional reductions before VP preparation: admissible
// cases factorize, non-admissible ones become explicit ordered traversal.
struct NormalizeHigherDimReductionPattern : OpRewritePattern<ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getDims().size() <= 1 || op->hasAttr(kDynoStripminedAttr)) {
      return rewriter.notifyMatchFailure(op,
                                         "not a higher-dimensional reduction");
    }

    // The default floating point policy should already be injected before
    // invoking this pattern in DynoStripminingPass.
    auto fpPolicy = getReductionFloatingPointPolicy(op);
    if (!fpPolicy) {
      return rewriter.notifyMatchFailure(op, "missing valid dyno.fp_policy");
    }

    auto elementType = op.getTile().getType().getElementType();
    if (!isFactorizationAdmissible(op.getKind(), elementType, *fpPolicy)) {
      if (failed(rewriteOrderedReduction(op, rewriter))) {
        return rewriter.notifyMatchFailure(op,
                                           "failed ordered reduction lowering");
      }
      return success();
    }

    SmallVector<int64_t> reducedSourceDims;
    reducedSourceDims.reserve(op.getDims().size());
    Value current = op.getTile();
    for (auto it = op.getDims().rbegin(); it != op.getDims().rend(); ++it) {
      int64_t sourceDim = *it;
      unsigned currentDim =
          mapSourceDimToCurrentReductionDim(sourceDim, reducedSourceDims);
      Value init = sourceDim == op.getDims().front() ? op.getInit() : Value();
      current = dyno::ReductionOp::create(
          rewriter, op.getLoc(), op.getKind(), current,
          ArrayRef<int64_t>{static_cast<int64_t>(currentDim)}, init);
      copyReductionFloatingPointPolicy(
          op, cast<ReductionOp>(current.getDefiningOp()));
      reducedSourceDims.push_back(sourceDim);
    }
    rewriter.replaceOp(op, current);
    return success();
  }
};

// Any higher-rank reduction that survives normalization must stay in explicit
// ordered loop form rather than reaching the VP conversion boundary directly.
struct LowerHigherRankReductionPattern : OpRewritePattern<ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const final {
    if (op->hasAttr(kDynoStripminedAttr) || op.getDims().size() != 1 ||
        op.getTile().getType().getRank() <= 1) {
      return rewriter.notifyMatchFailure(
          op, "not a non-scalar single-dimension reduction");
    }
    if (failed(rewriteOrderedReduction(op, rewriter))) {
      return rewriter.notifyMatchFailure(
          op, "failed higher-rank reduction lowering");
    }
    return success();
  }
};

// Select between the explicit parallel and sequential rank-1 lowering modes
// after consulting the centralized legality helpers.
struct LowerRank1ReductionPattern : OpRewritePattern<ReductionOp> {
  LowerRank1ReductionPattern(MLIRContext *context, unsigned vf, bool scalable,
                             ReductionModeKind reductionMode)
      : OpRewritePattern<ReductionOp>(context, 3), vf(vf), scalable(scalable),
        reductionMode(reductionMode) {}

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const final {
    if (failed(isRank1StripmineCandidate(op, rewriter))) {
      return rewriter.notifyMatchFailure(op,
                                         "not a rank-1 stripmine candidate");
    }

    // The default floating point policy should already be injected before
    // invoking this pattern in DynoStripminingPass.
    auto fpPolicy = getReductionFloatingPointPolicy(op);
    if (!fpPolicy) {
      return rewriter.notifyMatchFailure(op, "missing valid dyno.fp_policy");
    }

    auto elementType = op.getTile().getType().getElementType();
    ReductionModeKind selectedMode = chooseReductionMode(
        op.getKind(), elementType, *fpPolicy, reductionMode);
    if (selectedMode == ReductionModeKind::Parallel &&
        !isParallelStripmineAdmissible(op.getKind(), elementType, *fpPolicy)) {
      return rewriter.notifyMatchFailure(
          op, "parallel strip-mining is illegal for the active policy");
    }

    if (selectedMode == ReductionModeKind::Parallel) {
      if (failed(rewriteParallelRank1Reduction(op, rewriter, vf, scalable,
                                               *fpPolicy))) {
        return rewriter.notifyMatchFailure(op, "failed parallel strip-mining");
      }
      return success();
    }

    if (failed(rewriteSequentialRank1Reduction(op, rewriter, vf, scalable))) {
      return rewriter.notifyMatchFailure(op, "failed sequential strip-mining");
    }
    return success();
  }

private:
  unsigned vf;
  bool scalable;
  ReductionModeKind reductionMode;
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

static FailureOr<Operation *> sliceRoot(OpBuilder &builder, Operation *op,
                                        Value tile, unsigned dim,
                                        OpFoldResult offset, OpFoldResult size,
                                        bool dropUnitDim) {
  auto spec = SliceSpec::getSingleDimSlice(builder, tile, dim, offset, size,
                                           dropUnitDim);
  if (failed(spec)) {
    return failure();
  }
  SliceState state;
  state.initialize(op);
  return sliceRootOperation(builder, op, *spec, state);
}

static bool isAlreadyStripminedRoot(Operation *op,
                                    ArrayRef<OpFoldResult> shape) {
  if (op->hasAttr(kDynoStripminedAttr)) {
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
struct DynoLowerLeadingDimsPattern : OpRewritePattern<RootOp> {
  DynoLowerLeadingDimsPattern(MLIRContext *context, unsigned uf)
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

    auto loc = op.getLoc();
    auto shape = reifyRootShape(rewriter, op);
    if (failed(shape) || shape->empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to reify leading root extent");
    }
    Value dim = getOrCreateIndexValue(rewriter, shape->front(), loc);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value step =
        arith::ConstantIndexOp::create(rewriter, loc, std::max(1u, uf));

    // Preserve leading dimensions as explicit loops so only scalar or 1-D tiles
    // survive to the VP boundary.
    auto outerLoop =
        scf::ForOp::create(rewriter, loc, zero, dim, step, ValueRange{});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value candidateEnd = arith::AddIOp::create(
          rewriter, loc, outerLoop.getInductionVar(), step);
      Value inBounds = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, candidateEnd, dim);
      Value blockEnd =
          arith::SelectOp::create(rewriter, loc, inBounds, candidateEnd, dim);
      auto innerLoop =
          scf::ForOp::create(rewriter, loc, outerLoop.getInductionVar(),
                             blockEnd, one, ValueRange{});
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      if (failed(sliceRoot(rewriter, op, rootTile, 0,
                           innerLoop.getInductionVar(),
                           rewriter.getIndexAttr(1),
                           /*dropUnitDim=*/true))) {
        rewriter.eraseOp(innerLoop);
        rewriter.eraseOp(outerLoop);
        return rewriter.notifyMatchFailure(
            op, "failed to lower preserved leading dimension");
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned uf;
};

static LogicalResult rewriteRootStripmining(Operation *op,
                                            RewriterBase &rewriter, unsigned vf,
                                            bool scalable) {
  if (op->hasAttr(kDynoStripminedAttr)) {
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

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> inits = {dim, zero};
  SmallVector<Type> resultTypes = {dim.getType(), zero.getType()};

  auto whileOp = scf::WhileOp::create(
      rewriter, loc, resultTypes, inits,
      [&](OpBuilder &, Location, ValueRange) {},
      [&](OpBuilder &, Location, ValueRange) {});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(whileOp.getBeforeBody());
    Value avl = whileOp.getBeforeArguments()[0];
    Value cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                       avl, zero);
    scf::ConditionOp::create(rewriter, loc, cond, whileOp.getBeforeArguments());
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(whileOp.getAfterBody());
    Value avl = whileOp.getAfterArguments()[0];
    Value idx = whileOp.getAfterArguments()[1];

    Value vl = vp::GetVLOp::create(rewriter, loc, avl, vf, type, scalable);
    auto newRoot = sliceRoot(rewriter, op, getRootTileValue(op),
                             shape->size() - 1, idx, vl,
                             /*dropUnitDim=*/false);
    if (failed(newRoot)) {
      rewriter.eraseOp(whileOp);
      return failure();
    }
    // Mark the materialized inner root so a later greedy visit does not try
    // to stripmine the same trailing-dimension chunk again.
    (*newRoot)->setAttr(kDynoStripminedAttr, rewriter.getUnitAttr());

    Value newAvl = arith::SubIOp::create(rewriter, loc, avl, vl);
    Value newIdx = arith::AddIOp::create(rewriter, loc, idx, vl);
    scf::YieldOp::create(rewriter, loc, ValueRange{newAvl, newIdx});
  }

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

  // Stripmining now treats the pass option as the default policy to stamp onto
  // reductions that have not made their floating-point legality explicit yet.
  bool sawInvalidReductionPolicy = false;
  getOperation()->walk([&](ReductionOp op) {
    if (sawInvalidReductionPolicy) {
      return;
    }
    auto policyAttr = op->getDiscardableAttr(kDynoFpPolicyAttr);
    if (!policyAttr) {
      setReductionFloatingPointPolicy(op, *parsedFpPolicy);
      return;
    }
    if (!getReductionFloatingPointPolicy(op)) {
      op.emitOpError() << "expected " << kDynoFpPolicyAttr
                       << " to be one of: strict, relaxed";
      sawInvalidReductionPolicy = true;
    }
  });
  if (sawInvalidReductionPolicy) {
    signalPassFailure();
    return;
  }

  RewritePatternSet reductionPatterns(&getContext());
  reductionPatterns.add<NormalizeHigherDimReductionPattern>(&getContext());
  reductionPatterns.add<LowerHigherRankReductionPattern>(&getContext());
  reductionPatterns.add<LowerRank1ReductionPattern>(&getContext(), vf, scalable,
                                                    *parsedReductionMode);
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(reductionPatterns)))) {
    signalPassFailure();
    return;
  }

  if (*parsedReductionMode == ReductionModeKind::Parallel) {
    bool sawIllegalParallel = false;
    getOperation()->walk([&](ReductionOp op) {
      if (sawIllegalParallel || op->hasAttr(kDynoParallelReductionAttr)) {
        return;
      }
      auto tileType = op.getTile().getType();
      if (tileType.getRank() != 1 || op.getDims().size() != 1 ||
          op.getDims().front() != 0) {
        return;
      }
      auto policy = getReductionFloatingPointPolicy(op);
      if (!policy) {
        op.emitOpError("missing valid dyno.fp_policy after stripmining "
                       "materialization");
        sawIllegalParallel = true;
        return;
      }
      if (!isParallelStripmineAdmissible(op.getKind(),
                                         tileType.getElementType(), *policy)) {
        op.emitOpError()
            << "parallel strip-mining is illegal for this combiner/type under "
               "the active "
            << kDynoFpPolicyAttr;
        sawIllegalParallel = true;
      }
    });
    if (sawIllegalParallel) {
      signalPassFailure();
      return;
    }
  }

  RewritePatternSet leadingDimPatterns(&getContext());
  leadingDimPatterns.add<DynoLowerLeadingDimsPattern<StoreOp>,
                         DynoLowerLeadingDimsPattern<ScatterOp>,
                         DynoLowerLeadingDimsPattern<MaskOp>>(&getContext(),
                                                              uf);
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(leadingDimPatterns)))) {
    signalPassFailure();
    return;
  }

  bool hasLoopStripminingRoot = false;
  getOperation()->walk([&](Operation *op) {
    hasLoopStripminingRoot |= needsLoopStripmining(op);
  });
  if (!hasLoopStripminingRoot) {
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
