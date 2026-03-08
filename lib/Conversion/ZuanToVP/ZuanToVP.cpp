#include "Conversion/ZuanToVP.h"
#include "VP/IR/VP.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/ConvertToVP.h"
#include "Zuan/Utils/Unrolling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace zuan {

namespace {

static Type getProcessingType(DynamicOp dynamicOp) {
  // Find the max bitwidth type.
  Type type = nullptr;
  int64_t maxBitwidth = 0;
  dynamicOp->walk([&](Operation *op) {
    auto resultTypes = op->getResultTypes();
    if (!resultTypes.empty()) {
      for (auto resultType : resultTypes) {
        if (auto tileType = dyn_cast<TileType>(resultType)) {
          auto elemType = tileType.getElementType();
          // FIXME: working around for index.
          int64_t currBitwidth = 64;
          if (elemType.isIntOrFloat()) {
            currBitwidth = elemType.getIntOrFloatBitWidth();
          }
          if (currBitwidth > maxBitwidth) {
            maxBitwidth = currBitwidth;
            type = elemType;
          }
        }
      }
    }
  });
  return type;
}

struct ZuanStripminingReduction1DPattern : OpRewritePattern<ReductionOp> {
  // Benefit is set to larger than any other stripmining pattern. So unrolled
  // reduction will not be stripmined again.
  explicit ZuanStripminingReduction1DPattern(MLIRContext *context, unsigned vf,
                                             bool scalable)
      : OpRewritePattern<ReductionOp>(context, 2), vf(vf), scalable(scalable) {}

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const final {
    auto tile = op.getTile();
    auto dims = op.getDims();

    if (dims.size() != 1 || dims[0] != 0 || tile.getType().getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "not 1-D reduction");
    }

    auto loc = op.getLoc();
    auto dynamicOp = op->getParentOfType<DynamicOp>();

    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    shapeInfo.inferShape(dynamicOp, shapeInferenceState);

    // shapeInfo.dump(llvm::dbgs());

    auto tileShape = *shapeInfo.lookupShape(tile);
    if (auto val = tileShape[0].getValue()) {
      if (isa<vp::GetVLOp>(val->getDefiningOp())) {
        return rewriter.notifyMatchFailure(op, "already stripmined");
      }
    }

    UnrollState state;
    state.initialize(dynamicOp);

    // Hoist the reduction to a new dynamic op. So it will be the scalar defined
    // outside the dynamic op. This should avoid duplicated computation.
    // Additional loads are expected to be eliminated with CSE, after lowered to
    // VP dialect.
    rewriter.setInsertionPoint(dynamicOp);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Type type = tile.getType().getElementType();
    Value dim = tileShape[0].getOrCreateValue(rewriter, loc);
    Value vlmax = vp::GetVLOp::create(rewriter, loc, dim, vf, type, scalable);

    Value zeroElem = arith::ConstantOp::create(rewriter, 
        loc, type, rewriter.getZeroAttr(type));

    auto newDynamicOp =
        DynamicOp::create(rewriter, loc, type, dynamicOp.getInits(), nullptr);
    state.valueMap.map(dynamicOp.getBody().getArguments(),
                       newDynamicOp.getBody().getArguments());
    rewriter.setInsertionPointToStart(&newDynamicOp.getBody().front());

    SmallVector<OpFoldResult> sizes{vlmax};
    Value initAcc = zuan::SplatOp::create(rewriter, loc, zeroElem, sizes);
    SmallVector<Value> inits = {dim, zero, initAcc};
    SmallVector<Type> resultTypes = {dim.getType(), zero.getType(),
                                     initAcc.getType()};

    auto whileOp = scf::WhileOp::create(rewriter, 
        loc, resultTypes, inits,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto avl = args[0];
          Value cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt,
                                               avl, zero);
          scf::ConditionOp::create(b, loc, cond, args);
        },
        [&](OpBuilder &b, Location loc, ValueRange args) -> void {
          auto avl = args[0];
          auto idx = args[1];
          auto acc = args[2];

          Value vl = vp::GetVLOp::create(rewriter, loc, avl, vf, type, scalable);
          UnrollOptions options(idx, vl, 0, true);

          auto newSourceTile = getUnrolledValue(rewriter, tile, options, state);
          auto newAcc = createCombiningOp(rewriter, loc, op.getKind(), acc,
                                          newSourceTile);
          newAcc.getDefiningOp()->setAttr("zuan_passthru_operand",
                                          b.getIndexAttr(0));

          Value newAvl = arith::SubIOp::create(b, loc, avl, vl);
          Value newIdx = arith::AddIOp::create(b, loc, idx, vl);
          scf::YieldOp::create(b, loc, ValueRange{newAvl, newIdx, newAcc});
        });

    Value init = op.getInit();
    if (init) {
      init = getUnrolledValue(rewriter, init, getCloneOptions(), state);
    }

    auto finalAcc = whileOp->getResult(2);
    Value finalRed = zuan::ReductionOp::create(rewriter, loc, op.getKind(),
                                                        finalAcc, dims, init);

    YieldOp::create(rewriter, loc, finalRed, nullptr);

    rewriter.setInsertionPoint(op);
    Value splat = zuan::SplatOp::create(rewriter, 
        loc, newDynamicOp->getResult(0), ArrayRef<int64_t>{});
    rewriter.replaceOp(op, splat);

    return success();
  }

private:
  unsigned vf;
  bool scalable;
};

struct ZuanStripminingPattern : OpRewritePattern<DynamicOp> {
  explicit ZuanStripminingPattern(MLIRContext *context, unsigned vf,
                                  bool scalable)
      : OpRewritePattern<DynamicOp>(context, 1), vf(vf), scalable(scalable) {}

  LogicalResult matchAndRewrite(DynamicOp op,
                                PatternRewriter &rewriter) const final {
    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    shapeInfo.inferShape(op, shapeInferenceState);

    splitDynamicOpForUnrolling(rewriter, op, 0, shapeInfo);

    auto yieldOp = op.getYieldOp();
    auto yieldRegion = &yieldOp.getBody();
    if (yieldRegion->getOps().empty()) {
      // Scalars are produced with 1-D reduction or dummy splat. So no need to
      // stripmine them anymore.
      return rewriter.notifyMatchFailure(op, "empty yield region");
    }

    if (yieldOp->getNumOperands() != 0) {
      return rewriter.notifyMatchFailure(op, "expected no operand");
    }

    auto loc = op.getLoc();

    auto referenceOp = &*yieldRegion->getOps().begin();
    auto iface = dyn_cast<ZuanUnrollingInterface>(referenceOp);
    assert(iface && "expected an unrolling interface");

    auto shape = iface.getShapeToUnroll(shapeInfo);

    if (shape->empty()) {
      return rewriter.notifyMatchFailure(op, "0-D operations");
    }

    if (auto val = shape->back().getValue()) {
      if (isa<vp::GetVLOp>(val->getDefiningOp())) {
        return rewriter.notifyMatchFailure(op, "already stripmined");
      }
    }

    // Strip-mining introduces EVL on the minor dimension. This pattern is not
    // responsible for eliminating a dynamic outer dimension in a 2-D tile; that
    // is handled by `ZuanTilingPattern` below.
    auto unrollIdx = shape->size() - 1;

    auto type = getProcessingType(op);

    rewriter.setInsertionPoint(op);
    Value dim = (*shape).back().getOrCreateValue(rewriter, loc);

    UnrollState state;
    state.initialize(op);

    dim = getUnrolledValue(rewriter, dim, getCloneOptions(), state);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    SmallVector<Value> inits = {dim, zero};
    SmallVector<Type> resultTypes = {dim.getType(), zero.getType()};

    scf::WhileOp::create(rewriter, 
        loc, resultTypes, inits,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto avl = args[0];
          Value cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt,
                                               avl, zero);
          scf::ConditionOp::create(b, loc, cond, args);
        },
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto avl = args[0];
          auto idx = args[1];

          Value vl = vp::GetVLOp::create(rewriter, loc, avl, vf, type, scalable);
          UnrollOptions options(idx, vl, unrollIdx, false);
          op.unroll(b, options, state);

          Value newAvl = arith::SubIOp::create(b, loc, avl, vl);
          Value newIdx = arith::AddIOp::create(b, loc, idx, vl);

          scf::YieldOp::create(b, loc, ValueRange{newAvl, newIdx});
        });

    // whileOp->getParentOfType<func::FuncOp>().dump();

    rewriter.eraseOp(op);

    return success();
  }

private:
  unsigned vf;
  bool scalable;
};

struct ZuanTilingPattern : OpRewritePattern<DynamicOp> {
  // The benefit of tiling is set larger. So the the tiling will be at the outer
  // loop, and stripmining will be at the inner loop, this should lead to better
  // cache locality.
  explicit ZuanTilingPattern(MLIRContext *context, unsigned uf)
      : OpRewritePattern<DynamicOp>(context, 2), uf(uf) {}

  LogicalResult matchAndRewrite(DynamicOp op,
                                PatternRewriter &rewriter) const final {
    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    shapeInfo.inferShape(op, shapeInferenceState);

    splitDynamicOpForUnrolling(rewriter, op, 0, shapeInfo);

    // TODO: Refactor this preparation code.
    auto yieldOp = op.getYieldOp();
    auto yieldRegion = &yieldOp.getBody();
    if (yieldRegion->getOps().empty()) {
      // TODO: Split the scalar operations.
      // Scalars are produced with 1-D reduction or dummy splat. So no need to
      // stripmine them anymore.
      return rewriter.notifyMatchFailure(op, "empty yield region");
    }

    if (yieldOp->getNumOperands() != 0) {
      return rewriter.notifyMatchFailure(op, "expected no operand");
    }

    auto loc = op.getLoc();

    auto referenceOp = &*yieldRegion->getOps().begin();
    auto iface = dyn_cast<ZuanUnrollingInterface>(referenceOp);
    assert(iface && "expected an unrolling interface");

    auto shape = iface.getShapeToUnroll(shapeInfo);

    if (shape->size() < 2) {
      return rewriter.notifyMatchFailure(op, "< 2-D operations");
    }

    if (auto integer = shape->front().getInteger()) {
      if (*integer <= uf) {
        return rewriter.notifyMatchFailure(op, "already tiled");
      }
    }

    // This is the pass that is expected to eliminate a dynamic outer dimension
    // before the VP conversion path. After tiling, the main loop sees a static
    // outer chunk of size `uf`, and the tail loop sees a static outer size of 1.
    // `ConvertToVP` still checks for any leftover `DynamicOuterLoopAndVector`
    // plans and rejects them explicitly instead of assuming this pass always ran.
    //
    // All shapes are now the same and >= target-rank, so should be safe to
    // access the first.
    auto dim = (*shape)[0].getOrCreateValue(rewriter, loc);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, uf);

    UnrollState state;
    state.initialize(op);

    // Make sure no use-before-def is produced.
    dim = getUnrolledValue(rewriter, dim, getCloneOptions(), state);

    Value size = arith::ConstantIndexOp::create(rewriter, loc, uf);
    Value div = arith::DivUIOp::create(rewriter, loc, dim, size);
    Value ub = arith::MulIOp::create(rewriter, loc, div, size);

    [[maybe_unused]]
    auto loop = scf::ForOp::create(rewriter, 
        loc, zero, ub, step, ValueRange{},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          UnrollOptions options(iv, b.getIndexAttr(uf), 0, false);
          op.unroll(b, options, state);
          scf::YieldOp::create(b, loc);
        });

    if (uf != 1) {
      UnrollState tailState;
      tailState.initialize(op);

      [[maybe_unused]]
      auto taileLoop = scf::ForOp::create(rewriter, 
          loc, ub, dim, one, ValueRange{},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            UnrollOptions options(iv, b.getIndexAttr(1), 0, false);
            op.unroll(b, options, tailState);
            scf::YieldOp::create(b, loc);
          });
    }

    // loop->getParentOfType<func::FuncOp>().dump();
    rewriter.eraseOp(op);
    ;

    return success();
  }

private:
  unsigned uf;
};

struct ConvertZuanToVPPattern : OpRewritePattern<DynamicOp> {
  explicit ConvertZuanToVPPattern(MLIRContext *context, unsigned vf,
                                  bool scalable)
      : OpRewritePattern(context), vf(vf), scalable(scalable) {}

  LogicalResult matchAndRewrite(DynamicOp op,
                                PatternRewriter &rewriter) const final {
    // op->dump();
    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    shapeInfo.inferShape(op, shapeInferenceState);

    VPConversionState state;
    state.vf = vf;
    state.scalable = scalable;
    state.initialize(op);

    if (failed(convertToVP(rewriter, op, shapeInfo, state))) {
      return rewriter.notifyMatchFailure(op, "failed to convert dynamic op to VP");
    }

    // op->getParentOfType<func::FuncOp>().dump();
    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned vf;
  bool scalable;
};

} // namespace

void ZuanStripminingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ZuanStripminingReduction1DPattern, ZuanStripminingPattern>(
      &getContext(), vf, scalable);
  patterns.add<ZuanTilingPattern>(&getContext(), uf);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ZuanStripminingPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<zuan::ZuanDialect, vp::VPDialect, scf::SCFDialect,
                  arith::ArithDialect, memref::MemRefDialect>();
}

void ConvertZuanToVPPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertZuanToVPPattern>(&getContext(), vf, scalable);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ConvertZuanToVPPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<zuan::ZuanDialect, vp::VPDialect, scf::SCFDialect,
                  arith::ArithDialect, memref::MemRefDialect>();
}

void registerZuanStripminingPass() { PassRegistration<ZuanStripminingPass>(); }
void registerConvertZuanToVPPass() { PassRegistration<ConvertZuanToVPPass>(); }

} // namespace zuan
} // namespace mlir
