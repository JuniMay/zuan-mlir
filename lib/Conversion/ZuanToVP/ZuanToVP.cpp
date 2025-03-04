#include "Conversion/ZuanToVP.h"
#include "VP/IR/VP.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/Unrolling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

    if (dims.size() != 1 && dims[0] != 0 && tile.getType().getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "not 1-D reduction");
    }

    auto loc = op.getLoc();
    auto dynamicOp = op->getParentOfType<DynamicOp>();

    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    shapeInfo.inferShape(dynamicOp, shapeInferenceState);

    shapeInfo.dump(llvm::dbgs());

    auto tileShape = *shapeInfo.getShape(tile);
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
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Type type = tile.getType().getElementType();
    Value dim = tileShape[0].getOrCreateValue(rewriter, loc);
    Value vlmax = rewriter.create<vp::GetVLOp>(loc, dim, vf, type, scalable);

    Value zeroElem = rewriter.create<arith::ConstantOp>(
        loc, type, rewriter.getZeroAttr(type));

    auto newDynamicOp =
        rewriter.create<DynamicOp>(loc, type, dynamicOp.getInits(), nullptr);
    state.valueMap.map(dynamicOp.getBody().getArguments(),
                       newDynamicOp.getBody().getArguments());
    rewriter.setInsertionPointToStart(&newDynamicOp.getBody().front());

    SmallVector<OpFoldResult> sizes{vlmax};
    Value initAcc = rewriter.create<zuan::SplatOp>(loc, zeroElem, sizes);
    SmallVector<Value> inits = {dim, zero, initAcc};
    SmallVector<Type> resultTypes = {dim.getType(), zero.getType(),
                                     initAcc.getType()};

    auto whileOp = rewriter.create<scf::WhileOp>(
        loc, resultTypes, inits,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto avl = args[0];
          Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               avl, zero);
          b.create<scf::ConditionOp>(loc, cond, args);
        },
        [&](OpBuilder &b, Location loc, ValueRange args) -> void {
          auto avl = args[0];
          auto idx = args[1];
          auto acc = args[2];

          Value vl = rewriter.create<vp::GetVLOp>(loc, avl, vf, type, scalable);
          UnrollOptions options(idx, vl, 0, true);

          auto newSourceTile = getUnrolledValue(rewriter, tile, options, state);
          auto newAcc = createCombiningOp(rewriter, loc, op.getKind(), acc,
                                          newSourceTile);
          newAcc.getDefiningOp()->setAttr("zuan_passthru_operand",
                                          b.getIndexAttr(0));

          Value newAvl = b.create<arith::SubIOp>(loc, avl, vl);
          Value newIdx = b.create<arith::AddIOp>(loc, idx, vl);
          b.create<scf::YieldOp>(loc, ValueRange{newAvl, newIdx, newAcc});
        });

    auto finalAcc = whileOp->getResult(2);
    Value finalRed = rewriter.create<zuan::ReductionOp>(
        loc, op.getKind(), finalAcc, dims, op.getInit());

    rewriter.create<YieldOp>(loc, finalRed, nullptr);

    rewriter.setInsertionPoint(op);
    Value splat = rewriter.create<zuan::SplatOp>(
        loc, newDynamicOp->getResult(0), ArrayRef<int64_t>{});
    rewriter.replaceOp(op, splat);

    return success();
  }

private:
  unsigned vf;
  bool scalable;
};

} // namespace

void ZuanStripminingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ZuanStripminingReduction1DPattern>(&getContext(), vf, scalable);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ZuanStripminingPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<zuan::ZuanDialect, vp::VPDialect, scf::SCFDialect,
                  arith::ArithDialect>();
}

void registerZuanStripminingPass() { PassRegistration<ZuanStripminingPass>(); }

} // namespace zuan
} // namespace mlir