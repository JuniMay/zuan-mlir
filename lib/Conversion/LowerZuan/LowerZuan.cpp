#include "Conversion/LowerZuan.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Interfaces/ZuanUnrollingInterface.h"
#include "Zuan/Utils/ShapeInference.h"
#include "Zuan/Utils/Unrolling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <cassert>

namespace mlir {
namespace zuan {

namespace {

struct ZuanUnrollLeadingDimPattern : OpRewritePattern<DynamicOp> {
  explicit ZuanUnrollLeadingDimPattern(MLIRContext *context,
                                       unsigned targetRank)
      : OpRewritePattern<DynamicOp>(context), targetRank(targetRank) {}

  LogicalResult matchAndRewrite(DynamicOp op,
                                PatternRewriter &rewriter) const final {
    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    shapeInfo.inferShape(op, shapeInferenceState);

    splitDynamicOpForUnrolling(rewriter, op, 0, shapeInfo);

    auto yieldOp = op.getYieldOp();
    auto yieldRegion = &yieldOp.getRegion();

    if (yieldRegion->getOps().empty()) {
      // The scalars will not be unrolled, so no need to create a dummy loop for
      // them. Just handle it to other passes.
      return rewriter.notifyMatchFailure(
          op, "empty yield region, other patterns are needed");
    }

    // `LowerZuan` only ensures that tiles are reduced to at most `targetRank`.
    // It does not guarantee that a remaining 2-D tile has a static outer
    // dimension. The VP path relies on `-zuan-stripmining`/tiling to eliminate
    // dynamic outer 2-D cases before `ConvertToVP`; otherwise the generic loop
    // lowering in this pass is the fallback.
    if (isDynamicOpUnrolled(op, targetRank, shapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "expected all the shapes to be <= targetRank");
    }

    auto referenceOp = &*yieldRegion->getOps().begin();
    auto iface = dyn_cast<ZuanUnrollingInterface>(referenceOp);
    assert(iface && "expected an unrolling interface");

    auto loc = op->getLoc();

    auto shape = iface.getShapeToUnroll(shapeInfo);
    // All shapes are now the same and >= target-rank, so should be safe
    // to access the first.
    auto dim = (*shape)[0].getOrCreateValue(rewriter, loc);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    UnrollState state;
    state.initialize(op);

    // Make sure no use-before-def is produced.
    dim = getUnrolledValue(rewriter, dim, getCloneOptions(), state);

    auto loop = scf::ForOp::create(rewriter, 
        loc, zero, dim, one, ValueRange{},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          UnrollOptions options(iv, b.getIndexAttr(1), 0, true);
          op.unroll(b, options, state);
          scf::YieldOp::create(b, loc);
        });

    // loop->getParentOfType<func::FuncOp>().dump();
    rewriter.replaceOp(op, loop);

    return success();
  }

private:
  unsigned targetRank = 2;
};

struct ZuanLowerMatmulPattern : OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const final {
    OpBuilder::InsertionGuard guard(rewriter);

    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    auto dynamicOp = op->getParentOfType<DynamicOp>();
    shapeInfo.inferShape(dynamicOp, shapeInferenceState);

    std::optional<std::pair<Value, Value>> masks;
    if (auto maskOp = dyn_cast<MaskOp>(op->getParentOp())) {
      masks = std::make_pair(maskOp.getMask(), maskOp.getMaskedoff());
      rewriter.setInsertionPoint(maskOp);
    }

    auto loc = op.getLoc();

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsShape = *shapeInfo.getShapeWithEquivalence(lhs);
    auto rhsShape = *shapeInfo.getShapeWithEquivalence(rhs);

    auto lhsRank = lhs.getType().getRank();
    auto rhsRank = rhs.getType().getRank();

    auto elementType = lhs.getType().getElementType();
    auto zero = arith::ConstantOp::create(rewriter, 
        loc, elementType, rewriter.getZeroAttr(elementType));

    auto resShape = *shapeInfo.getShapeWithEquivalence(op.getResult());
    auto ofrShape = llvm::map_to_vector(resShape, [&](DimSize dim) {
      return dim.getOrCreateOpFoldResult(rewriter, loc);
    });
    Value acc = zuan::SplatOp::create(rewriter, loc, zero, ofrShape);
    Value ub = lhsShape.back().getOrCreateValue(rewriter, loc);
    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto forOp = scf::ForOp::create(rewriter, 
        loc, lb, ub, step, acc,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          auto accIter = iterArgs[0];

          UnrollState state{IRMapping{}, nullptr};
          state.initialize(dynamicOp);
          // Unrolling on the last dimension, regardless of the rank.
          UnrollOptions options(iv, b.getIndexAttr(1), lhsShape.size() - 1,
                                true);
          auto unrolledLhs = getUnrolledValue(b, lhs, options, state);

          if (rhsRank < lhsRank) {
            // m * k @ k * (1) => m * (1)
            options.overrideUnrollIdx(rhsShape.size() - 1);
          } else {
            options.overrideUnrollIdx(rhsShape.size() - 2);
          }
          auto unrolledRhs = getUnrolledValue(b, rhs, options, state);

          Value res;
          if (masks) {
            auto [mask, maskedoff] = *masks;
            auto type = op.getResult().getType();

            auto outerOp = MaskOp::create(b, 
                loc, type, mask, [&](OpBuilder &b, Location loc) {
                  Value outer = OuterOp::create(b, loc, CombiningKind::MUL,
                                                  unrolledLhs, unrolledRhs);
                  MaskYieldOp::create(b, loc, outer);
                });
            Value outer = outerOp.getResult(0);
            auto maskOp = MaskOp::create(b, 
                loc, type, mask,
                [&](OpBuilder &b, Location loc) {
                  Value add;
                  if (isa<FloatType>(elementType)) {
                    add = arith::AddFOp::create(b, loc, accIter, outer);
                  } else {
                    add = arith::AddIOp::create(b, loc, accIter, outer);
                  }
                  MaskYieldOp::create(b, loc, add);
                },
                maskedoff);
            res = maskOp.getResult(0);
          } else {
            res = OuterOp::create(b, loc, CombiningKind::MUL, unrolledLhs,
                                    unrolledRhs);
            if (isa<FloatType>(elementType)) {
              res = arith::AddFOp::create(b, loc, accIter, res);
            } else {
              res = arith::AddIOp::create(b, loc, accIter, res);
            }
          }

          scf::YieldOp::create(b, loc, res);
        });

    // op->getParentOfType<func::FuncOp>().dump();

    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

struct ZuanLowerReductionPattern : OpRewritePattern<ReductionOp> {
  using OpRewritePattern<ReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReductionOp op,
                                PatternRewriter &rewriter) const final {
    SmallVector<int64_t> reductionDims{op.getDims()};

    if (op.getTile().getType().getRank() == 0) {
      return rewriter.notifyMatchFailure(op, "0-D corner case is ignored");
    }
    if (reductionDims.size() == 1 && reductionDims[0] == 0 &&
        op.getTile().getType().getRank() == 1) {
      // 1-D reduction, ignore for now.
      return rewriter.notifyMatchFailure(op, "1-D reduction is ignored");
    }

    OpBuilder::InsertionGuard guard(rewriter);

    ShapeInfo shapeInfo;
    ShapeInferenceState shapeInferenceState;
    auto dynamicOp = op->getParentOfType<DynamicOp>();
    shapeInfo.inferShape(dynamicOp, shapeInferenceState);

    Value mask = nullptr;
    if (auto maskOp = dyn_cast<MaskOp>(op->getParentOp())) {
      // TODO: Investigate if maskedoff should be used in reduction.
      // `vector.multi_reduction` does not support the passthru value. And in VP
      // dialect, masked off is also not supported. We already have an optional
      // init value, so maybe the maskedoff should be ignored.
      mask = maskOp.getMask();
      rewriter.setInsertionPoint(maskOp);
    }

    auto loc = op.getLoc();
    auto sourceShapeRef = *shapeInfo.lookupShape(op.getTile());
    auto resultShapeRef = *shapeInfo.lookupShape(op.getResult());

    SmallVector<OpFoldResult> resultShape = resultShapeRef.reify(rewriter, loc);

    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto elementType = op.getTile().getType().getElementType();
    auto acc = op.getInit();
    if (!acc) {
      auto zero = arith::ConstantOp::create(rewriter, 
          loc, elementType, rewriter.getZeroAttr(elementType));
      acc = zuan::SplatOp::create(rewriter, loc, zero, resultShape);
    }

    // sort reduction dims in descending order, so that expanding leading dims
    // does not effect the dim idx;
    llvm::sort(reductionDims.begin(), reductionDims.end(),
               std::greater<int64_t>());

    Value currAcc = acc;
    Value currSource = op.getTile();
    Value currMask = mask;

    SmallVector<scf::ForOp> loops;
    SmallVector<Value> valuesToYield;

    UnrollState state{IRMapping{}, nullptr};
    state.initialize(dynamicOp);

    for (auto dim : reductionDims) {
      auto ub = sourceShapeRef[dim].getOrCreateValue(rewriter, loc);
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
      valuesToYield.push_back(loop.getResult(0));
    }

    // Now at the innermost loop.
    auto partialReduced =
        createCombiningOp(rewriter, loc, op.getKind(), currAcc, currSource);
    if (currMask) {
      partialReduced =
          maskOperation(rewriter, loc, partialReduced.getDefiningOp(), currMask,
                        currAcc)
              ->getResult(0);
    }
    scf::YieldOp::create(rewriter, loc, partialReduced);
    for (size_t i = 0; i < loops.size() - 1; ++i) {
      rewriter.setInsertionPointToEnd(loops[i].getBody());
      scf::YieldOp::create(rewriter, loc, valuesToYield[i + 1]);
    }
    rewriter.replaceOp(op, loops.front());
    // loops.front()->getParentOfType<func::FuncOp>().dump();
    return success();
  }
};

} // namespace

void LowerZuanPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ZuanLowerMatmulPattern, ZuanLowerReductionPattern>(
      &getContext());
  patterns.add<ZuanUnrollLeadingDimPattern>(&getContext(), targetRank);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
