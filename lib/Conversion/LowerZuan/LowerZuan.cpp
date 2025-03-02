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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"

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

    // Check all the type shape <= targetRank.
    bool unrolled = true;
    op->walk([&](ZuanUnrollingInterface iface) {
      auto shape = iface.getShapeToUnroll(shapeInfo);
      if (shape->size() > targetRank) {
        unrolled = false;
        return WalkResult::interrupt();
      } else {
        return WalkResult::advance();
      }
    });

    if (!unrolled) {
      return rewriter.notifyMatchFailure(op, "already unrolled");
    }

    // TODO
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
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getZeroAttr(elementType));

    auto resShape = *shapeInfo.getShapeWithEquivalence(op.getResult());
    auto ofrShape = llvm::map_to_vector(resShape, [&](DimSize dim) {
      return dim.getOrCreateOpFoldResult(rewriter, loc);
    });
    Value acc = rewriter.create<zuan::SplatOp>(loc, zero, ofrShape);
    Value ub = lhsShape.back().getOrCreateValue(rewriter, loc);
    Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lb, ub, step, acc,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          auto accIter = iterArgs[0];

          UnrollState state;
          SetVector<Value> valuesDefinedAbove;
          mlir::getUsedValuesDefinedAbove(dynamicOp.getBody(),
                                          valuesDefinedAbove);
          state.valueMap.map(valuesDefinedAbove.getArrayRef(),
                             valuesDefinedAbove.getArrayRef());

          // Unrolling on the last dimension, regardless of the rank.
          UnrollOptions options(iv, b.getIndexAttr(1), lhsShape.size() - 1);
          options.overrideReduceUnitDim(true);
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

            auto outerOp = b.create<MaskOp>(
                loc, type, mask, [&](OpBuilder &b, Location loc) {
                  Value outer = b.create<OuterOp>(loc, CombiningKind::MUL,
                                                  unrolledLhs, unrolledRhs);
                  b.create<MaskYieldOp>(loc, outer);
                });
            Value outer = outerOp.getResult(0);
            auto maskOp = b.create<MaskOp>(
                loc, type, mask,
                [&](OpBuilder &b, Location loc) {
                  Value add;
                  if (isa<FloatType>(elementType)) {
                    add = b.create<arith::AddFOp>(loc, accIter, outer);
                  } else {
                    add = b.create<arith::AddIOp>(loc, accIter, outer);
                  }
                  b.create<MaskYieldOp>(loc, add);
                },
                maskedoff);
            res = maskOp.getResult(0);
          } else {
            res = b.create<OuterOp>(loc, CombiningKind::MUL, unrolledLhs,
                                    unrolledRhs);
            if (isa<FloatType>(elementType)) {
              res = b.create<arith::AddFOp>(loc, accIter, res);
            } else {
              res = b.create<arith::AddIOp>(loc, accIter, res);
            }
          }

          b.create<scf::YieldOp>(loc, res);
        });

    // op->getParentOfType<func::FuncOp>().dump();

    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

} // namespace

void LowerZuanPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ZuanLowerMatmulPattern>(&getContext());

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
