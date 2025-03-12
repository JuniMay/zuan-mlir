#include "Zuan/Transforms/StrengthReduction.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/ConvertToVP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <utility>

namespace mlir {
namespace zuan {

namespace {

struct StepToSplatPattern : OpRewritePattern<StepOp> {
  using OpRewritePattern<StepOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StepOp op,
                                PatternRewriter &rewriter) const override {
    auto dim = op.getDim().getZExtValue();
    auto start = op.getStart();
    auto sizes = op.getMixedSizes();

    auto stepSize = sizes[dim];

    if (auto cst = stepSize.dyn_cast<Attribute>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(cst)) {
        if (intAttr.getInt() != 1) {
          return rewriter.notifyMatchFailure(op, "non-unit step size");
        }
      }
    } else if (auto cst = stepSize.dyn_cast<Value>()) {
      if (auto defOp = cst.getDefiningOp<arith::ConstantOp>()) {
        auto step = cast<IntegerAttr>(defOp.getValue()).getInt();
        if (step != 1) {
          return rewriter.notifyMatchFailure(op, "non-unit step size");
        }
      } else {
        return rewriter.notifyMatchFailure(op, "non-constant step size");
      }
    } else {
      llvm_unreachable("unexpected OpFoldResult type");
    }

    rewriter.replaceOpWithNewOp<SplatOp>(op, start, sizes);
    return success();
  }
};

/// Convert splat-cast to cast-splat.
struct SplatCastPattern : OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto source = op.getTile();
    auto result = op.getResult();

    if (auto splatOp = source.getDefiningOp<SplatOp>()) {
      auto splatValue = splatOp.getValue();
      auto resultType = result.getType().getElementType();
      auto cast = createCastOp(rewriter, op.getLoc(), op.getKind(), resultType,
                               splatValue);
      auto newSplat =
          rewriter.create<SplatOp>(op.getLoc(), cast, splatOp.getMixedDims());
      rewriter.replaceOp(op, newSplat);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "non-splat source");
  }
};

struct StepCastPattern : OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto source = op.getTile();
    auto result = op.getResult();

    if (auto stepOp = source.getDefiningOp<StepOp>()) {
      auto startValue = stepOp.getStart();
      auto resultType = result.getType().getElementType();
      auto cast = createCastOp(rewriter, op.getLoc(), op.getKind(), resultType,
                               startValue);
      auto newStep = rewriter.create<StepOp>(op.getLoc(), cast,
                                             stepOp.getDim().getZExtValue(),
                                             stepOp.getMixedSizes());
      rewriter.replaceOp(op, newStep);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "non-step source");
  }
};

struct SplatElementwisePattern : OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern<OpTrait::Elementwise>::OpTraitRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<OpTrait::SameOperandsAndResultType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "requires same operand/result type");
    }

    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
      return rewriter.notifyMatchFailure(op, "requires no regions/successors");
    }

    if (op->getNumOperands() == 0) {
      return rewriter.notifyMatchFailure(op, "requires at least one operand");
    }

    if (op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(op, "requires single result");
    }

    if (op->hasAttr("zuan_passthru_operand")) {
      return rewriter.notifyMatchFailure(op, "got passthru operand");
    }

    SmallVector<Value> newOperands;
    SmallVector<OpFoldResult> splatDims;
    for (auto operand : op->getOperands()) {
      if (auto splatOp = operand.getDefiningOp<SplatOp>()) {
        newOperands.push_back(splatOp.getValue());
        splatDims = splatOp.getMixedDims();
      } else {
        return rewriter.notifyMatchFailure(op, "non-splat operand");
      }
    }

    assert(!newOperands.empty() && "unexpected empty operand list");

    auto commonType = newOperands.front().getType();
    auto numResults = op->getNumResults();
    SmallVector<Type> resultTypes(numResults, commonType);

    auto newOp = rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                                 newOperands, resultTypes, op->getAttrs());
    auto newValue = newOp->getResult(0);
    auto splat = rewriter.create<SplatOp>(op->getLoc(), newValue, splatDims);
    rewriter.replaceOp(op, splat);
    return success();
  }
};

} // namespace

void populateZuanStrengthReductionPatterns(RewritePatternSet &patterns) {
  patterns.add<StepToSplatPattern>(patterns.getContext());
  patterns.add<SplatCastPattern, StepCastPattern>(patterns.getContext());
  patterns.add<SplatElementwisePattern>(patterns.getContext());
}

void ZuanStrengthReductionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateZuanStrengthReductionPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void registerZuanStrengthReductionPass() {
  PassRegistration<ZuanStrengthReductionPass>();
}

} // namespace zuan
} // namespace mlir
