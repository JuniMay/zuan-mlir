#include "Zuan/Transforms/StrengthReduction.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/ConvertToVP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
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

/// Canonicalize addition of step and splat to a single step with modified
/// start.
struct SplatStepAddPattern : OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // step + splat -> step with start as splat + old start
    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);

    SplatOp splat;
    StepOp step;

    if (auto splatOp = lhs.getDefiningOp<SplatOp>()) {
      splat = splatOp;
    } else if (auto splatOp = rhs.getDefiningOp<SplatOp>()) {
      splat = splatOp;
    } else {
      return rewriter.notifyMatchFailure(op, "non-splat operand");
    }

    if (auto stepOp = lhs.getDefiningOp<StepOp>()) {
      step = stepOp;
    } else if (auto stepOp = rhs.getDefiningOp<StepOp>()) {
      step = stepOp;
    } else {
      return rewriter.notifyMatchFailure(op, "non-step operand");
    }

    auto splatted = splat.getValue();
    auto start = step.getStart();

    auto newStart =
        rewriter.create<arith::AddIOp>(op.getLoc(), splatted, start);
    auto newStep = rewriter.create<StepOp>(op.getLoc(), newStart,
                                           step.getDim().getZExtValue(),
                                           step.getMixedSizes());
    rewriter.replaceOp(op, newStep);
    return success();
  }
};

/// Convert gather to load when all indices are splats or steps along the
/// corresponding dimension.
struct GatherToLoadPattern : OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    // Each index:
    // - splat -> subview with splatted value as offset, 0 as stride
    // - step or scaled step -> contiguous or strided
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    // Leading indices are all splats, the last index is a step.
    auto indices = op.getIndices();
    auto memref = op.getBase();

    if (indices.empty()) {
      // load the scalar and splat with empty sizes.
      auto load =
          rewriter.create<memref::LoadOp>(op.getLoc(), memref, ValueRange{});
      auto splat =
          rewriter.create<SplatOp>(op.getLoc(), load, ArrayRef<int64_t>());
      rewriter.replaceOp(op, splat);
      return success();
    }

    // Check leading indices.
    for (unsigned i = 0, e = indices.size(); i < e; ++i) {
      auto index = indices[i];
      if (auto splatOp = index.getDefiningOp<SplatOp>()) {
        auto splatSizes = splatOp.getMixedDims();

        if (splatSizes.size() != indices.size()) {
          return rewriter.notifyMatchFailure(
              op, "splat rank and memref rank mismatch");
        }

        offsets.push_back(splatOp.getValue());
        sizes.push_back(splatSizes[i]);
        strides.push_back(rewriter.getIndexAttr(0));
      } else if (auto stepOp = index.getDefiningOp<StepOp>()) {
        auto stepSizes = stepOp.getMixedSizes();
        auto dim = stepOp.getDim();
        auto start = stepOp.getStart();

        if (stepSizes.size() != indices.size()) {
          return rewriter.notifyMatchFailure(
              op, "step rank and memref rank mismatch");
        }

        if (dim.getZExtValue() != i) {
          return rewriter.notifyMatchFailure(op,
                                             "not step along the dimension");
        }

        offsets.push_back(start);
        sizes.push_back(stepSizes[i]);
        strides.push_back(rewriter.getIndexAttr(1));
      } else {
        // TODO: support scaled step as stride.
        return rewriter.notifyMatchFailure(op, "non-splat index");
      }
    }
    // Create subview.
    auto subview = rewriter.create<memref::SubViewOp>(op.getLoc(), memref,
                                                      offsets, sizes, strides);
    auto load = rewriter.create<LoadOp>(op.getLoc(), subview);

    rewriter.replaceOp(op, load);
    return success();
  }
};

} // namespace

void populateZuanStrengthReductionPatterns(RewritePatternSet &patterns) {
  patterns
      .add<StepToSplatPattern, SplatCastPattern, StepCastPattern,
           SplatElementwisePattern, SplatStepAddPattern, GatherToLoadPattern>(
          patterns.getContext());
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
