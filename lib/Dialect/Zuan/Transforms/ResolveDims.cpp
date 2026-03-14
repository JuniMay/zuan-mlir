#include "Zuan/Transforms/ResolveDims.h"

#include "Zuan/IR/Zuan.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace zuan {

void ResolveZuanDimsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  DimOp::getCanonicalizationPatterns(patterns, &getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ResolveZuanDimsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<zuan::ZuanDialect>();
}

void registerResolveZuanDimsPass() {
  PassRegistration<ResolveZuanDimsPass>();
}

} // namespace zuan
} // namespace mlir
