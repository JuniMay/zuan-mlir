#include "Dyno/Transforms/ResolveDims.h"

#include "Dyno/IR/Dyno.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace dyno {

void ResolveDynoDimsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  DimOp::getCanonicalizationPatterns(patterns, &getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

void ResolveDynoDimsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<dyno::DynoDialect>();
}

void registerResolveDynoDimsPass() {
  PassRegistration<ResolveDynoDimsPass>();
}

} // namespace dyno
} // namespace mlir
