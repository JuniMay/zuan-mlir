#ifndef DIALECT_ZUAN_TRANSFORMS_STRENGTHREDUCTION_H
#define DIALECT_ZUAN_TRANSFORMS_STRENGTHREDUCTION_H

#include "Zuan/IR/Zuan.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
namespace zuan {

void populateZuanStrengthReductionPatterns(RewritePatternSet &patterns);

struct ZuanStrengthReductionPass
    : PassWrapper<ZuanStrengthReductionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZuanStrengthReductionPass)

  StringRef getArgument() const final { return "zuan-strength-reduction"; }
  StringRef getDescription() const final {
    return "Perform strength reduction on Zuan operations";
  }

  ZuanStrengthReductionPass() = default;
  ZuanStrengthReductionPass(const ZuanStrengthReductionPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ZuanDialect>();
  }
};

void registerZuanStrengthReductionPass();


} // namespace zuan
} // namespace mlir

#endif // DIALECT_ZUAN_TRANSFORMS_STRENGTHREDUCTION_H
