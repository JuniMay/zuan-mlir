#ifndef DIALECT_DYNO_TRANSFORMS_STRENGTHREDUCTION_H
#define DIALECT_DYNO_TRANSFORMS_STRENGTHREDUCTION_H

#include "Dyno/IR/Dyno.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
namespace dyno {

void populateDynoStrengthReductionPatterns(RewritePatternSet &patterns);

struct DynoStrengthReductionPass
    : PassWrapper<DynoStrengthReductionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DynoStrengthReductionPass)

  StringRef getArgument() const final { return "dyno-strength-reduction"; }
  StringRef getDescription() const final {
    return "Perform strength reduction on Dyno operations";
  }

  DynoStrengthReductionPass() = default;
  DynoStrengthReductionPass(const DynoStrengthReductionPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<DynoDialect>();
  }
};

void registerDynoStrengthReductionPass();


} // namespace dyno
} // namespace mlir

#endif // DIALECT_DYNO_TRANSFORMS_STRENGTHREDUCTION_H
