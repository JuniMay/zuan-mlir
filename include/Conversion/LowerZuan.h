#ifndef CONVERSION_LOWERZUAN_H
#define CONVERSION_LOWERZUAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
namespace mlir {
namespace zuan {

struct LowerZuanPass : PassWrapper<LowerZuanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerZuanPass);

  StringRef getArgument() const override { return "lower-zuan"; }
  StringRef getDescription() const override {
    return "Lower Zuan operations to target-ranked operations";
  }

  LowerZuanPass() = default;
  LowerZuanPass(const LowerZuanPass &) {}

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;

  Option<unsigned> targetRank{
      *this, "target-rank",
      llvm::cl::desc("The target rank of the lowered operations, minimum 2"),
      llvm::cl::init(2)};
};

void registerLowerZuanPass();

} // namespace zuan
} // namespace mlir

#endif // CONVERSION_LOWERZUAN_H