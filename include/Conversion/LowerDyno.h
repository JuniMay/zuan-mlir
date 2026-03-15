#ifndef CONVERSION_LOWERDYNO_H
#define CONVERSION_LOWERDYNO_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
namespace mlir {
namespace dyno {

struct LowerDynoPass : PassWrapper<LowerDynoPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDynoPass);

  StringRef getArgument() const override { return "lower-dyno"; }
  StringRef getDescription() const override {
    return "Structurally slice Dyno effect roots to a target rank";
  }

  LowerDynoPass() = default;
  LowerDynoPass(const LowerDynoPass &) {}

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;

  Option<unsigned> targetRank{
      *this, "target-rank",
      llvm::cl::desc("The target rank of the lowered operations, minimum 2"),
      llvm::cl::init(2)};
};

void registerLowerDynoPass();

} // namespace dyno
} // namespace mlir

#endif // CONVERSION_LOWERDYNO_H
