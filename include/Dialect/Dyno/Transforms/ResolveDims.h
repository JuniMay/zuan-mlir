#ifndef DIALECT_DYNO_TRANSFORMS_RESOLVEDIMS_H
#define DIALECT_DYNO_TRANSFORMS_RESOLVEDIMS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
namespace dyno {

struct ResolveDynoDimsPass
    : PassWrapper<ResolveDynoDimsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveDynoDimsPass)

  StringRef getArgument() const override { return "resolve-dyno-dims"; }
  StringRef getDescription() const override {
    return "Resolve remaining dyno.dim queries into concrete index values";
  }

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};

void registerResolveDynoDimsPass();

} // namespace dyno
} // namespace mlir

#endif // DIALECT_DYNO_TRANSFORMS_RESOLVEDIMS_H
