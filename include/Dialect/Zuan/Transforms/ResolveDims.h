#ifndef DIALECT_ZUAN_TRANSFORMS_RESOLVEDIMS_H
#define DIALECT_ZUAN_TRANSFORMS_RESOLVEDIMS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
namespace zuan {

struct ResolveZuanDimsPass
    : PassWrapper<ResolveZuanDimsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveZuanDimsPass)

  StringRef getArgument() const override { return "resolve-zuan-dims"; }
  StringRef getDescription() const override {
    return "Resolve remaining zuan.dim queries into concrete index values";
  }

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};

void registerResolveZuanDimsPass();

} // namespace zuan
} // namespace mlir

#endif // DIALECT_ZUAN_TRANSFORMS_RESOLVEDIMS_H
