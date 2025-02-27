//===- LinalgToZuan.h - Linalg to Zuan conversion pass ----------*- C++ -*-===//
//
// This file declares the Linalg to Zuan dialect conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_LINALGTOZUAN_H
#define CONVERSION_LINALGTOZUAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
namespace mlir {
namespace zuan {

void populateLinalgToZuanConversionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns);

struct ConvertLinalgToZuanPass
    : PassWrapper<ConvertLinalgToZuanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToZuanPass);

  StringRef getArgument() const final { return "convert-linalg-to-zuan"; }
  StringRef getDescription() const final {
    return "Convert Linalg dialect to Zuan dialect";
  }
  
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};

void registerConvertLinalgToZuanPass();

} // namespace zuan
} // namespace mlir

#endif // CONVERSION_LINALGTOZUAN_H
