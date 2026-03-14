//===- LinalgToDyno.h - Linalg to Dyno conversion pass ----------*- C++ -*-===//
//
// This file declares the Linalg to Dyno dialect conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_LINALGTODYNO_H
#define CONVERSION_LINALGTODYNO_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
namespace mlir {
namespace dyno {

struct LinalgConversionState {
  IRMapping valueMap;
  // The transformed input & init buffers
  DenseMap<Value, Value> transformedMemRefs;

  /// Common shapes.
  SmallVector<OpFoldResult> ofrShape;
  SmallVector<int64_t> staticShape;

  linalg::LinalgOp linalgOp;

  /// The converted values of scf if condition.
  SmallVector<Value> masks;

  /// The indices for gather/scatter.
  DenseMap<OpOperand*, SmallVector<Value>> nonProjectedPermutationIndices;

  LinalgConversionState() = default;

  LinalgConversionState(SmallVector<OpFoldResult> ofrShape,
                        linalg::LinalgOp linalgOp);

  void pushMask(Value mask) { masks.push_back(mask); }
  void popMask() { masks.pop_back(); }

  std::optional<Value> getMask() {
    if (masks.empty()) {
      return std::nullopt;
    }
    return masks.back();
  }
};

void populateLinalgToDynoConversionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns);

struct ConvertLinalgToDynoPass
    : PassWrapper<ConvertLinalgToDynoPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToDynoPass);

  StringRef getArgument() const final { return "convert-linalg-to-dyno"; }
  StringRef getDescription() const final {
    return "Convert Linalg dialect to Dyno dialect";
  }

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};

void registerConvertLinalgToDynoPass();

} // namespace dyno
} // namespace mlir

#endif // CONVERSION_LINALGTODYNO_H
