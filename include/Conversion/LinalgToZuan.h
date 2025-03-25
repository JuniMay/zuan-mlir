//===- LinalgToZuan.h - Linalg to Zuan conversion pass ----------*- C++ -*-===//
//
// This file declares the Linalg to Zuan dialect conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_LINALGTOZUAN_H
#define CONVERSION_LINALGTOZUAN_H

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
namespace zuan {

struct LinalgConversionState {
  IRMapping valueMap;
  // The transformed input & init buffers
  DenseMap<Value, Value> transformedMemRefs;

  /// Common shapes.
  SmallVector<OpFoldResult> ofrShape;
  SmallVector<int64_t> staticShape;

  Block *dynamicBlock;
  Block *yieldBlock;

  linalg::LinalgOp linalgOp;

  /// The converted values of scf if condition.
  SmallVector<Value> masks;

  /// The indices for gather/scatter.
  DenseMap<OpOperand*, SmallVector<Value>> nonProjectedPermutationIndices;

  LinalgConversionState() = default;

  LinalgConversionState(SmallVector<OpFoldResult> ofrShape, Block *dynamicBlock,
                        Block *yieldBlock, linalg::LinalgOp linalgOp);

  void pushMask(Value mask) { masks.push_back(mask); }
  void popMask() { masks.pop_back(); }

  std::optional<Value> getMask() {
    if (masks.empty()) {
      return std::nullopt;
    }
    return masks.back();
  }
};

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
