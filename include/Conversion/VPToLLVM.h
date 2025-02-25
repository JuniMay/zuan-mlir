#ifndef CONVERSION_VPTOLLVM_H
#define CONVERSION_VPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace vp {

void populateVPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns,
                                        bool enableRVV);
void configureVPToLLVMConversionLegality(LLVMConversionTarget &target);

struct ConvertVPToLLVMPass
    : PassWrapper<ConvertVPToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVPToLLVMPass)

  StringRef getArgument() const final { return "convert-vp-to-llvm"; }
  StringRef getDescription() const final {
    return "Convert VP dialect to LLVM dialect";
  }
  ConvertVPToLLVMPass() = default;
  ConvertVPToLLVMPass(const ConvertVPToLLVMPass &) {}

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;

  Option<unsigned> indexBitwidth{
      *this, "index-bitwidth",
      llvm::cl::desc(
          "Bitwidth of the index type, 0 to use size of machine word"),
      llvm::cl::init(kDeriveIndexBitwidthFromDataLayout)};

  Option<bool> enableRVV{
      *this, "enable-rvv",
      llvm::cl::desc("Enable conversion to RVV-specific intrinsics"),
      llvm::cl::init(false)};
};

void registerConvertVPToLLVMPass();

} // namespace vp
} // namespace mlir

#endif // CONVERSION_VPTOLLVM_H