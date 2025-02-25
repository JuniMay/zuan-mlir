
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsRISCV.h"

#include "Target/LLVMIR/Dialect/VP/VPToLLVMIRTranslation.h"
#include "VP/IR/VP.h"

using namespace mlir;

namespace {

struct VPDialectLLVMIRTranslationInterface : LLVMTranslationDialectInterface {

  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    Operation &opInst = *op;
#include "VP/IR/VPOpsConversions.inc"

    return failure();
  }
};
} // namespace

namespace mlir {
namespace vp {

void registerVPDialectTranslation(DialectRegistry &registry) {
  registry.insert<VPDialect>();
  registry.addExtension(+[](MLIRContext *ctx, VPDialect *dialect) {
    dialect->addInterfaces<VPDialectLLVMIRTranslationInterface>();
  });
}

void registerVPDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerVPDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}

} // namespace vp
} // namespace mlir