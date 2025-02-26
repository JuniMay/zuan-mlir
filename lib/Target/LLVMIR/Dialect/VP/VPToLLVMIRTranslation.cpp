//===- VPToLLVMIRTranslation.cpp - VP to LLVM IR translation --------------===//
//
// This file implements the translation of VP intrinsics to LLVM IR.
//
//===----------------------------------------------------------------------===//

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

    auto &ctx = moduleTranslation.getLLVMContext();
    if (auto op = dyn_cast<vp::VPIntrFCmpOp>(opInst)) {
      auto operands = moduleTranslation.lookupValues(op.getOperands());
      auto predAttr = op->getAttrOfType<StringAttr>("predicate").getValue();
      auto predMetadata = llvm::MDString::get(ctx, predAttr);
      auto predValue = llvm::MetadataAsValue::get(ctx, predMetadata);
      // The metatdata is #2 in the operand list.
      operands.insert(operands.begin() + 2, predValue);
      SmallVector<llvm::Type *> overloadedTypes{operands[0]->getType()};
      auto inst = LLVM::detail::createIntrinsicCall(
          builder, llvm::Intrinsic::vp_fcmp, operands, overloadedTypes);
      moduleTranslation.mapValue(op.getResult(), inst);
      return success();
    }

    if (auto op = dyn_cast<vp::VPIntrICmpOp>(opInst)) {
      auto operands = moduleTranslation.lookupValues(op.getOperands());
      auto predAttr = op->getAttrOfType<StringAttr>("predicate").getValue();
      auto predMetadata = llvm::MDString::get(ctx, predAttr);
      auto predValue = llvm::MetadataAsValue::get(ctx, predMetadata);
      // The metatdata is #2 in the operand list.
      operands.insert(operands.begin() + 2, predValue);
      SmallVector<llvm::Type *> overloadedTypes{operands[0]->getType()};
      auto inst = LLVM::detail::createIntrinsicCall(
          builder, llvm::Intrinsic::vp_icmp, operands, overloadedTypes);
      moduleTranslation.mapValue(op.getResult(), inst);
      return success();
    }

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