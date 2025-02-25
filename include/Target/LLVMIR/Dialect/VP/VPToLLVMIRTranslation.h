#ifndef TARGET_LLVMIR_VP_VPTOLLVMIRTRANSLATION_H
#define TARGET_LLVMIR_VP_VPTOLLVMIRTRANSLATION_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
namespace mlir {
namespace vp {

void registerVPDialectTranslation(DialectRegistry &registry);
void registerVPDialectTranslation(MLIRContext &context);

} // namespace vp
} // namespace mlir

#endif