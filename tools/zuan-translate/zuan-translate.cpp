#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "Target/LLVMIR/Dialect/VP/VPToLLVMIRTranslation.h"

namespace mlir {
void registerAdaptivToLLVMIRTranslation();
}

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::TranslateFromMLIRRegistration withdescription(
      "zuan-to-llvmir", "translate MLIR generated with zuan to LLVMIR",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule) {
          return mlir::failure();
        }

        llvmModule->print(output, nullptr);
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
        mlir::registerAllToLLVMIRTranslations(registry);
        mlir::vp::registerVPDialectTranslation(registry);
      });

  return failed(mlir::mlirTranslateMain(argc, argv, "Zuan Translation Tool"));
}
