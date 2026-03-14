#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllExtensions.h"

#include "Conversion/LinalgToDyno.h"
#include "Conversion/LowerDyno.h"
#include "Conversion/VPToLLVM.h"
#include "Conversion/DynoToVP.h"
#include "VP/IR/VP.h"
#include "Dyno/IR/Dyno.h"
#include "Dyno/Transforms/ResolveDims.h"
#include "Dyno/Transforms/StrengthReduction.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::vp::registerConvertVPToLLVMPass();
  mlir::dyno::registerConvertLinalgToDynoPass();
  mlir::dyno::registerLowerDynoPass();
  mlir::dyno::registerDynoStripminingPass();
  mlir::dyno::registerConvertDynoToVPPass();
  mlir::dyno::registerResolveDynoDimsPass();
  mlir::dyno::registerDynoStrengthReductionPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::vp::VPDialect>();
  registry.insert<mlir::dyno::DynoDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Dyno optimizer driver\n", registry));
}
