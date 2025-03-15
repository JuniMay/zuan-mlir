#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllExtensions.h"

#include "Conversion/LinalgToZuan.h"
#include "Conversion/LowerZuan.h"
#include "Conversion/VPToLLVM.h"
#include "Conversion/ZuanToVP.h"
#include "VP/IR/VP.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Transforms/StrengthReduction.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::vp::registerConvertVPToLLVMPass();
  mlir::zuan::registerConvertLinalgToZuanPass();
  mlir::zuan::registerLowerZuanPass();
  mlir::zuan::registerZuanStripminingPass();
  mlir::zuan::registerConvertZuanToVPPass();
  mlir::zuan::registerZuanStrengthReductionPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::vp::VPDialect>();
  registry.insert<mlir::zuan::ZuanDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Adaptiv optimizer driver\n", registry));
}
