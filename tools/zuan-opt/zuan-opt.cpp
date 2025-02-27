#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Conversion/LinalgToZuan.h"
#include "Conversion/VPToLLVM.h"
#include "VP/IR/VP.h"
#include "Zuan/IR/Zuan.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::vp::registerConvertVPToLLVMPass();
  mlir::zuan::registerConvertLinalgToZuanPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::vp::VPDialect>();
  registry.insert<mlir::zuan::ZuanDialect>();
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Adaptiv optimizer driver\n", registry));
}
