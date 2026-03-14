#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "VP/IR/VP.h"
#include "Dyno/IR/Dyno.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::vp::VPDialect>();
  registry.insert<mlir::dyno::DynoDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
