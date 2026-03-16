#ifndef CONVERSION_DYNOTOVP_H
#define CONVERSION_DYNOTOVP_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include <string>
namespace mlir {
namespace dyno {

struct DynoStripminingPass
    : PassWrapper<DynoStripminingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DynoStripminingPass);

  StringRef getArgument() const override { return "dyno-stripmining"; }
  StringRef getDescription() const override {
    return "Stripmine & Tiling Dyno operations";
  }

  DynoStripminingPass() = default;
  DynoStripminingPass(const DynoStripminingPass &) {}

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;

  /// Vectorization factor to control the scalable vectors. In RVV, this
  /// determines the LMUL.
  Option<unsigned> vf{*this, "vf", llvm::cl::desc("Vectorization factor"),
                      llvm::cl::init(4)};
  /// 2-D operations will be decomposed into multiple 1-D operations. This
  /// factor controls the size of the major dimension.
  Option<unsigned> uf{*this, "uf", llvm::cl::desc("Unroll factor"),
                      llvm::cl::init(4)};
  /// If use scalable vectors.
  Option<bool> scalable{*this, "scalable",
                        llvm::cl::desc("Use scalable vectors"),
                        llvm::cl::init(true)};
  Option<std::string> fpPolicy{
      *this, "fp-policy",
      llvm::cl::desc(
          "Fallback floating-point reduction policy: strict, relaxed"),
      llvm::cl::init("strict")};
  Option<std::string> reductionMode{
      *this, "reduction-mode",
      llvm::cl::desc("1-D reduction lowering mode: auto, sequential, parallel"),
      llvm::cl::init("auto")};
};

struct ConvertDynoToVPPass
    : PassWrapper<ConvertDynoToVPPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertDynoToVPPass);

  StringRef getArgument() const override { return "convert-dyno-to-vp"; }
  StringRef getDescription() const override {
    return "Convert Dyno dialect to VP dialect";
  }

  ConvertDynoToVPPass() = default;
  ConvertDynoToVPPass(const ConvertDynoToVPPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override;

private:
  /// Vectorization factor to control the scalable vectors. In RVV, this
  /// determines the LMUL.
  Option<unsigned> vf{*this, "vf", llvm::cl::desc("Vectorization factor"),
                      llvm::cl::init(4)};
  /// If use scalable vectors.
  Option<bool> scalable{*this, "scalable",
                        llvm::cl::desc("Use scalable vectors"),
                        llvm::cl::init(true)};
};

void registerDynoStripminingPass();
void registerConvertDynoToVPPass();

} // namespace dyno
} // namespace mlir

#endif // CONVERSION_DYNOTOVP_H
