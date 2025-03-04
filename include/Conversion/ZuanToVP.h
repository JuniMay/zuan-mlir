#ifndef CONVERSION_ZUANTOVP_H
#define CONVERSION_ZUANTOVP_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
namespace mlir {
namespace zuan {

struct ZuanStripminingPass
    : PassWrapper<ZuanStripminingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZuanStripminingPass);

  StringRef getArgument() const override { return "zuan-stripmining"; }
  StringRef getDescription() const override {
    return "Stripmine & Tiling Zuan operations";
  }

  ZuanStripminingPass() = default;
  ZuanStripminingPass(const ZuanStripminingPass &) {}

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
};

struct ConvertZuanToVPPass
    : PassWrapper<ConvertZuanToVPPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertZuanToVPPass);

  StringRef getArgument() const override { return "convert-zuan-to-vp"; }
  StringRef getDescription() const override {
    return "Convert Zuan dialect to VP dialect";
  }

  ConvertZuanToVPPass() = default;
  ConvertZuanToVPPass(const ConvertZuanToVPPass &) {}

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

void registerZuanStripminingPass();
void registerConvertZuanToVPPass();

} // namespace zuan
} // namespace mlir

#endif // CONVERSION_ZUANTOVP_H
