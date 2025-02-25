//===- VPDialect.td - VP Dialect ---------------------------*- tablegen -*-===//
//
// This file declares the Vector Predication dialect.
//
//===----------------------------------------------------------------------===//

#ifndef VP_IR_VPDIALECT
#define VP_IR_VPDIALECT

include "mlir/IR/OpBase.td"

def VP_Dialect : Dialect {
  let name = "vp";
  let cppNamespace = "::mlir::vp";
  let summary = "High level presentations of vector predication operations";
  let description = [{
    Vector Predication (VP) dialect is a high level representation of LLVM
    VP intrinsics. Most arith and vector operations are reused to simplify
    this dialect.

    This dialect may be lowered to some experiemental intrinsics, or target-
    specific intrinsics (e.g. RISC-V Vector Extension).
  }];

  let dependentDialects = ["vector::VectorDialect"];
}

// Base class of VP dialect operations.
class VP_Op<string mnemonic, list<Trait> traits = []> :
    Op<VP_Dialect, mnemonic, traits> {}

#endif // VP_IR_VPDIALECT
