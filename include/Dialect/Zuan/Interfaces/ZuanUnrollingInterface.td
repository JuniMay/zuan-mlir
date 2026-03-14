//===- ZuanUnrollingInterface.td ---------------------------*- tablegen -*-===//
//
// This file defines the unrolling interface.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_INTERFACES_ZUANUNROLLINGINTERFACE_TD
#define ZUAN_INTERFACES_ZUANUNROLLINGINTERFACE_TD

include "mlir/IR/OpBase.td"

def ZuanUnrollingInterface : OpInterface<"ZuanUnrollingInterface"> {
  let description = [{
    The `ZuanUnrollingInterface` from zuan dialect defines an operation that
    can be unrolled into operations on lower-ranked tiles.
  }];
  let cppNamespace = "::mlir::zuan";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Unroll the operation by recursively unrolling its operands.",
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"unroll",
      /*args=*/(ins "::mlir::OpBuilder &":$builder,
                    "::mlir::zuan::UnrollOptions":$options,
                    "::mlir::zuan::UnrollState &":$state)>
  ];
}

#endif // ZUAN_INTERFACES_ZUANUNROLLINGINTERFACE_TD
