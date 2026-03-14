//===- DynoUnrollingInterface.td ---------------------------*- tablegen -*-===//
//
// This file defines the unrolling interface.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_INTERFACES_DYNOUNROLLINGINTERFACE_TD
#define DYNO_INTERFACES_DYNOUNROLLINGINTERFACE_TD

include "mlir/IR/OpBase.td"

def DynoUnrollingInterface : OpInterface<"DynoUnrollingInterface"> {
  let description = [{
    The `DynoUnrollingInterface` from dyno dialect defines an operation that
    can be unrolled into operations on lower-ranked tiles.
  }];
  let cppNamespace = "::mlir::dyno";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Unroll the operation by recursively unrolling its operands.",
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"unroll",
      /*args=*/(ins "::mlir::OpBuilder &":$builder,
                    "::mlir::dyno::UnrollOptions":$options,
                    "::mlir::dyno::UnrollState &":$state)>
  ];
}

#endif // DYNO_INTERFACES_DYNOUNROLLINGINTERFACE_TD
