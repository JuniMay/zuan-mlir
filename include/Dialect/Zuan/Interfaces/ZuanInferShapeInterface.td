//===- ZuanInferShapeInterface.td --------------------------*- tablegen -*-===//
//
// This file defines the shape inference interface for operations that have
// can have their operands and results shape inferred.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_INTERFACES_ZUANINFERSHAPEINTERFACE_TD
#define ZUAN_INTERFACES_ZUANINFERSHAPEINTERFACE_TD

include "mlir/IR/OpBase.td"

def ZuanInferShapeInterface : OpInterface<"ZuanInferShapeInterface"> {
  let description = [{
    The `ZuanInferShapeInterface` from zuan dialect defines an operation that
    holds the information of the shape of its operands and results. This
    includes the shape equivalence and the constraints that the shape of the
    operands and results must satisfy.
  }];
  let cppNamespace = "::mlir::zuan";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Inject the local information into the ShapeInfo struct.",
      /*retTy=*/"void",
      /*methodName=*/"inferShape",
      /*args=*/(ins "ShapeInfo &":$shapeInfo, "ShapeInferenceState &":$state)>
  ];
}

#endif // ZUAN_INTERFACES_ZUANINFERSHAPEINTERFACE_TD
