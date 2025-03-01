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
                    "::mlir::zuan::UnrollState &":$state)>,
    InterfaceMethod<
      /*desc=*/[{
        Get the shape-to-unroll of the operation. If the operation does not have
        a shape semantics, e.g., it is a scalar operation, or it is the dynamic
        op, return `std::nullopt`.
      }],
      /*retTy=*/"::std::optional<::mlir::zuan::ShapeVector>",
      /*methodName=*/"getShapeToUnroll",
      /*args=*/(ins "::mlir::zuan::ShapeInfo &":$shapeInfo)>,
    InterfaceMethod<
      /*desc=*/[{
        Get the unrolled types of the operations before actually unrolling it.
        Typically result type, but could be different for yield operations.
      }],
      /*retTy=*/"::mlir::SmallVector<::mlir::Type>",
      /*methodName=*/"getUnrolledTypes",
      /*args=*/(ins "::mlir::zuan::UnrollOptions":$options)>
  ];
}

#endif // ZUAN_INTERFACES_ZUANUNROLLINGINTERFACE_TD
