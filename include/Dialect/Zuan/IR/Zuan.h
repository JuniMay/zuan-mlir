//===- Zuan.h - Zuan Dialect ------------------------------------*- C++ -*-===//
//
// This file defines the Zuan Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_ZUAN_IR_ZUAN_H
#define DIALECT_ZUAN_IR_ZUAN_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Zuan/Interfaces/ZuanInferShapeInterface.h"
#include "Zuan/Interfaces/ZuanUnrollingInterface.h"

#include "Zuan/IR/ZuanEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Zuan/IR/ZuanAttributes.h.inc"

#include "Zuan/IR/ZuanOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Zuan/IR/ZuanOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Zuan/IR/ZuanOps.h.inc"

#endif // DIALECT_ZUAN_IR_ZUAN_H
