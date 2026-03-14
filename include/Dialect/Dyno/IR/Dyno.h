//===- Dyno.h - Dyno Dialect ------------------------------------*- C++ -*-===//
//
// This file defines the Dyno Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_DYNO_IR_DYNO_H
#define DIALECT_DYNO_IR_DYNO_H

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

#include "Dyno/Interfaces/DynoUnrollingInterface.h"

#include "Dyno/IR/DynoEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Dyno/IR/DynoAttributes.h.inc"

#include "Dyno/IR/DynoOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dyno/IR/DynoOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dyno/IR/DynoOps.h.inc"

namespace mlir {
namespace dyno {

void createMaskOpRegion(OpBuilder &builder, Location loc, Operation *maskedOp);

/// Mask a given operation with a mas. If the mask is nullptr, return the
/// original operation. This function does not handle def-use chains, and the
/// caller is responsible for updating the uses of the original operation.
Operation *maskOperation(OpBuilder &builder, Location loc, Operation *maskedOp,
                         Value mask,
                         Value maskedoff = nullptr);

} // namespace dyno
} // namespace mlir

#endif // DIALECT_DYNO_IR_DYNO_H
