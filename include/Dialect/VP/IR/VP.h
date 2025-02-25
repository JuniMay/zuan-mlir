//===- VP.h - VP Dialect ----------------------------------------*- C++ -*-===//
//
// This file defines the VP Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VP_IR_VP_H
#define DIALECT_VP_IR_VP_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "VP/IR/VPOpsDialect.h.inc"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#define GET_OP_CLASSES
#include "VP/IR/VPOps.h.inc"

namespace mlir {
namespace vp {

/// Build the body of a predicate operation.
void createPredicateOpRegion(OpBuilder &builder, Operation *predicatedOp);

/// Predicate the given operation with EVL, mask and passthru, return the
/// new predicate operation.
Operation *predicateOperation(OpBuilder &builder, Operation *predicatedOp,
                              Value evl, Value mask = nullptr,
                              Value passthru = nullptr,
                              Value maskedoff = nullptr);

} // namespace vp
} // namespace mlir

#endif // DIALECT_VP_IR_VP_H
