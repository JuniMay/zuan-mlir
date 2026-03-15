//===- Builders.h - Shared Dyno builder utilities --------------*- C++ -*-===//
//
// This file declares shared builder helpers used by Dyno lowering utilities.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_UTILS_BUILDERS_H
#define DYNO_UTILS_BUILDERS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace dyno {

enum class CombiningKind : uint32_t;

Value createCombiningOp(OpBuilder &builder, Location loc, CombiningKind kind,
                        Value lhs, Value rhs);

} // namespace dyno
} // namespace mlir

#endif // DYNO_UTILS_BUILDERS_H
