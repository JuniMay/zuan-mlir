//===- Builders.cpp - Shared Dyno builder utilities ------------*- C++ -*-===//
//
// This file implements shared builder helpers used by Dyno lowering utilities.
//
//===----------------------------------------------------------------------===//

#include "Dyno/Utils/Builders.h"

#include "Dyno/IR/Dyno.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace mlir {
namespace dyno {

static Type getScalarType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.getElementType();
  }
  return type;
}

Value createCombiningOp(OpBuilder &builder, Location loc, CombiningKind kind,
                        Value lhs, Value rhs) {
  Type scalarType = getScalarType(lhs.getType());
  bool isIntegerLike = isa<IndexType>(scalarType) ||
                       (isa<IntegerType>(scalarType) &&
                        cast<IntegerType>(scalarType).isSignless());

  switch (kind) {
  case CombiningKind::ADD:
    return isIntegerLike ? Value(arith::AddIOp::create(builder, loc, lhs, rhs))
                         : Value(arith::AddFOp::create(builder, loc, lhs, rhs));
  case CombiningKind::MUL:
    return isIntegerLike ? Value(arith::MulIOp::create(builder, loc, lhs, rhs))
                         : Value(arith::MulFOp::create(builder, loc, lhs, rhs));
  case CombiningKind::MINIMUMF:
    return arith::MinimumFOp::create(builder, loc, lhs, rhs);
  case CombiningKind::MAXIMUMF:
    return arith::MaximumFOp::create(builder, loc, lhs, rhs);
  case CombiningKind::MAXNUMF:
    return arith::MaxNumFOp::create(builder, loc, lhs, rhs);
  case CombiningKind::MINNUMF:
    return arith::MinNumFOp::create(builder, loc, lhs, rhs);
  case CombiningKind::AND:
    return arith::AndIOp::create(builder, loc, lhs, rhs);
  case CombiningKind::OR:
    return arith::OrIOp::create(builder, loc, lhs, rhs);
  case CombiningKind::XOR:
    return arith::XOrIOp::create(builder, loc, lhs, rhs);
  case CombiningKind::MAXUI:
    return arith::MaxUIOp::create(builder, loc, lhs, rhs);
  case CombiningKind::MINUI:
    return arith::MinUIOp::create(builder, loc, lhs, rhs);
  case CombiningKind::MAXSI:
    return arith::MaxSIOp::create(builder, loc, lhs, rhs);
  case CombiningKind::MINSI:
    return arith::MinSIOp::create(builder, loc, lhs, rhs);
  }

  llvm_unreachable("unsupported combining kind");
}

} // namespace dyno
} // namespace mlir
