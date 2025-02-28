//===- ZuanTypes.cpp - Zuan Type System Implementation ----------*- C++ -*-===//
//
// This file implements the Zuan type system for the Zuan IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

#include "Zuan/IR/Zuan.h"
#include "mlir/Support/LLVM.h"

#define GET_TYPEDEF_CLASSES
#include "Zuan/IR/ZuanOpsTypes.cpp.inc"

namespace mlir {
namespace zuan {

void ZuanDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Zuan/IR/ZuanOpsTypes.cpp.inc"
      >();
}

Type TileType::parse(AsmParser &parser) {
  Type elementType;
  SmallVector<int64_t, 2> shape;
  if (parser.parseLess() || parser.parseDimensionList(shape, true, true) ||
      parser.parseType(elementType) || parser.parseGreater()) {
    return nullptr;
  }
  return TileType::get(shape, elementType);
}

void TileType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  if (!getShape().empty()) {
    printer << "x";
  }
  printer.printType(getElementType());
  printer << ">";
}

bool TileType::isDimCompatible(int64_t a, int64_t b) {
  if (ShapedType::isDynamic(a) || ShapedType::isDynamic(b)) {
    return true;
  }
  return a == b;
}

bool TileType::isShapeCompatible(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [lhsSize, rhsSize] : llvm::zip(lhs, rhs)) {
    if (!isDimCompatible(lhsSize, rhsSize)) {
      return false;
    }
  }
  return true;
}

bool TileType::isCompatible(Type a, Type b) {
  if (auto aTile = llvm::dyn_cast<TileType>(a)) {
    if (auto bTile = llvm::dyn_cast<TileType>(b)) {
      return aTile.getElementType() == bTile.getElementType() &&
             TileType::isShapeCompatible(aTile.getShape(), bTile.getShape());
    }
  }
  return false;
}

SmallVector<int64_t> TileType::selectStaticShape(ArrayRef<int64_t> lhs,
                                                 ArrayRef<int64_t> rhs) {
  SmallVector<int64_t> result;
  result.reserve(lhs.size());
  for (auto [lhsSize, rhsSize] : llvm::zip(lhs, rhs)) {
    if (ShapedType::isDynamic(lhsSize) && ShapedType::isDynamic(rhsSize)) {
      result.push_back(ShapedType::kDynamic);
    } else if (ShapedType::isDynamic(lhsSize)) {
      result.push_back(rhsSize);
    } else if (ShapedType::isDynamic(rhsSize)) {
      result.push_back(lhsSize);
    }
  }
  return result;
}

} // namespace zuan
} // namespace mlir
