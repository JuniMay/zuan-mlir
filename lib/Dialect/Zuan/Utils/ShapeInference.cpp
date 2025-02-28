//===- ShapeInference.cpp - Shape Inference for Zuan ops --------*- C++ -*-===//
//
// This file implements the shape inference for Zuan operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <variant>

#include "Zuan/Interfaces/ZuanInferShapeInterface.h"
#include "Zuan/Utils/ShapeInference.h"

namespace mlir {
namespace zuan {

DimSize::DimSize(Value memref, unsigned dim) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType) {
    llvm_unreachable("expected memref type");
  }
  if (dim >= memrefType.getRank()) {
    llvm_unreachable("invalid dimension");
  }
  auto shape = memrefType.getShape();
  if (!ShapedType::isDynamic(shape[dim])) {
    dimsize = shape[dim];
    return;
  }
  if (auto definingOp = memref.getDefiningOp()) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(definingOp)) {
      auto droppedDims = subviewOp.getDroppedDims();
      auto subviewSizes = subviewOp.getMixedSizes();
      auto unreducedDim = computeUnreducedDim<OpFoldResult>(
          dim, subviewSizes,
          [&](unsigned idx) { return droppedDims.test(idx); });
      if (auto value = unreducedDim.dyn_cast<Value>()) {
        dimsize = value;
      } else {
        auto attr = cast<IntegerAttr>(unreducedDim.dyn_cast<Attribute>());
        dimsize = attr.getInt();
      }
      return;
    }
  }
  // Fallback.
  dimsize = std::make_pair(memref, dim);
}

DimSize::DimSize(OpFoldResult ofr) {
  if (auto value = ofr.dyn_cast<Value>()) {
    dimsize = value;
  } else {
    auto attr = cast<IntegerAttr>(ofr.dyn_cast<Attribute>());
    dimsize = attr.getInt();
  }
}

bool DimSize::operator<(const DimSize &rhs) const {
  if (dimsize.index() != rhs.dimsize.index()) {
    return dimsize.index() < rhs.dimsize.index();
  } else if (auto lhsDim = std::get_if<int64_t>(&dimsize)) {
    return *lhsDim < *std::get_if<int64_t>(&rhs.dimsize);
  } else if (auto lhsValue = std::get_if<Value>(&dimsize)) {
    return (*lhsValue).getAsOpaquePointer() <
           (*std::get_if<Value>(&rhs.dimsize)).getAsOpaquePointer();
  } else {
    auto [lhsMemref, lhsDim] =
        *std::get_if<std::pair<Value, unsigned>>(&dimsize);
    auto [rhsMemref, rhsDim] =
        *std::get_if<std::pair<Value, unsigned>>(&rhs.dimsize);
    if (lhsMemref.getAsOpaquePointer() != rhsMemref.getAsOpaquePointer()) {
      return lhsMemref.getAsOpaquePointer() < rhsMemref.getAsOpaquePointer();
    }
    return lhsDim < rhsDim;
  }
}

void DimSize::dump(llvm::raw_ostream &os) const {
  if (auto value = std::get_if<Value>(&dimsize)) {
    os << *value;
    return;
  }
  if (auto constant = std::get_if<int64_t>(&dimsize)) {
    os << *constant;
    return;
  }
  auto [memref, dim] = *std::get_if<std::pair<Value, unsigned>>(&dimsize);
  os << "memref(" << memref << ", " << dim << ")";
}

Value DimSize::getOrCreateValue(OpBuilder &builder, Location loc) const {
  if (auto val = std::get_if<Value>(&dimsize)) {
    return *val;
  } else if (auto val = std::get_if<int64_t>(&dimsize)) {
    return builder.create<arith::ConstantIndexOp>(loc, *val);
  } else {
    auto [memref, dim] = std::get<std::pair<Value, unsigned>>(dimsize);
    return builder.create<memref::DimOp>(loc, memref, dim);
  }
}

OpFoldResult DimSize::getOrCreateOpFoldResult(OpBuilder &builder,
                                              Location loc) const {
  if (auto val = std::get_if<Value>(&dimsize)) {
    return *val;
  } else if (auto val = std::get_if<int64_t>(&dimsize)) {
    return builder.getIndexAttr(*val);
  } else {
    auto [memref, dim] = std::get<std::pair<Value, unsigned>>(dimsize);
    Value dimVal = builder.create<memref::DimOp>(loc, memref, dim);
    return dimVal;
  }
}

std::optional<Value> DimSize::getValue() const {
  if (auto value = std::get_if<Value>(&dimsize)) {
    return *value;
  }
  return std::nullopt;
}

std::optional<int64_t> DimSize::getInteger() const {
  if (auto integer = std::get_if<int64_t>(&dimsize)) {
    return *integer;
  }
  return std::nullopt;
}

std::optional<ArrayRef<DimSize>> ShapeInfo::getShape(Value value) const {
  if (shapes.contains(value)) {
    return shapes.lookup(value);
  }
  return std::nullopt;
}

std::optional<SmallVector<DimSize>>
ShapeInfo::getShapeWithEquivalence(Value value) {
  if (auto shape = getShape(value)) {
    SmallVector<DimSize> result = llvm::map_to_vector(*shape, [&](DimSize dim) {
      return dimEquivalences.getOrInsertLeaderValue(dim);
    });
    return result;
  }
  return std::nullopt;
}

void ShapeInfo::markEquivalent(DimSize lhs, DimSize rhs) {
  dimEquivalences.unionSets(lhs, rhs);
}

void ShapeInfo::markEquivalent(ArrayRef<DimSize> lhs, ArrayRef<DimSize> rhs) {
  assert(lhs.size() == rhs.size() && "expected same rank");
  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    markEquivalent(lhsDim, rhsDim);
  }
}

void ShapeInfo::markEquivalent(Value lhs, Value rhs) {
  auto lhsShape = getShape(lhs);
  auto rhsShape = getShape(rhs);
  assert(lhsShape || rhsShape && "expected shapes to be present");
  if (!lhsShape) {
    setShape(lhs, SmallVector<DimSize>{*rhsShape});
    return;
  }
  if (!rhsShape) {
    setShape(rhs, SmallVector<DimSize>{*lhsShape});
    return;
  }
  markEquivalent(*lhsShape, *rhsShape);
}

void ShapeInfo::markEquivalent(Value val, ArrayRef<DimSize> shape) {
  auto valShape = getShape(val);
  if (valShape) {
    markEquivalent(*valShape, shape);
  } else {
    setShape(val, SmallVector<DimSize>{shape});
  }
}

void ShapeInfo::setShape(Value value, SmallVector<DimSize> shape) {
  shapes.insert({value, shape});
}

void ShapeInfo::dump(llvm::raw_ostream &os) {
  os << "ShapeInfo:\n";
  for (auto [value, _] : shapes) {
    auto shape = *getShapeWithEquivalence(value);
    os << "  " << value << " -> [";
    llvm::interleaveComma(shape, os, [&](DimSize dim) { dim.dump(os); });
  }
}

void ShapeInfo::inferShape(Operation *rootOp, ShapeInferenceState &state) {
  if (auto shapedOp = dyn_cast<ZuanInferShapeInterface>(rootOp)) {
    shapedOp.inferShape(*this, state);
  }

  // TODO: Fallbacks
}

Value getOrCreateIndexValue(OpBuilder &builder, OpFoldResult ofr,
                            Location loc) {
  if (auto value = ofr.dyn_cast<Value>()) {
    return value;
  }
  auto integer = cast<IntegerAttr>(ofr.dyn_cast<Attribute>()).getInt();
  return builder.create<arith::ConstantIndexOp>(loc, integer);
}

} // namespace zuan
} // namespace mlir
