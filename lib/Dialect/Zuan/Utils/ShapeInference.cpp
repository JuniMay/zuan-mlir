//===- ShapeInference.cpp - Shape Inference for Zuan ops --------*- C++ -*-===//
//
// This file implements the shape inference for Zuan operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include <variant>

#include "Zuan/IR/Zuan.h"
#include "Zuan/Interfaces/ZuanInferShapeInterface.h"
#include "Zuan/Utils/ShapeInference.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace zuan {

TileShape TileShape::dropDim(unsigned i) const {
  TileShape result;
  result.append(dims.begin(), dims.begin() + i);
  result.append(dims.begin() + i + 1, dims.end());
  return result;
}

TileShape TileShape::takeFront(unsigned n) const {
  return TileShape(ArrayRef<DimSize>(dims).take_front(n));
}

TileShape TileShape::takeBack(unsigned n) const {
  return TileShape(ArrayRef<DimSize>(dims).take_back(n));
}

SmallVector<OpFoldResult> TileShape::reify(OpBuilder &builder,
                                           Location loc) const {
  return llvm::map_to_vector(dims, [&](DimSize dim) {
    return dim.getOrCreateOpFoldResult(builder, loc);
  });
}

SmallVector<int64_t> TileShape::staticShape() const {
  return llvm::map_to_vector(dims, [&](DimSize dim) {
    return dim.getInteger().value_or(ShapedType::kDynamic);
  });
}

std::optional<std::pair<Value, unsigned>> DimSize::getAsMemrefDim(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto dimOp = dyn_cast_if_present<memref::DimOp>(definingOp)) {
    auto memref = dimOp.getSource();
    auto index = dimOp.getIndex();
    if (auto constantOp = index.getDefiningOp<arith::ConstantOp>()) {
      auto indexInteger = cast<IntegerAttr>(constantOp.getValue()).getInt();
      return std::make_pair(memref, indexInteger);
    }
  }
  return std::nullopt;
}

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
        if (auto pair = getAsMemrefDim(value)) {
          dimsize = *pair;
        } else {
          dimsize = value;
        }
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
    if (auto pair = getAsMemrefDim(value)) {
      dimsize = *pair;
    } else {
      dimsize = value;
    }
  } else {
    auto attr = cast<IntegerAttr>(ofr.dyn_cast<Attribute>());
    dimsize = attr.getInt();
  }
}

bool DimSize::operator<(const DimSize &rhs) const {
  if (dimsize.index() != rhs.dimsize.index()) {
    return dimsize.index() < rhs.dimsize.index();
  } else if (auto lhsSpecial = std::get_if<SpecialKey>(&dimsize)) {
    return static_cast<unsigned>(*lhsSpecial) <
           static_cast<unsigned>(*std::get_if<SpecialKey>(&rhs.dimsize));
  } else if (auto lhsDim = std::get_if<int64_t>(&dimsize)) {
    return *lhsDim < *std::get_if<int64_t>(&rhs.dimsize);
  } else if (auto lhsValue = std::get_if<Value>(&dimsize)) {
    return (*lhsValue).getAsOpaquePointer() <
           (*std::get_if<Value>(&rhs.dimsize)).getAsOpaquePointer();
  } else {
    auto [lhsMemref, lhsDim] = std::get<std::pair<Value, unsigned>>(dimsize);
    auto [rhsMemref, rhsDim] =
        std::get<std::pair<Value, unsigned>>(rhs.dimsize);
    if (lhsMemref != rhsMemref) {
      return lhsMemref.getAsOpaquePointer() < rhsMemref.getAsOpaquePointer();
    }
    return lhsDim < rhsDim;
  }
}

void DimSize::dump(llvm::raw_ostream &os) const {
  if (auto special = std::get_if<SpecialKey>(&dimsize)) {
    os << (*special == SpecialKey::Empty ? "<empty>" : "<tombstone>");
    return;
  }
  if (auto value = std::get_if<Value>(&dimsize)) {
    os << *value;
    return;
  }
  if (auto constant = std::get_if<int64_t>(&dimsize)) {
    os << *constant;
    return;
  }
  auto [memref, dim] = std::get<std::pair<Value, unsigned>>(dimsize);
  os << "memref(" << memref << ", " << dim << ")";
}

Value DimSize::getOrCreateValue(OpBuilder &builder, Location loc) const {
  if (std::holds_alternative<SpecialKey>(dimsize)) {
    llvm_unreachable("dense map sentinel does not have a value");
  } else if (auto val = std::get_if<Value>(&dimsize)) {
    return *val;
  } else if (auto val = std::get_if<int64_t>(&dimsize)) {
    return arith::ConstantIndexOp::create(builder, loc, *val);
  } else {
    auto [memref, dim] = std::get<std::pair<Value, unsigned>>(dimsize);
    return memref::DimOp::create(builder, loc, memref, dim);
  }
}

OpFoldResult DimSize::getOrCreateOpFoldResult(OpBuilder &builder,
                                              Location loc) const {
  if (std::holds_alternative<SpecialKey>(dimsize)) {
    llvm_unreachable("dense map sentinel does not have an OpFoldResult");
  } else if (auto val = std::get_if<Value>(&dimsize)) {
    return *val;
  } else if (auto val = std::get_if<int64_t>(&dimsize)) {
    return builder.getIndexAttr(*val);
  } else {
    auto [memref, dim] = std::get<std::pair<Value, unsigned>>(dimsize);
    Value dimVal = memref::DimOp::create(builder, loc, memref, dim);
    return dimVal;
  }
}

std::optional<Value> DimSize::getValue() const {
  if (std::holds_alternative<SpecialKey>(dimsize)) {
    return std::nullopt;
  }
  if (auto value = std::get_if<Value>(&dimsize)) {
    return *value;
  }
  return std::nullopt;
}

std::optional<int64_t> DimSize::getInteger() const {
  if (std::holds_alternative<SpecialKey>(dimsize)) {
    return std::nullopt;
  }
  if (auto integer = std::get_if<int64_t>(&dimsize)) {
    return *integer;
  }
  return std::nullopt;
}

DimSize DimSize::getDenseMapEmptyKey() { return DimSize(SpecialKey::Empty); }

DimSize DimSize::getDenseMapTombstoneKey() {
  return DimSize(SpecialKey::Tombstone);
}

unsigned DimSize::getHashValue() const {
  if (auto special = std::get_if<SpecialKey>(&dimsize)) {
    return llvm::DenseMapInfo<unsigned>::getHashValue(
        static_cast<unsigned>(*special));
  }
  if (auto integer = std::get_if<int64_t>(&dimsize)) {
    return llvm::DenseMapInfo<int64_t>::getHashValue(*integer);
  }
  if (auto value = std::get_if<Value>(&dimsize)) {
    return llvm::DenseMapInfo<Value>::getHashValue(*value);
  }
  auto [memref, dim] = std::get<std::pair<Value, unsigned>>(dimsize);
  return static_cast<unsigned>(llvm::hash_combine(memref.getAsOpaquePointer(),
                                                  dim));
}

const TileShape *ShapeInfo::lookupShape(Value value) const {
  auto it = shapes.find(value);
  if (it == shapes.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::optional<TileShape> ShapeInfo::getShapeWithEquivalence(Value value) {
  if (auto *shape = lookupShape(value)) {
    ShapeVector result =
        llvm::map_to_vector(shape->asArrayRef(), [&](const DimSize &dim) {
      return dimEquivalences.getOrInsertLeaderValue(dim);
    });
    return TileShape(std::move(result));
  }
  return std::nullopt;
}

void ShapeInfo::markDimEquivalent(DimSize lhs, DimSize rhs) {
  dimEquivalences.unionSets(lhs, rhs);
}

void ShapeInfo::markShapeEquivalent(ArrayRef<DimSize> lhs,
                                    ArrayRef<DimSize> rhs) {
  assert(lhs.size() == rhs.size() && "expected same rank");
  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    markDimEquivalent(lhsDim, rhsDim);
  }
}

void ShapeInfo::markShapeEquivalent(Value lhs, Value rhs) {
  auto *lhsShape = lookupShape(lhs);
  auto *rhsShape = lookupShape(rhs);
  assert(lhsShape || rhsShape && "expected shapes to be present");
  if (!lhsShape) {
    setShape(lhs, TileShape(rhsShape->asArrayRef()));
    return;
  }
  if (!rhsShape) {
    setShape(rhs, TileShape(lhsShape->asArrayRef()));
    return;
  }
  markShapeEquivalent(lhsShape->asArrayRef(), rhsShape->asArrayRef());
}

void ShapeInfo::markShapeEquivalent(ValueRange lhs, ValueRange rhs) {
  assert(lhs.size() == rhs.size() && "expected same number of values");
  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    markShapeEquivalent(lhsDim, rhsDim);
  }
}

void ShapeInfo::markShapeEquivalent(Value val, ArrayRef<DimSize> shape) {
  auto *valShape = lookupShape(val);
  if (valShape) {
    markShapeEquivalent(valShape->asArrayRef(), shape);
  } else {
    setShape(val, TileShape(shape));
  }
}

bool ShapeInfo::setShapeIfAbsent(Value value, TileShape shape) {
  if (lookupShape(value)) {
    return false;
  }
  setShape(value, std::move(shape));
  return true;
}

void ShapeInfo::setShape(Value value, TileShape shape) {
  shapes.try_emplace(value, std::make_unique<TileShape>(std::move(shape)));
}

void ShapeInfo::dump(llvm::raw_ostream &os) {
  os << "ShapeInfo:\n";
  for (auto &[value, _] : shapes) {
    auto shape = *getShapeWithEquivalence(value);
    os << "  " << value << " -> \n[\n";
    llvm::interleaveComma(shape.asArrayRef(), os, [&](DimSize dim) {
      dim.dump(os);
    });
    os << "\n]\n";
  }
}

void ShapeInfo::inferShape(Operation *rootOp, ShapeInferenceState &state) {
  if (auto shapedOp = dyn_cast<ZuanInferShapeInterface>(rootOp)) {
    shapedOp.inferShape(*this, state);
    return;
  }

  // This should cover most operations in `arith` and `math` dialects.
  if (rootOp->hasTrait<OpTrait::Elementwise>() &&
      rootOp->hasTrait<OpTrait::SameOperandsAndResultType>() &&
      rootOp->hasTrait<OpTrait::OneResult>() &&
      rootOp->hasTrait<OpTrait::ZeroRegions>()) {
    if (!isa<TileType>(rootOp->getResult(0).getType())) {
      return;
    }

    /// Get the specified operand index that is shape-equivalent to the result.
    if (auto idx =
            rootOp->getAttrOfType<IntegerAttr>("zuan_passthru_operand")) {
      auto opd = rootOp->getOperand(idx.getInt());
      this->markShapeEquivalent(rootOp->getResult(0), opd);
      // This will be translated to tail-undisturbed manner in VP dialect.
    } else {
      auto opd = rootOp->getOperand(0);
      this->markShapeEquivalent(rootOp->getResult(0), opd);
      for (auto opd : llvm::drop_begin(rootOp->getOperands(), 1)) {
        this->markShapeEquivalent(opd, rootOp->getOperand(0));
      }
    }

    if (auto maskPair = state.getMask()) {
      auto [mask, maskedoff] = *maskPair;
      this->markShapeEquivalent(rootOp->getResult(0), mask);
      if (maskedoff) {
        this->markShapeEquivalent(rootOp->getResult(0), maskedoff);
      }
    }

    return;
  }

  // Compare operations.
  if (isa<arith::CmpIOp, arith::CmpFOp>(rootOp)) {
    Value lhs = rootOp->getOperand(0);
    Value rhs = rootOp->getOperand(1);
    if (!isa<TileType>(lhs.getType())) {
      return;
    }
    this->markShapeEquivalent(rootOp->getResult(0), lhs);
    this->markShapeEquivalent(lhs, rhs);
    if (auto maskPair = state.getMask()) {
      auto [mask, maskedoff] = *maskPair;
      this->markShapeEquivalent(rootOp->getResult(0), mask);
      if (maskedoff) {
        this->markShapeEquivalent(rootOp->getResult(0), maskedoff);
      }
    }
    return;
  }

  // While op.
  if (auto whileOp = dyn_cast<scf::WhileOp>(rootOp)) {
    auto beforeBlock = whileOp.getBeforeBody();
    auto afterBlock = whileOp.getAfterBody();
    for (auto [init, args] :
         llvm::zip(whileOp.getInits(), beforeBlock->getArguments())) {
      if (isa<TileType>(init.getType())) {
        this->markShapeEquivalent(init, args);
      }
    }
    for (auto &op : beforeBlock->getOperations()) {
      inferShape(&op, state);
    }
    auto conditionOp = whileOp.getConditionOp();
    auto results = whileOp->getResults();
    for (auto [passed, arg, result] : llvm::zip(
             conditionOp.getArgs(), afterBlock->getArguments(), results)) {
      if (isa<TileType>(passed.getType())) {
        this->markShapeEquivalent(passed, arg);
        this->markShapeEquivalent(passed, result);
      }
    }
    for (auto &op : afterBlock->getOperations()) {
      inferShape(&op, state);
    }
  }

  // Genral loops. This is made for `scf.for`, not sure if applicable to others.
  if (auto iface = dyn_cast<LoopLikeOpInterface>(rootOp)) {
    // If this is a iterative accumulation operation.
    auto inits = iface.getInits();
    auto iterArgs = iface.getRegionIterArgs();
    for (auto [init, arg] : llvm::zip(inits, iterArgs)) {
      if (isa<TileType>(init.getType())) {
        this->markShapeEquivalent(init, arg);
      }
    }

    for (auto &region : iface.getLoopRegions()) {
      for (auto &op : region->getOps()) {
        inferShape(&op, state);
      }
    }

    if (auto results = iface.getLoopResults()) {
      auto yieldValues = iface.getYieldedValues();
      for (auto [result, yield] : llvm::zip(*results, yieldValues)) {
        if (isa<TileType>(result.getType())) {
          this->markShapeEquivalent(result, yield);
        }
      }
    }
  }
}

unsigned computeUnreducedIdx(unsigned idx, size_t unreducedRank,
                             function_ref<bool(unsigned)> isReduced) {
  /// The index in original shape that we're trying to match with the current
  /// result shape idx.
  unsigned currSourceIdx = 0;
  /// The index in result shape that is not matched yet.
  unsigned currResultIdx = 0;
  while (currSourceIdx < unreducedRank) {
    if (isReduced(currSourceIdx)) {
      // This dim is reduced, have not found the source dim for currResultIdx
      currSourceIdx += 1;
      continue;
    }
    // This dim is not reduced, found the source dim for currResultIdx.
    if (currResultIdx == idx) {
      // We found the source dim for the current result dim.
      break;
    }
    // Not found yet, trying to find for the next result dim.
    currSourceIdx += 1;
    currResultIdx += 1;
  }
  return currSourceIdx;
}

Value getOrCreateIndexValue(OpBuilder &builder, OpFoldResult ofr,
                            Location loc) {
  if (auto value = ofr.dyn_cast<Value>()) {
    return value;
  }
  auto integer = cast<IntegerAttr>(ofr.dyn_cast<Attribute>()).getInt();
  return arith::ConstantIndexOp::create(builder, loc, integer);
}

} // namespace zuan
} // namespace mlir
