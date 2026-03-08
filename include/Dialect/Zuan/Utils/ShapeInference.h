//===- ShapeInference.h - Shape inference for Zuan ops ----------*- C++ -*-===//
//
// This file declares the shape inference for Zuan operations.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_UTILS_SHAPEINFERENCE_H
#define ZUAN_UTILS_SHAPEINFERENCE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include <memory>
#include <utility>
#include <variant>

namespace mlir {
namespace zuan {

/// Size of a dimension in the shape.
struct DimSize {
  DimSize(int64_t size) : dimsize(size) {}
  DimSize(Value memref, unsigned dim);
  DimSize(OpFoldResult ofr);

  bool operator<(const DimSize &rhs) const;
  bool operator==(const DimSize &rhs) const { return dimsize == rhs.dimsize; }
  bool operator!=(const DimSize &rhs) const { return !(*this == rhs); }

  void dump(llvm::raw_ostream &os) const;
  /// Get the dim size as a value, if it is not a value, create a new value.
  Value getOrCreateValue(OpBuilder &builder, Location loc) const;
  /// Get the dim size as an OpFoldResult. If it is a value, return the value.
  /// If it is a constant, return an attribute. If it is a memref, create a
  /// memref.dim operation and return the value.
  OpFoldResult getOrCreateOpFoldResult(OpBuilder &builder, Location loc) const;

  /// Get the value of the dimsize, if it is a value. Otherwise, return nullptr.
  std::optional<Value> getValue() const;

  /// Get the integer value of the dimsize, if it is an integer. Otherwise,
  std::optional<int64_t> getInteger() const;

  static DimSize getDenseMapEmptyKey();
  static DimSize getDenseMapTombstoneKey();
  unsigned getHashValue() const;


private:
  enum class SpecialKey {
    Empty,
    Tombstone
  };

  explicit DimSize(SpecialKey key) : dimsize(key) {}

  /// The size can be a value, constant or corresponding to the size of a
  /// memref.
  ///
  /// The memref size is created lazily because we do not want the shape
  /// inference process to generate unnecessary memref.dim operations, which
  /// might lead to infinite loops in the greedy pattern rewriter.
  std::variant<
      // DenseMap sentinels.
      SpecialKey,
      // This dim is a runtime value.
      Value,
      // This dim is static.
      int64_t,
      // This dim corresponds to a dim in a memref.
      std::pair<Value, unsigned>>
      dimsize;

  static std::optional<std::pair<Value, unsigned>> getAsMemrefDim(Value value);
};

using ShapeVector = SmallVector<DimSize>;

struct TileShape {
  TileShape() = default;
  TileShape(ArrayRef<DimSize> dims) : dims(dims.begin(), dims.end()) {}
  TileShape(ShapeVector dims) : dims(std::move(dims)) {}

  size_t size() const { return dims.size(); }
  size_t rank() const { return dims.size(); }
  bool empty() const { return dims.empty(); }
  DimSize dim(unsigned i) const { return dims[i]; }
  DimSize front() const { return dims.front(); }
  DimSize back() const { return dims.back(); }
  ArrayRef<DimSize> asArrayRef() const { return dims; }

  auto begin() const { return dims.begin(); }
  auto end() const { return dims.end(); }
  auto begin() { return dims.begin(); }
  auto end() { return dims.end(); }

  DimSize operator[](size_t i) const { return dims[i]; }
  DimSize &operator[](size_t i) { return dims[i]; }

  void push_back(DimSize dim) { dims.push_back(dim); }
  void append(ArrayRef<DimSize> newDims) {
    dims.append(newDims.begin(), newDims.end());
  }

  template <typename IteratorT>
  void append(IteratorT beginIt, IteratorT endIt) {
    dims.append(beginIt, endIt);
  }

  TileShape dropDim(unsigned i) const;
  TileShape takeFront(unsigned n) const;
  TileShape takeBack(unsigned n) const;

  SmallVector<OpFoldResult> reify(OpBuilder &builder, Location loc) const;
  SmallVector<int64_t> staticShape() const;

private:
  ShapeVector dims;
};

/// Stateful information for shape inference.
struct ShapeInferenceState {
  ShapeInferenceState() = default;

  /// Get the top mask in the stack.
  std::optional<std::pair<Value, Value>> getMask() const {
    if (maskStack.empty()) {
      return std::nullopt;
    }
    return maskStack.back();
  }
  void setMask(Value mask, Value maskedoff = nullptr) {
    maskStack.emplace_back(mask, maskedoff);
  }
  std::optional<std::pair<Value, Value>> resetMask() {
    if (maskStack.empty()) {
      return std::nullopt;
    }
    auto pair = maskStack.pop_back_val();
    return pair;
  }

private:
  SmallVector<std::pair<Value, Value>> maskStack;
};

class ScopedMaskState {
public:
  ScopedMaskState(ShapeInferenceState &state, Value mask, Value maskedoff)
      : state(state), active(mask != nullptr) {
    if (active) {
      state.setMask(mask, maskedoff);
    }
  }

  ~ScopedMaskState() {
    if (active) {
      state.resetMask();
    }
  }

private:
  ShapeInferenceState &state;
  bool active;
};

struct ShapeInfo {
  ShapeInfo() = default;

  /// Get the stored shape for a value.
  const TileShape *lookupShape(Value value) const;
  /// Get the stored shape, and map each dim into the leader of the equivalence
  /// class, and return the mapped shape.
  std::optional<TileShape> getShapeWithEquivalence(Value value);

  /// Mark two dims as equivalent.
  void markDimEquivalent(DimSize lhs, DimSize rhs);
  void markShapeEquivalent(ArrayRef<DimSize> lhs, ArrayRef<DimSize> rhs);
  /// Mark two values as shape-equivalent.
  ///
  /// If two shapes are already present, this will mark all corresponding dims
  /// into the equivalence class. Otherwise, propagate the shape from the
  /// present one to the absent one.
  void markShapeEquivalent(Value lhs, Value rhs);
  void markShapeEquivalent(ValueRange lhs, ValueRange rhs);
  /// Mark the shape of a value equivalent to another shape. If the value is
  /// already computed, just mark two shapes as equivalent. Otherwise set the
  /// shape.
  void markShapeEquivalent(Value val, ArrayRef<DimSize> shape);

  bool setShapeIfAbsent(Value value, TileShape shape);

  void markEquivalent(DimSize lhs, DimSize rhs) { markDimEquivalent(lhs, rhs); }
  void markEquivalent(ArrayRef<DimSize> lhs, ArrayRef<DimSize> rhs) {
    markShapeEquivalent(lhs, rhs);
  }
  void markEquivalent(Value lhs, Value rhs) { markShapeEquivalent(lhs, rhs); }
  void markEquivalent(ValueRange lhs, ValueRange rhs) {
    markShapeEquivalent(lhs, rhs);
  }
  void markEquivalent(Value val, ArrayRef<DimSize> shape) {
    markShapeEquivalent(val, shape);
  }
  void markEquivalent(Value val, const TileShape &shape) {
    markShapeEquivalent(val, shape.asArrayRef());
  }

  /// Entry function for shape inference.
  void inferShape(Operation *rootOp, ShapeInferenceState &state);

  void dump(llvm::raw_ostream &os);

private:
  /// The value shapes.
  DenseMap<Value, std::unique_ptr<TileShape>> shapes;
  /// The equivalence relationship of dim sizes.
  llvm::EquivalenceClasses<DimSize> dimEquivalences;

  /// Set shape for a value.
  ///
  /// Setter is kept as private because we do not want the shape information to
  /// be overwritten when traversing the IR. That is, no matter how the
  /// traversing order is, all shape information are kept in a non-decreasing
  /// manner.
  void setShape(Value value, TileShape shape);
};

unsigned computeUnreducedIdx(unsigned idx, size_t rank,
                             function_ref<bool(unsigned)> isReduced);

/// Helper function to handle dim reduction. Given the idx in the reduced shape,
/// the original shape, a callback to test if the idx is reduced, compute the
/// corresponding idx in the original shape.
template <typename T>
T computeUnreducedDim(unsigned idx, ArrayRef<T> originalShape,
                      function_ref<bool(unsigned)> isReduced) {
  auto sourceIdx = computeUnreducedIdx(idx, originalShape.size(), isReduced);
  return originalShape[sourceIdx];
}

Value getOrCreateIndexValue(OpBuilder &builder, OpFoldResult ofr, Location loc);

} // namespace zuan
} // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::zuan::DimSize> {
  static mlir::zuan::DimSize getEmptyKey() {
    return mlir::zuan::DimSize::getDenseMapEmptyKey();
  }

  static mlir::zuan::DimSize getTombstoneKey() {
    return mlir::zuan::DimSize::getDenseMapTombstoneKey();
  }

  static unsigned getHashValue(const mlir::zuan::DimSize &value) {
    return value.getHashValue();
  }

  static bool isEqual(const mlir::zuan::DimSize &lhs,
                      const mlir::zuan::DimSize &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // ZUAN_UTILS_SHAPEINFERENCE_H
