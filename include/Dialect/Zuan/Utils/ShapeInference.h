//===- ShapeInference.h - Shape inference for Zuan ops ----------*- C++ -*-===//
//
// This file declares the shape inference for Zuan operations.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_TRANSFORMS_SHAPEINFERENCE_H
#define ZUAN_TRANSFORMS_SHAPEINFERENCE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <variant>

namespace mlir {
namespace zuan {

/// Size of a dimension in the shape.
struct DimSize {
  DimSize(Value value) : dimsize(value) {}
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

private:
  /// The size can be a value, constant or corresponding to the size of a
  /// memref.
  ///
  /// The memref size is created lazily because we do not want the shape
  /// inference process to generate unnecessary memref.dim operations, which
  /// might lead to infinite loops in the greedy pattern rewriter.
  std::variant<
      // This dim is a runtime value.
      Value,
      // This dim is static.
      int64_t,
      // This dim corresponds to a dim in a memref.
      std::pair<Value, unsigned>>
      dimsize;
};

/// Stateful information for shape inference.
struct ShapeInferenceState {
  ShapeInferenceState() = default;

  /// Get the top mask in the stack.
  std::optional<Value> getMask() const {
    if (maskStack.empty()) {
      return std::nullopt;
    }
    return maskStack.back();
  }
  void pushMask(Value mask) { maskStack.push_back(mask); }
  std::optional<Value> popMask() {
    if (maskStack.empty()) {
      return std::nullopt;
    }
    Value mask = maskStack.pop_back_val();
    return mask;
  }

private:
  /// The stack of masks.
  SmallVector<Value> maskStack;
};

struct ShapeInfo {
  ShapeInfo() = default;

  /// Get the stored shape for a value.
  std::optional<ArrayRef<DimSize>> getShape(Value value) const;
  /// Get the stored shape, and map each dim into the leader of the equivalence
  /// class, and return the mapped shape.
  std::optional<SmallVector<DimSize>> getShapeWithEquivalence(Value value);

  /// Mark two dims as equivalent.
  void markEquivalent(DimSize lhs, DimSize rhs);
  void markEquivalent(ArrayRef<DimSize> lhs, ArrayRef<DimSize> rhs);
  /// Mark two values as shape-equivalent.
  ///
  /// If two shapes are already present, this will mark all corresponding dims
  /// into the equivalence class. Otherwise, propagate the shape from the
  /// present one to the absent one.
  void markEquivalent(Value lhs, Value rhs);
  /// Mark the shape of a value equivalent to another shape. If the value is
  /// already computed, just mark two shapes as equivalent. Otherwise set the
  /// shape.
  void markEquivalent(Value val, ArrayRef<DimSize> shape);

  /// Entry function for shape inference.
  void inferShape(Operation *rootOp, ShapeInferenceState &state);

  void dump(llvm::raw_ostream &os);

private:
  /// The value shapes.
  DenseMap<Value, SmallVector<DimSize>> shapes;
  /// The equivalence relationship of dim sizes.
  llvm::EquivalenceClasses<DimSize> dimEquivalences;

  /// Set shape for a value.
  ///
  /// Setter is kept as private because we do not want the shape information to
  /// be overwritten when traversing the IR. That is, no matter how the
  /// traversing order is, all shape information are kept in a non-decreasing
  /// manner.
  void setShape(Value value, SmallVector<DimSize> shape);
};

/// Helper function to handle dim reduction. Given the idx in the reduced shape,
/// the original shape, a callback to test if the idx is reduced, compute the
/// corresponding idx in the original shape.
template <typename T>
T computeUnreducedDim(unsigned idx, ArrayRef<T> originalShape,
                      function_ref<bool(unsigned)> isReduced) {
  /// The index in original shape that we're trying to match with the current
  /// result shape idx.
  unsigned currSourceIdx = 0;
  /// The index in result shape that is not matched yet.
  unsigned currResultIdx = 0;
  while (currSourceIdx < originalShape.size()) {
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
  return originalShape[currSourceIdx];
}

Value getOrCreateIndexValue(OpBuilder &builder, OpFoldResult ofr, Location loc);

} // namespace zuan
} // namespace mlir

#endif // ZUAN_TRANSFORMS_SHAPEINFERENCE_H