//===- Slicing.h - Dyno structural slicing utilities -----------*- C++ -*-===//
//
// This file declares structural result-domain slicing helpers for Dyno.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_UTILS_SLICING_H
#define DYNO_UTILS_SLICING_H

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace dyno {

class TileType;

struct SliceSpec {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<bool> droppedDims;

  unsigned getSourceRank() const { return offsets.size(); }
  unsigned getResultRank() const;
  bool dropsDim(unsigned dim) const { return droppedDims[dim]; }

  static FailureOr<SliceSpec> getIdentity(OpBuilder &builder, Value value);
  static FailureOr<SliceSpec>
  getSingleDimSlice(OpBuilder &builder, Value value, unsigned dim,
                    OpFoldResult offset, OpFoldResult size, bool dropUnitDim);
  static FailureOr<SliceSpec> getPrefixSlice(OpBuilder &builder, Value value,
                                             ArrayRef<OpFoldResult> offsets,
                                             ArrayRef<OpFoldResult> sizes,
                                             ArrayRef<bool> droppedDims = {});

  SmallVector<int64_t> getSlicedShape(ArrayRef<int64_t> shape) const;
  TileType getSlicedTileType(TileType type) const;
};

struct SliceState {
  // SliceState caches SSA mappings while materializing one concrete slice.
  // Callers must create a fresh state for each distinct offsets/sizes tuple.
  IRMapping valueMap;

  void initialize(Operation *root);
};

FailureOr<Value> cloneOrReuseValue(OpBuilder &builder, Value value,
                                   SliceState &state);
FailureOr<Value> sliceValue(OpBuilder &builder, Value value,
                            const SliceSpec &spec, SliceState &state);
FailureOr<Value> sliceMemrefView(OpBuilder &builder, Value memref,
                                 const SliceSpec &spec, SliceState &state);
FailureOr<Operation *> sliceRootOperation(OpBuilder &builder, Operation *root,
                                          const SliceSpec &spec,
                                          SliceState &state);

} // namespace dyno
} // namespace mlir

#endif // DYNO_UTILS_SLICING_H
