//===- Unrolling.h - Zuan Unrolling Utilities -------------------*- C++ -*-===//
//
// This file declares the unrolling utilities for Zuan operations.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_UTILS_UNROLLING_H
#define ZUAN_UTILS_UNROLLING_H

#include "Zuan/Utils/ShapeInference.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace zuan {

class TileType;
class DynamicOp;
enum class CombiningKind : uint32_t;

/// Options and parameters that controls the unrolling process.
struct UnrollOptions {
  UnrollOptions(const UnrollOptions &) = default;
  UnrollOptions &operator=(const UnrollOptions &) = default;

  UnrollOptions(OpFoldResult offset, OpFoldResult chunkSize, unsigned idx,
                bool reduceUnitDim)
      : offset(offset), chunkSize(chunkSize), unrollIdx(idx),
        reduceUnitDim(reduceUnitDim) {}

  void overrideReduceUnitDim(bool reduce) { reduceUnitDim = reduce; }
  void overrideUnrollIdx(unsigned newDim) { unrollIdx = newDim; }

  /// If this dimension-to-unroll should be reduced.
  bool shouldReduce() const {
    if (auto attr = chunkSize.dyn_cast<Attribute>()) {
      auto chunkSizeInt = cast<IntegerAttr>(attr).getInt();
      return chunkSizeInt == 1 && reduceUnitDim;
    }
    return false;
  }

  OpFoldResult getOffset() const { return offset; }
  OpFoldResult getChunkSize() const { return chunkSize; }
  unsigned getUnrollIdx() const { return unrollIdx; }

  /// Sentinel dim value for not unrolling
  static constexpr unsigned kNoUnrollIdx = -1;

private:
  /// The offset of this iteration.
  OpFoldResult offset;
  /// The chunk size of this iteration.
  OpFoldResult chunkSize;
  /// The dimension to unroll.
  unsigned unrollIdx;
  /// If reduce the unit dimension (as specified by chunkSize) after unrolling.
  bool reduceUnitDim;
};

struct UnrollState {
  UnrollState() = default;
  /// The valueMap stores the values defined above the operation, or operations
  /// that does not need to be unrolled and is cloned as is. The unrolling
  /// process traverse the def-use from yielded scalars or memory write to their
  /// dependencies. Operations-to-unroll are handled each time it is needed and
  /// are not stored in the valueMap.
  IRMapping valueMap;
  /// The final yield block for store operations to insert.
  Block *yieldBlock;
};

/// Check if a memref value is defined inside a dynamic op.
bool isMemrefDefinedInsideDynamicOp(Value value);

/// Entry function to unroll zuan operations.
Operation *unrollOp(OpBuilder &builder, Operation *op, UnrollOptions options,
                    UnrollState &state);

/// Unrolling helper for operand.
Value getUnrolledValue(OpBuilder &builder, Value operand, UnrollOptions options,
                       UnrollState &state);

Value getUnrolledMemref(OpBuilder &builder, Value memref, UnrollOptions options,
                        UnrollState &state);

/// Slice the memref by the configurations specified in the options.
Value createMemrefSlice(OpBuilder &rewriter, Value memref,
                        UnrollOptions options);

SmallVector<int64_t> getUnrolledShape(ArrayRef<int64_t> shape,
                                      UnrollOptions options);

TileType getUnrolledTileType(TileType tileType, UnrollOptions options);

/// Create a combining operation with the given kind.
Value createCombiningOp(OpBuilder &b, Location loc, zuan::CombiningKind kind,
                        Value lhs, Value rhs);

/// Split the dynamic op for unrolling. If the rank of store in yield op is
/// larger than target stores, and the dimsize does not equivalent to the
/// selected one, create a new dynamic op and move all the incompatible stores
/// to the new dynamic op.
void splitDynamicOpForUnrolling(RewriterBase &rewriter, DynamicOp dynamicOp,
                                unsigned unrollIdx, ShapeInfo &shapeInfo);

} // namespace zuan
} // namespace mlir

#endif // ZUAN_UTILS_UNROLLING_H
