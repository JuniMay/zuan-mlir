//===- ShapeInference.h - Local shape reification for Zuan -----*- C++ -*-===//
//
// This file declares local, demand-driven shape reification helpers for Zuan.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_UTILS_SHAPEINFERENCE_H
#define ZUAN_UTILS_SHAPEINFERENCE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace zuan {

/// Reify the full shape of a Zuan tile value on demand.
///
/// The returned shape is expressed as mixed static/dynamic `OpFoldResult`s so
/// callers can keep shape reasoning local and only materialize SSA `index`
/// values when needed.
FailureOr<SmallVector<OpFoldResult>> reifyZuanShape(OpBuilder &builder,
                                                    Value value);

/// Reify one dimension of a Zuan tile value on demand.
///
/// This is the scalar counterpart to `reifyZuanShape`: it returns either a
/// constant index attribute for static dims or a dynamic `OpFoldResult` that
/// can later be materialized into SSA if required by the rewrite.
FailureOr<OpFoldResult> reifyZuanDim(OpBuilder &builder, Value value,
                                     unsigned dim);

/// Rewrite direct `zuan.dim` users of a tile result into plain index values.
///
/// This is used by destructive rewrites before erasing a tile-producing op:
/// while the producer still exists, resolve any direct shape-query users so the
/// producer can later be dropped without losing its dimension semantics.
LogicalResult resolveDimUsersOfResult(OpResult result,
                                      PatternRewriter &rewriter);

/// Apply `resolveDimUsersOfResult` to every result of an operation.
///
/// This is the common entry point for destructive slice cleanup when an entire
/// tile-producing op is about to be erased.
LogicalResult resolveDimUsersOfOp(Operation *op, PatternRewriter &rewriter);

std::optional<int64_t> getConstantZuanIntValue(OpFoldResult ofr);

unsigned computeUnreducedIdx(unsigned idx, size_t rank,
                             function_ref<bool(unsigned)> isReduced);

template <typename T>
T computeUnreducedDim(unsigned idx, ArrayRef<T> originalShape,
                      function_ref<bool(unsigned)> isReduced) {
  auto sourceIdx = computeUnreducedIdx(idx, originalShape.size(), isReduced);
  return originalShape[sourceIdx];
}

Value getOrCreateIndexValue(OpBuilder &builder, OpFoldResult ofr, Location loc);

} // namespace zuan
} // namespace mlir

#endif // ZUAN_UTILS_SHAPEINFERENCE_H
