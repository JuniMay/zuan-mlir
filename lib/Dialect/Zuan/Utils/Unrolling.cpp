//===- Unrolling.cpp - Zuan Unrolling Utilities -----------------*- C++ -*-===//
//
// This file implements the unrolling utilities for Zuan operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <cassert>

#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/Unrolling.h"

namespace mlir {
namespace zuan {

bool isMemrefDefinedInsideDynamicOp(Value value) {
  auto *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return false;
  }
  auto dynamicOp = definingOp->getParentOfType<DynamicOp>();
  return dynamicOp != nullptr;
}

Value getUnrolledMemref(OpBuilder &builder, Value memref, UnrollOptions options,
                        UnrollState &state) {
  if (isMemrefDefinedInsideDynamicOp(memref)) {
    /// Unroll the subview or other memref operations.
    return getUnrolledValue(builder, memref, options, state);
  } else {
    /// Slice the memref itself.
    return createMemrefSlice(builder, memref, options);
  }
}

Value createMemrefSlice(OpBuilder &rewriter, Value memref,
                        UnrollOptions options) {
  assert(!isMemrefDefinedInsideDynamicOp(memref) &&
         "creating slice on a memref defined inside dynamic op is not "
         "well-defined.");

  auto loc = memref.getLoc();

  auto memrefType = cast<MemRefType>(memref.getType());
  auto memrefShape = memrefType.getShape();

  auto rank = memrefShape.size();

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  for (size_t i = 0; i < rank; ++i) {
    if (i == options.getDim()) {
      // This memref is not defined inside the dynamic op, so it should be safe
      // to directly reuse the op fold results.
      offsets[i] = options.getOffset();
      sizes.push_back(options.getChunkSize());
    } else if (ShapedType::isDynamic(memrefShape[i])) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(memref);
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value dim = rewriter.create<memref::DimOp>(loc, memref, idx);
      sizes.push_back(dim);
    } else {
      sizes.push_back(rewriter.getIndexAttr(memrefShape[i]));
    }
  }

  // Get the unreduced memref type. The offset will be just the same after rank
  // reduction.
  Type unreducedMemrefType =
      memref::SubViewOp::inferResultType(memrefType, offsets, sizes, strides);
  auto [unreducedStrides, unreducedOffset] =
      cast<MemRefType>(unreducedMemrefType).getStridesAndOffset();
  auto unreducedSizes = cast<MemRefType>(unreducedMemrefType).getShape();

  SmallVector<int64_t> resultSizes;
  SmallVector<int64_t> resultStrides;

  // Unroll the requested dim.
  for (size_t i = 0; i < rank; ++i) {
    if (i == options.getDim() && options.shouldReduce()) {
      // Drop this dim.
    } else {
      resultSizes.push_back(unreducedSizes[i]);
      resultStrides.push_back(unreducedStrides[i]);
    }
  }
  /// Manually create the layout here because when multiple contiguous unit dim
  /// presents, the strides cannot be correctly inferred by reduced type
  /// inference in subview op.
  auto layout = StridedLayoutAttr::get(unreducedMemrefType.getContext(),
                                       unreducedOffset, resultStrides);
  auto resultType =
      MemRefType::get(resultSizes, memrefType.getElementType(), layout);
  Value subview = rewriter.create<memref::SubViewOp>(loc, resultType, memref,
                                                     offsets, sizes, strides);
  return subview;
}

Operation *unrollOp(OpBuilder &builder, Operation *op, UnrollOptions options,
                    UnrollState &state) {
  assert(op && "expected a non-null operation");
}

} // namespace zuan
} // namespace mlir