//===- ReductionSemantics.h - Dyno reduction legality helpers --*- C++ -*-===//
//
// This file declares shared helpers for Dyno reduction identities, legality,
// and canonical reduced-dimension bookkeeping.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_UTILS_REDUCTIONSEMANTICS_H
#define DYNO_UTILS_REDUCTIONSEMANTICS_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace dyno {

enum class CombiningKind : uint32_t;
class ReductionOp;

enum class FloatingPointPolicy {
  Strict,
  Relaxed,
};

enum class ReductionModeKind {
  Auto,
  Sequential,
  Parallel,
};

/// Build the implicit scalar identity for a supported combiner/type pair.
FailureOr<Value> buildReductionIdentity(OpBuilder &builder, Location loc,
                                        CombiningKind kind, Type elementType);
/// Return whether the combiner admits an implicit identity for the element
/// type.
bool hasImplicitIdentity(CombiningKind kind, Type elementType);
/// Return whether the combiner is defined for the given scalar element type.
bool isReductionTypeSupported(CombiningKind kind, Type elementType);
/// Return whether repeated 1-D factorization preserves the formal semantics.
bool isFactorizationAdmissible(CombiningKind kind, Type elementType,
                               FloatingPointPolicy policy);
/// Return whether lane-wise chunk accumulation plus final lane reduction is
/// legal.
bool isParallelStripmineAdmissible(CombiningKind kind, Type elementType,
                                   FloatingPointPolicy policy);
/// Parse the string spelling of a reduction floating-point policy.
std::optional<FloatingPointPolicy> parseFloatingPointPolicy(StringRef policy);
/// Return the canonical string spelling of a reduction floating-point policy.
StringRef stringifyFloatingPointPolicy(FloatingPointPolicy policy);
/// Read the optional floating-point policy attached to a reduction.
std::optional<FloatingPointPolicy>
getReductionFloatingPointPolicy(ReductionOp op);
/// Attach the floating-point policy to a reduction as discardable IR state.
void setReductionFloatingPointPolicy(ReductionOp op,
                                     FloatingPointPolicy policy);
/// Preserve the floating-point policy when deriving a new reduction from one.
void copyReductionFloatingPointPolicy(ReductionOp from, ReductionOp to);
/// Resolve `auto` mode into the concrete 1-D reduction mode to lower.
ReductionModeKind chooseReductionMode(CombiningKind kind, Type elementType,
                                      FloatingPointPolicy policy,
                                      ReductionModeKind requested);

/// Verify that reduction dims are in-range, unique, and canonically sorted.
LogicalResult verifyCanonicalReductionDims(Operation *op, int64_t sourceRank,
                                           ArrayRef<int64_t> dims);
/// Return the source dims that survive after reducing `reducedDims`.
SmallVector<int64_t> getPreservedReductionDims(int64_t sourceRank,
                                               ArrayRef<int64_t> reducedDims);
/// Map a preserved source dim to its result-dim index after reduction.
std::optional<unsigned> mapSourceDimToResultDim(unsigned sourceDim,
                                                ArrayRef<int64_t> reducedDims);
/// Renumber one source dim after removing earlier reduced source dims.
unsigned mapSourceDimToCurrentReductionDim(
    unsigned sourceDim, ArrayRef<int64_t> previouslyReducedSourceDims);

} // namespace dyno
} // namespace mlir

#endif // DYNO_UTILS_REDUCTIONSEMANTICS_H
