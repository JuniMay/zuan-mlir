//===- ReductionAttrs.h - Dyno reduction lowering markers ------*- C++ -*-===//
//
// This file declares the internal attribute names shared by Dyno reduction
// strip-mining and VP conversion.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_UTILS_REDUCTIONATTRS_H
#define DYNO_UTILS_REDUCTIONATTRS_H

namespace mlir {
namespace dyno {

/// Prevents generated reductions from being normalized or strip-mined again.
inline constexpr const char *kDynoStripminedAttr = "dyno.stripmined";
/// Marks the final register reduction emitted by parallel strip-mining.
inline constexpr const char *kDynoParallelReductionAttr =
    "dyno.parallel_stripmine";
/// Marks the chunk-local 1-D reduction emitted by sequential strip-mining.
inline constexpr const char *kDynoSequentialReductionAttr =
    "dyno.sequential_stripmine";
/// Records that a parallel floating reduction may carry reassociation.
inline constexpr const char *kDynoParallelReassocAttr = "dyno.parallel_reassoc";

} // namespace dyno
} // namespace mlir

#endif // DYNO_UTILS_REDUCTIONATTRS_H
