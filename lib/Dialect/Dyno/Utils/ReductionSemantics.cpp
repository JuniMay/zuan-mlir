//===- ReductionSemantics.cpp - Dyno reduction legality helpers ----------===//
//
// This file implements shared helpers for Dyno reduction identities,
// admissibility, and canonical reduced-dimension bookkeeping.
//
//===----------------------------------------------------------------------===//

#include "Dyno/Utils/ReductionSemantics.h"

#include "Dyno/IR/Dyno.h"
#include "Dyno/Utils/ReductionAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/StringSwitch.h"
#include <limits>

namespace mlir {
namespace dyno {

static bool isSignlessInteger(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.isSignless();
}

static bool isIntegerOrIndex(Type type) {
  return isa<IndexType>(type) || isSignlessInteger(type);
}

static bool isFloating(Type type) { return isa<FloatType>(type); }

static bool isIntegerReduction(CombiningKind kind, Type elementType) {
  if (!isIntegerOrIndex(elementType)) {
    return false;
  }
  switch (kind) {
  case CombiningKind::ADD:
  case CombiningKind::MUL:
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
  case CombiningKind::MAXUI:
  case CombiningKind::MINUI:
  case CombiningKind::MAXSI:
  case CombiningKind::MINSI:
    return true;
  case CombiningKind::MINNUMF:
  case CombiningKind::MAXNUMF:
  case CombiningKind::MINIMUMF:
  case CombiningKind::MAXIMUMF:
    return false;
  }

  llvm_unreachable("unexpected combining kind");
}

static bool isFloatingReduction(CombiningKind kind, Type elementType) {
  if (!isFloating(elementType)) {
    return false;
  }
  switch (kind) {
  case CombiningKind::ADD:
  case CombiningKind::MUL:
  case CombiningKind::MINNUMF:
  case CombiningKind::MAXNUMF:
  case CombiningKind::MINIMUMF:
  case CombiningKind::MAXIMUMF:
    return true;
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
  case CombiningKind::MAXUI:
  case CombiningKind::MINUI:
  case CombiningKind::MAXSI:
  case CombiningKind::MINSI:
    return false;
  }

  llvm_unreachable("unexpected combining kind");
}

// These floating combiners change meaning when parenthesization changes, so
// strict-policy legality must reject factorization and parallel strip-mining.
static bool isFloatingReassociationSensitive(CombiningKind kind) {
  switch (kind) {
  case CombiningKind::ADD:
  case CombiningKind::MUL:
    return true;
  case CombiningKind::MINNUMF:
  case CombiningKind::MAXNUMF:
  case CombiningKind::MINIMUMF:
  case CombiningKind::MAXIMUMF:
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
  case CombiningKind::MAXUI:
  case CombiningKind::MINUI:
  case CombiningKind::MAXSI:
  case CombiningKind::MINSI:
    return false;
  }

  llvm_unreachable("unexpected combining kind");
}

std::optional<FloatingPointPolicy> parseFloatingPointPolicy(StringRef policy) {
  return llvm::StringSwitch<std::optional<FloatingPointPolicy>>(policy)
      .Case("strict", FloatingPointPolicy::Strict)
      .Case("relaxed", FloatingPointPolicy::Relaxed)
      .Default(std::nullopt);
}

StringRef stringifyFloatingPointPolicy(FloatingPointPolicy policy) {
  switch (policy) {
  case FloatingPointPolicy::Strict:
    return "strict";
  case FloatingPointPolicy::Relaxed:
    return "relaxed";
  }

  llvm_unreachable("unexpected floating-point policy");
}

std::optional<FloatingPointPolicy>
getReductionFloatingPointPolicy(ReductionOp op) {
  auto policyAttr =
      dyn_cast_or_null<StringAttr>(op->getDiscardableAttr(kDynoFpPolicyAttr));
  if (!policyAttr) {
    return std::nullopt;
  }
  return parseFloatingPointPolicy(policyAttr.getValue());
}

void setReductionFloatingPointPolicy(ReductionOp op,
                                     FloatingPointPolicy policy) {
  op->setDiscardableAttr(kDynoFpPolicyAttr,
                         StringAttr::get(op.getContext(),
                                         stringifyFloatingPointPolicy(policy)));
}

void copyReductionFloatingPointPolicy(ReductionOp from, ReductionOp to) {
  if (auto policy = getReductionFloatingPointPolicy(from)) {
    setReductionFloatingPointPolicy(to, *policy);
  }
}

bool isReductionTypeSupported(CombiningKind kind, Type elementType) {
  switch (kind) {
  case CombiningKind::ADD:
  case CombiningKind::MUL:
    return isIntegerOrIndex(elementType) || isFloating(elementType);
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
  case CombiningKind::MAXUI:
  case CombiningKind::MINUI:
  case CombiningKind::MAXSI:
  case CombiningKind::MINSI:
    return isSignlessInteger(elementType);
  case CombiningKind::MINNUMF:
  case CombiningKind::MAXNUMF:
  case CombiningKind::MINIMUMF:
  case CombiningKind::MAXIMUMF:
    return isFloating(elementType);
  }

  llvm_unreachable("unexpected combining kind");
}

bool hasImplicitIdentity(CombiningKind kind, Type elementType) {
  return isReductionTypeSupported(kind, elementType);
}

FailureOr<Value> buildReductionIdentity(OpBuilder &builder, Location loc,
                                        CombiningKind kind, Type elementType) {
  if (!isReductionTypeSupported(kind, elementType)) {
    return failure();
  }

  if (auto indexType = dyn_cast<IndexType>(elementType)) {
    switch (kind) {
    case CombiningKind::ADD:
      return Value(arith::ConstantIndexOp::create(builder, loc, 0));
    case CombiningKind::MUL:
      return Value(arith::ConstantIndexOp::create(builder, loc, 1));
    default:
      return failure();
    }
  }

  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    unsigned width = integerType.getWidth();
    APInt value(width, 0, /*isSigned=*/false);

    switch (kind) {
    case CombiningKind::ADD:
    case CombiningKind::OR:
    case CombiningKind::XOR:
    case CombiningKind::MAXUI:
      value = APInt(width, 0, /*isSigned=*/false);
      break;
    case CombiningKind::MUL:
      value = APInt(width, 1, /*isSigned=*/false);
      break;
    case CombiningKind::AND:
    case CombiningKind::MINUI:
      value = APInt::getAllOnes(width);
      break;
    case CombiningKind::MAXSI:
      value = APInt::getSignedMinValue(width);
      break;
    case CombiningKind::MINSI:
      value = APInt::getSignedMaxValue(width);
      break;
    case CombiningKind::MINNUMF:
    case CombiningKind::MAXNUMF:
    case CombiningKind::MINIMUMF:
    case CombiningKind::MAXIMUMF:
      return failure();
    }

    auto attr = builder.getIntegerAttr(integerType, value);
    return Value(arith::ConstantOp::create(builder, loc, integerType, attr));
  }

  auto floatType = cast<FloatType>(elementType);
  switch (kind) {
  case CombiningKind::ADD:
    return Value(arith::ConstantOp::create(
        builder, loc, floatType, builder.getFloatAttr(floatType, 0.0)));
  case CombiningKind::MUL:
    return Value(arith::ConstantOp::create(
        builder, loc, floatType, builder.getFloatAttr(floatType, 1.0)));
  case CombiningKind::MINNUMF:
  case CombiningKind::MINIMUMF:
    return Value(arith::ConstantOp::create(
        builder, loc, floatType,
        builder.getFloatAttr(floatType,
                             std::numeric_limits<double>::infinity())));
  case CombiningKind::MAXNUMF:
  case CombiningKind::MAXIMUMF:
    return Value(arith::ConstantOp::create(
        builder, loc, floatType,
        builder.getFloatAttr(floatType,
                             -std::numeric_limits<double>::infinity())));
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
  case CombiningKind::MAXUI:
  case CombiningKind::MINUI:
  case CombiningKind::MAXSI:
  case CombiningKind::MINSI:
    return failure();
  }
}

bool isFactorizationAdmissible(CombiningKind kind, Type elementType,
                               FloatingPointPolicy policy) {
  // Factorization only changes the parenthesization of the canonical ordered
  // sequence. Integer-like combiners we support are exact under that rewrite,
  // while reassociation-sensitive floating combiners require the relaxed
  // policy.
  if (!isReductionTypeSupported(kind, elementType)) {
    return false;
  }
  if (isIntegerReduction(kind, elementType)) {
    return true;
  }
  if (!isFloatingReduction(kind, elementType)) {
    // Not a supporting floating point case.
    return false;
  }
  if (!isFloatingReassociationSensitive(kind)) {
    return true;
  }
  return policy == FloatingPointPolicy::Relaxed;
}

bool isParallelStripmineAdmissible(CombiningKind kind, Type elementType,
                                   FloatingPointPolicy policy) {
  // Parallel strip-mining is stronger than factorization: it needs an identity
  // for inactive lanes and also permits lane-grouped regrouping before the
  // final register reduction.
  if (!hasImplicitIdentity(kind, elementType)) {
    return false;
  }
  if (!isFactorizationAdmissible(kind, elementType, policy)) {
    return false;
  }
  if (isFloatingReduction(kind, elementType) &&
      isFloatingReassociationSensitive(kind)) {
    return policy == FloatingPointPolicy::Relaxed;
  }
  return true;
}

ReductionModeKind chooseReductionMode(CombiningKind kind, Type elementType,
                                      FloatingPointPolicy policy,
                                      ReductionModeKind requested) {
  if (requested != ReductionModeKind::Auto) {
    return requested;
  }
  if (isParallelStripmineAdmissible(kind, elementType, policy)) {
    return ReductionModeKind::Parallel;
  }
  return ReductionModeKind::Sequential;
}

LogicalResult verifyCanonicalReductionDims(Operation *op, int64_t sourceRank,
                                           ArrayRef<int64_t> dims) {
  llvm::SmallDenseSet<int64_t> seen;
  int64_t previousDim = -1;
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= sourceRank) {
      return op->emitOpError() << "expected reduction dims to be in range [0, "
                               << sourceRank << ")";
    }
    if (!seen.insert(dim).second) {
      return op->emitOpError("expected reduction dims to be unique");
    }
    if (dim <= previousDim) {
      // TODO: Verify the semantic of dim order in formalism.
      return op->emitOpError("expected reduction dims to be in canonical "
                             "ascending order");
    }
    previousDim = dim;
  }
  return success();
}

SmallVector<int64_t> getPreservedReductionDims(int64_t sourceRank,
                                               ArrayRef<int64_t> reducedDims) {
  llvm::SmallDenseSet<int64_t> reducedSet(reducedDims.begin(),
                                          reducedDims.end());
  SmallVector<int64_t> preservedDims;
  preservedDims.reserve(sourceRank - reducedDims.size());
  for (int64_t dim = 0; dim < sourceRank; ++dim) {
    if (!reducedSet.contains(dim)) {
      preservedDims.push_back(dim);
    }
  }
  return preservedDims;
}

std::optional<unsigned> mapSourceDimToResultDim(unsigned sourceDim,
                                                ArrayRef<int64_t> reducedDims) {
  if (llvm::is_contained(reducedDims, static_cast<int64_t>(sourceDim))) {
    return std::nullopt;
  }

  unsigned resultDim = 0;
  for (int64_t dim = 0; dim < static_cast<int64_t>(sourceDim); ++dim) {
    if (!llvm::is_contained(reducedDims, dim)) {
      ++resultDim;
    }
  }
  return resultDim;
}

unsigned mapSourceDimToCurrentReductionDim(
    unsigned sourceDim, ArrayRef<int64_t> previouslyReducedSourceDims) {
  unsigned shiftedDim = sourceDim;
  for (int64_t reducedDim : previouslyReducedSourceDims) {
    if (reducedDim < static_cast<int64_t>(sourceDim)) {
      --shiftedDim;
    }
  }
  return shiftedDim;
}

} // namespace dyno
} // namespace mlir
