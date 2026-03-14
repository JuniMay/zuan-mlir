#ifndef DIALECT_DYNO_UTILS_CONVERTTOVP_H
#define DIALECT_DYNO_UTILS_CONVERTTOVP_H

#include "Dyno/IR/Dyno.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace dyno {

struct VPShapePlan {
  enum class Kind {
    Scalar,
    Vector1D,
    RowPack2D,
    DynamicOuterLoopAndVector
  };

  Kind kind = Kind::Scalar;
  int64_t staticRows = 1;
  Value dynamicRows = nullptr;
  Value evl = nullptr;
  unsigned rank = 0;

  bool hasStaticRows() const { return dynamicRows == nullptr; }
  bool hasVector() const {
    return kind != Kind::Scalar;
  }
};

struct VPConversionState {
  unsigned vf;
  bool scalable;
  /// The value map for non-tile values.
  IRMapping valueMap;
  /// The value map for tile values. One tile may map to multiple vector values.
  DenseMap<Value, SmallVector<Value>> tileMap;

  VPConversionState() = default;

  std::optional<std::pair<Value, Value>> getMasks() const {
    if (maskStack.empty()) {
      return std::nullopt;
    }
    return maskStack.back();
  }

  void setMasks(Value mask, Value maskedoff) {
    maskStack.emplace_back(mask, maskedoff);
  }

  std::optional<std::pair<Value, Value>> resetMasks() {
    if (maskStack.empty()) {
      return std::nullopt;
    }
    return maskStack.pop_back_val();
  }

  void initialize(Operation *root);

private:
  SmallVector<std::pair<Value, Value>> maskStack;
};

class ScopedVPMaskState {
public:
  ScopedVPMaskState(VPConversionState &state, Value mask, Value maskedoff)
      : state(state), active(mask != nullptr) {
    if (active) {
      state.setMasks(mask, maskedoff);
    }
  }

  ~ScopedVPMaskState() {
    if (active) {
      state.resetMasks();
    }
  }

private:
  VPConversionState &state;
  bool active;
};

Value createCastOp(OpBuilder &b, Location loc, CastKind kind,
  Type outType, Value source);

LogicalResult convertToVP(OpBuilder &builder, Operation *op,
                          VPConversionState &state);

} // namespace dyno
} // namespace mlir

#endif // DIALECT_DYNO_UTILS_CONVERTTOVP_H
