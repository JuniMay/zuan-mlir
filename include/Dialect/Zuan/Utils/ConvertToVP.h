#ifndef DIALECT_ZUAN_UTILS_CONVERTTOVP_H
#define DIALECT_ZUAN_UTILS_CONVERTTOVP_H

#include "Zuan/IR/Zuan.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace zuan {

struct VPConversionState {
  unsigned vf;
  bool scalable;
  /// The value map for non-tile values.
  IRMapping valueMap;
  /// The value map for tile values. One tile may map to multiple vector values.
  DenseMap<Value, SmallVector<Value>> tileMap;

  VPConversionState() = default;

  std::optional<std::pair<Value, Value>> getMasks() const { return maskPair; }

  void setMasks(Value mask, Value maskedoff) {
    maskPair = std::make_pair(mask, maskedoff);
  }

  std::optional<std::pair<Value, Value>> resetMasks() {
    auto pair = maskPair;
    maskPair = std::nullopt;
    return pair;
  }

  void initialize(DynamicOp op);

private:
  /// The current mask and maskedoff values.
  std::optional<std::pair<Value, Value>> maskPair;
};

Value createCastOp(OpBuilder &b, Location loc, CastKind kind,
  Type outType, Value source);

void convertToVP(RewriterBase &rewriter, Operation* op, ShapeInfo &shapeInfo,
                 VPConversionState &state);

} // namespace zuan
} // namespace mlir

#endif // DIALECT_ZUAN_UTILS_CONVERTTOVP_H
