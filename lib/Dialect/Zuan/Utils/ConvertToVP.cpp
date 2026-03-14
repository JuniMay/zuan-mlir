#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

#include "VP/IR/VP.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/ConvertToVP.h"
#include "Zuan/Utils/ShapeInference.h"
#include "Zuan/Utils/Unrolling.h"
#include "mlir/Analysis/SliceAnalysis.h"

#define DEBUG_TYPE "zuan-to-vp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "] ")

namespace mlir {
namespace zuan {

static FailureOr<Value> cloneMappedValue(OpBuilder &builder, Value value,
                                         VPConversionState &state) {
  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return failure();
  }
  Operation *cloned = builder.clone(*result.getOwner(), state.valueMap);
  for (auto [oldResult, newResult] :
       llvm::zip(result.getOwner()->getResults(), cloned->getResults())) {
    if (!isa<TileType>(oldResult.getType())) {
      state.valueMap.map(oldResult, newResult);
    }
  }
  return cloned->getResult(result.getResultNumber());
}

static Operation *cloneAndMap(OpBuilder &builder, Operation *op,
                              VPConversionState &state) {
  Operation *cloned = builder.clone(*op, state.valueMap);
  for (auto [oldResult, newResult] :
       llvm::zip(op->getResults(), cloned->getResults())) {
    if (!isa<TileType>(oldResult.getType())) {
      state.valueMap.map(oldResult, newResult);
    }
  }
  return cloned;
}

static LogicalResult ensureTileMapped(OpBuilder &builder, Value value,
                                      VPConversionState &state) {
  if (state.tileMap.contains(value)) {
    return success();
  }
  auto *def = value.getDefiningOp();
  if (!def) {
    return failure();
  }
  return convertToVP(builder, def, state);
}

static FailureOr<Value> ensureScalarValue(OpBuilder &builder, Value value,
                                          VPConversionState &state) {
  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }
  if (!value.getDefiningOp()) {
    state.valueMap.map(value, value);
    return value;
  }
  if (failed(convertToVP(builder, value.getDefiningOp(), state))) {
    return failure();
  }
  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }
  return cloneMappedValue(builder, value, state);
}

static FailureOr<Value> materializeDimValue(OpBuilder &builder, Location loc,
                                            OpFoldResult dim,
                                            VPConversionState &state) {
  if (auto value = dim.dyn_cast<Value>()) {
    return cloneMappedValue(builder, value, state);
  }
  if (auto constant = getConstantZuanIntValue(dim)) {
    return arith::ConstantIndexOp::create(builder, loc, *constant).getResult();
  }
  return failure();
}

static FailureOr<VPShapePlan> planVPShape(OpBuilder &builder, Location loc,
                                          ArrayRef<OpFoldResult> shape,
                                          VPConversionState &state) {
  VPShapePlan plan;
  plan.rank = shape.size();
  if (shape.empty()) {
    plan.kind = VPShapePlan::Kind::Scalar;
    return plan;
  }
  if (shape.size() == 1) {
    auto evl = materializeDimValue(builder, loc, shape[0], state);
    if (failed(evl)) {
      return failure();
    }
    plan.kind = VPShapePlan::Kind::Vector1D;
    plan.evl = *evl;
    return plan;
  }
  if (shape.size() == 2) {
    if (auto rows = getConstantZuanIntValue(shape[0])) {
      // A static outer dimension is represented as a row-pack in the current
      // emitter. This is a legality classification, not a profitability model:
      // cost control is expected to happen earlier via tiling/unrolling.
      auto evl = materializeDimValue(builder, loc, shape[1], state);
      if (failed(evl)) {
        return failure();
      }
      plan.kind = VPShapePlan::Kind::RowPack2D;
      plan.staticRows = *rows;
      plan.evl = *evl;
      return plan;
    }
    auto dynamicRows = materializeDimValue(builder, loc, shape[0], state);
    auto evl = materializeDimValue(builder, loc, shape[1], state);
    if (failed(dynamicRows) || failed(evl)) {
      return failure();
    }
    plan.kind = VPShapePlan::Kind::DynamicOuterLoopAndVector;
    plan.dynamicRows = *dynamicRows;
    plan.evl = *evl;
    return plan;
  }
  return failure();
}

static FailureOr<unsigned> getStaticRowCount(const VPShapePlan &plan) {
  if (!plan.hasStaticRows()) {
    return failure();
  }
  return static_cast<unsigned>(plan.staticRows);
}

static Value lookupTileComponent(Value value, unsigned rowIdx,
                                 VPConversionState &state) {
  return state.tileMap.lookup(value)[rowIdx];
}

/// Get the mapped mask and maskedoff values.
static std::pair<Value, Value> getMaskPair(VPConversionState &state,
                                           unsigned idx) {
  if (auto pair = state.getMasks()) {
    auto [mask, maskedoff] = *pair;
    Value mappedMask = lookupTileComponent(mask, idx, state);
    Value mappedMaskedoff = nullptr;
    if (maskedoff) {
      mappedMaskedoff = lookupTileComponent(maskedoff, idx, state);
    }
    return std::make_pair(mappedMask, mappedMaskedoff);
  }
  return {nullptr, nullptr};
}

static mlir::VectorType getVectorType(Type elementType,
                                      VPConversionState &state) {
  return mlir::VectorType::get(state.vf, elementType, {state.scalable});
}

static Value
createSCFIfOp(OpBuilder &builder, Location loc, Value mask, Value maskedoff,
              std::function<Value(OpBuilder &, Location)> thenBuilder,
              std::function<Value(OpBuilder &, Location)> fallbackBuilder) {
  if (mask) {
    auto ifOp = scf::IfOp::create(
        builder, loc, mask,
        [&](OpBuilder &b, Location loc) {
          auto res = thenBuilder(b, loc);
          scf::YieldOp::create(b, loc, res);
        },
        [&](OpBuilder &b, Location loc) {
          if (maskedoff) {
            scf::YieldOp::create(b, loc, maskedoff);
          } else {
            Value fallback = fallbackBuilder(b, loc);
            scf::YieldOp::create(b, loc, fallback);
          }
        });
    return ifOp->getResult(0);
  } else {
    return thenBuilder(builder, loc);
  }
}

static Value buildZero(OpBuilder &builder, Location loc, Type type) {
  return arith::ConstantOp::create(builder, loc, type,
                                   builder.getZeroAttr(type));
}

static LogicalResult handleLoad(OpBuilder &builder, Location loc, Value result,
                                Value base, VPConversionState &state) {
  auto elementType = cast<MemRefType>(base.getType()).getElementType();
  auto shape = reifyZuanShape(builder, result);
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    // The current VP emitter stores tiles as a statically-sized row vector list
    // in `VPConversionState::tileMap`. A 2-D tile with a dynamic outer
    // dimension must therefore be removed earlier by `-zuan-stripmining`
    // and `ZuanTilingPattern`, or lowered through the loop-based path instead
    // of the VP path.
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }
  auto rank = shape->size();

  SmallVector<Value> vectors;
  auto zero = arith::ConstantIndexOp::create(builder, loc, 0);

  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector()) {
      SmallVector<OpFoldResult> offsets;
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

      if (rank == 2) {
        offsets.push_back(builder.getIndexAttr(i));
        sizes.push_back(builder.getIndexAttr(1));
      }
      offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(plan->evl);

      Value subview = memref::SubViewOp::create(builder, loc, base, offsets,
                                                sizes, strides);
      // Reduce the rank for the first dimension.
      auto reducedSubview = memref::SubViewOp::rankReduceIfNeeded(
          builder, loc, subview, {ShapedType::kDynamic});
      auto vecType = getVectorType(elementType, state);
      auto reducedType = cast<MemRefType>((*reducedSubview).getType());
      auto [reducedStrides, reducedOffset] = reducedType.getStridesAndOffset();
      (void)reducedOffset;

      // A rank-reduced row view can still carry a stride-0 layout from
      // broadcasted linalg indexing maps. `vp.load` models a contiguous vector
      // load, so materialize the scalar once and broadcast it explicitly for
      // those views instead of reading unrelated consecutive elements.
      if (reducedType.getRank() == 1 && reducedStrides.front() == 0) {
        Value scalar = memref::LoadOp::create(builder, loc, *reducedSubview,
                                              ValueRange{zero});
        auto broadcast =
            vector::BroadcastOp::create(builder, loc, vecType, scalar);
        auto predOp = vp::predicateOperation(builder, broadcast, plan->evl,
                                             mask, nullptr, maskedoff);
        vectors.push_back(predOp->getResult(0));
      } else {
        auto rowLoadOp = vp::LoadOp::create(builder, loc, vecType,
                                            *reducedSubview, ValueRange{zero});

        auto predOp = vp::predicateOperation(builder, rowLoadOp, plan->evl,
                                             mask, nullptr, maskedoff);
        vectors.push_back(predOp->getResult(0));
      }
    } else {
      // scalar load
      Value offset = arith::ConstantIndexOp::create(builder, loc, i);
      SmallVector<Value> offsets{};
      if (rank == 1) {
        offsets.push_back(offset);
      }
      if (mask) {
        auto ifOp = scf::IfOp::create(
            builder, loc, mask,
            [&](OpBuilder &b, Location loc) {
              Value loaded = memref::LoadOp::create(b, loc, base, offsets);
              scf::YieldOp::create(b, loc, loaded);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                scf::YieldOp::create(b, loc, maskedoff);
              } else {
                // Random value, default to 0
                Value zero = arith::ConstantOp::create(
                    b, loc, elementType, builder.getZeroAttr(elementType));
                scf::YieldOp::create(b, loc, zero);
              }
            });
        vectors.push_back(ifOp->getResult(0));
      } else {
        Value loaded = memref::LoadOp::create(builder, loc, base, offsets);
        vectors.push_back(loaded);
      }
    }
  }
  // Map the generated values.
  state.tileMap[result] = vectors;
  return success();
}

static LogicalResult handleStoreOp(OpBuilder &builder, StoreOp storeOp,
                                   VPConversionState &state) {
  auto loc = storeOp.getLoc();
  if (failed(ensureTileMapped(builder, storeOp.getValue(), state))) {
    return failure();
  }
  auto shape = reifyZuanShape(builder, storeOp.getValue());
  if (failed(shape)) {
    return failure();
  }
  auto base = ensureScalarValue(builder, storeOp.getBase(), state);
  if (failed(base)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    // See `handleLoad`: the current VP tile representation cannot materialize a
    // runtime-sized row list.
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }
  auto rank = shape->size();

  auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    auto row = lookupTileComponent(storeOp.getValue(), i, state);
    if (plan->hasVector()) {
      SmallVector<OpFoldResult> offsets;
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

      if (rank == 2) {
        offsets.push_back(builder.getIndexAttr(i));
        sizes.push_back(builder.getIndexAttr(1));
      }
      offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(plan->evl);

      Value subview = memref::SubViewOp::create(builder, loc, *base, offsets,
                                                sizes, strides);
      // Reduce the rank for the first dimension.
      auto reducedSubview = memref::SubViewOp::rankReduceIfNeeded(
          builder, loc, subview, {ShapedType::kDynamic});
      auto rowStoreOp = vp::StoreOp::create(builder, loc, row, *reducedSubview,
                                            ValueRange{zero});
      vp::predicateOperation(builder, rowStoreOp, plan->evl, mask, nullptr,
                             maskedoff);
    } else {
      // scalar store
      Value offset = arith::ConstantIndexOp::create(builder, loc, i);
      SmallVector<Value> offsets{};
      if (rank == 1) {
        offsets.push_back(offset);
      }
      if (mask) {
        scf::IfOp::create(
            builder, loc, mask,
            [&](OpBuilder &b, Location loc) {
              memref::StoreOp::create(b, loc, row, *base, offsets);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                // TODO: Check if this is needed.
                memref::StoreOp::create(b, loc, maskedoff, *base, offsets);
              }
            });
      } else {
        memref::StoreOp::create(builder, loc, row, *base, offsets);
      }
    }
  }
  return success();
}

static LogicalResult handleSplatOp(OpBuilder &builder, SplatOp splatOp,
                                   VPConversionState &state) {

  auto loc = splatOp->getLoc();

  auto result = splatOp.getResult();
  auto shape = reifyZuanShape(builder, result);
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }
  Value source = splatOp.getValue();
  bool sourceIsTile = isa<TileType>(source.getType());
  if (sourceIsTile) {
    if (failed(ensureTileMapped(builder, source, state))) {
      return failure();
    }
  } else {
    if (failed(ensureScalarValue(builder, source, state))) {
      return failure();
    }
  }

  SmallVector<Value> values;

  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector()) {
      if (sourceIsTile) {
        auto &sourceRows = state.tileMap[source];
        Value sourceRow =
            sourceRows[std::min<size_t>(i, sourceRows.size() - 1)];
        if (isa<VectorType>(sourceRow.getType())) {
          values.push_back(sourceRow);
        } else {
          auto vecType =
              getVectorType(result.getType().getElementType(), state);
          auto splat =
              vector::BroadcastOp::create(builder, loc, vecType, sourceRow);
          auto predOp = vp::predicateOperation(builder, splat, plan->evl, mask,
                                               nullptr, maskedoff);
          values.push_back(predOp->getResult(0));
        }
      } else {
        auto scalar = ensureScalarValue(builder, source, state);
        if (failed(scalar)) {
          return failure();
        }
        auto vecType = getVectorType(result.getType().getElementType(), state);
        auto splat =
            vector::BroadcastOp::create(builder, loc, vecType, *scalar);
        auto predOp = vp::predicateOperation(builder, splat, plan->evl, mask,
                                             nullptr, maskedoff);
        values.push_back(predOp->getResult(0));
      }
    } else {
      Value scalar = nullptr;
      if (sourceIsTile) {
        scalar = lookupTileComponent(source, 0, state);
      } else {
        auto mappedScalar = ensureScalarValue(builder, source, state);
        if (failed(mappedScalar)) {
          return failure();
        }
        scalar = *mappedScalar;
      }
      if (mask) {
        auto ifOp = scf::IfOp::create(
            builder, loc, mask,
            [&](OpBuilder &b, Location loc) {
              scf::YieldOp::create(b, loc, scalar);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                scf::YieldOp::create(b, loc, maskedoff);
              } else {
                // Random value, default to 0
                auto type = result.getType().getElementType();
                Value zero = arith::ConstantOp::create(
                    b, loc, type, builder.getZeroAttr(type));
                scf::YieldOp::create(b, loc, zero);
              }
            });
        values.push_back(ifOp->getResult(0));
      } else {
        values.push_back(scalar);
      }
    }
  }
  state.tileMap[result] = values;
  return success();
}

template <typename ElementwiseOp>
static LogicalResult handleElementwise(OpBuilder &builder, Operation *op,
                                       VPConversionState &state) {
  auto loc = op->getLoc();
  for (Value operand : op->getOperands()) {
    if (failed(ensureTileMapped(builder, operand, state))) {
      return failure();
    }
  }
  auto shape = reifyZuanShape(builder, op->getResult(0));
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  SmallVector<Value> resultValues;
  for (unsigned i = 0; i < *rows; ++i) {
    SmallVector<Value> operands;

    for (auto operand : op->getOperands()) {
      operands.push_back(lookupTileComponent(operand, i, state));
    }

    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector()) {
      auto elementwiseOp = ElementwiseOp::create(builder, loc, operands);
      auto predOp = vp::predicateOperation(builder, elementwiseOp, plan->evl,
                                           mask, nullptr, maskedoff);
      resultValues.push_back(predOp->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value res = ElementwiseOp::create(b, loc, operands);
            return res;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, op->getResult(0).getType());
          });
      resultValues.push_back(val);
    }
  }

  state.tileMap[op->getResult(0)] = resultValues;
  return success();
}

template <typename ArithOp>
static LogicalResult handleArithOp(OpBuilder &builder, Operation *op,
                                   VPConversionState &state) {
  LLVM_DEBUG(DBGS() << "Handling arith op: " << *op << "\n");

  if (!isa<TileType>(op->getResult(0).getType())) {
    LLVM_DEBUG(DBGS() << "Result is not a tile type, cloning op\n");
    cloneAndMap(builder, op, state);
    return success();
  }

  Type elementType =
      cast<TileType>(op->getResult(0).getType()).getElementType();

  bool hasPassthru = op->hasAttr("zuan_passthru_operand");
  auto passthruIdxAttr =
      op->getAttrOfType<IntegerAttr>("zuan_passthru_operand");
  auto passthruIdx = passthruIdxAttr ? passthruIdxAttr.getInt() : 0;

  auto loc = op->getLoc();

  for (Value operand : op->getOperands()) {
    if (failed(ensureTileMapped(builder, operand, state))) {
      return failure();
    }
  }
  auto shape = reifyZuanShape(builder, op->getResult(0));
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  auto lhsValues = state.tileMap[op->getOperand(0)];
  auto rhsValues = state.tileMap[op->getOperand(1)];

  SmallVector<Value> resultValues;

  for (unsigned i = 0; i < *rows; ++i) {
    Value lhs = lhsValues[i];
    Value rhs = rhsValues[i];

    Value passthru = nullptr;
    if (hasPassthru) {
      passthru = passthruIdx == 0 ? lhs : rhs;
    }

    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector()) {
      auto arithOp = ArithOp::create(builder, loc, lhs, rhs);
      auto predOp = vp::predicateOperation(builder, arithOp, plan->evl, mask,
                                           passthru, maskedoff);
      resultValues.push_back(predOp->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value res = ArithOp::create(b, loc, lhs, rhs);
            return res;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, elementType);
          });
      resultValues.push_back(val);
    }
  }
  state.tileMap[op->getResult(0)] = resultValues;
  return success();
}

template <typename CmpOp>
static LogicalResult handleCmpOp(OpBuilder &builder, CmpOp op,
                                 VPConversionState &state) {
  LLVM_DEBUG(DBGS() << "Handling cmp op: " << *op << "\n");

  if (!isa<TileType>(op->getResult(0).getType())) {
    LLVM_DEBUG(DBGS() << "Result is not a tile type, cloning op\n");
    cloneAndMap(builder, op, state);
    return success();
  }

  Location loc = op->getLoc();
  if (failed(ensureTileMapped(builder, op->getOperand(0), state)) ||
      failed(ensureTileMapped(builder, op->getOperand(1), state))) {
    return failure();
  }
  auto shape = reifyZuanShape(builder, op->getResult(0));
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  auto lhsValues = state.tileMap[op->getOperand(0)];
  auto rhsValues = state.tileMap[op->getOperand(1)];

  SmallVector<Value> resultValues;

  for (unsigned i = 0; i < *rows; ++i) {
    Value lhs = lhsValues[i];
    Value rhs = rhsValues[i];

    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector()) {
      auto cmpOp = CmpOp::create(builder, loc, op.getPredicate(), lhs, rhs);
      Operation *predOp = vp::predicateOperation(builder, cmpOp, plan->evl,
                                                 mask, nullptr, maskedoff);
      resultValues.push_back(predOp->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value res = CmpOp::create(b, loc, op.getPredicate(), lhs, rhs);
            return res;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, builder.getI1Type());
          });
      resultValues.push_back(val);
    }
  }
  state.tileMap[op->getResult(0)] = resultValues;
  return success();
}

static LogicalResult handleReductionOp(OpBuilder &builder,
                                       ReductionOp reductionOp,
                                       VPConversionState &state) {
  auto loc = reductionOp->getLoc();
  if (failed(ensureTileMapped(builder, reductionOp.getTile(), state))) {
    return failure();
  }
  auto shape = reifyZuanShape(builder, reductionOp.getTile());
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan)) {
    return failure();
  }
  if (plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector ||
      plan->kind == VPShapePlan::Kind::RowPack2D) {
    return failure();
  }

  auto maskPair = getMaskPair(state, 0);
  auto mask = maskPair.first;
  auto maskedoff = maskPair.second;

  auto kind = static_cast<vector::CombiningKind>(reductionOp.getKind());
  Value init = reductionOp.getInit();
  if (init) {
    if (failed(ensureTileMapped(builder, init, state))) {
      return failure();
    }
    init = lookupTileComponent(init, 0, state);
  }

  if (plan->kind == VPShapePlan::Kind::Scalar) {
    Value scalar = lookupTileComponent(reductionOp.getTile(), 0, state);
    Value reduced = init
                        ? createCombiningOp(builder, loc, reductionOp.getKind(),
                                            scalar, init)
                        : scalar;
    if (mask) {
      reduced = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &, Location) { return reduced; },
          [&](OpBuilder &b, Location scalarLoc) {
            return maskedoff ? maskedoff
                             : buildZero(b, scalarLoc, reduced.getType());
          });
    }
    state.tileMap[reductionOp.getResult()] = {reduced};
    return success();
  }

  auto vector = lookupTileComponent(reductionOp.getTile(), 0, state);
  auto reduction = vector::ReductionOp::create(builder, loc, kind, vector, init,
                                               arith::FastMathFlags::reassoc);
  auto predOp = vp::predicateOperation(builder, reduction, plan->evl, mask,
                                       nullptr, maskedoff);
  state.tileMap[reductionOp.getResult()] = {predOp->getResult(0)};
  return success();
}

static LogicalResult handleStepOp(OpBuilder &builder, StepOp stepOp,
                                  VPConversionState &state) {
  auto loc = stepOp->getLoc();
  auto dim = stepOp.getDim().getZExtValue();
  auto start = ensureScalarValue(builder, stepOp.getStart(), state);
  if (failed(start)) {
    return failure();
  }
  auto shape = reifyZuanShape(builder, stepOp.getResult());
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  SmallVector<Value> values;
  auto elementType = stepOp.getResult().getType().getElementType();
  auto vectorType = getVectorType(elementType, state);

  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector() && dim == shape->size() - 1) {
      Operation *step = vp::StepOp::create(builder, loc, vectorType);
      step = vp::predicateOperation(builder, step, plan->evl, mask, nullptr,
                                    maskedoff);
      auto startSplat =
          vector::BroadcastOp::create(builder, loc, vectorType, *start);
      Operation *add =
          arith::AddIOp::create(builder, loc, startSplat, step->getResult(0));
      add = vp::predicateOperation(builder, add, plan->evl, mask, nullptr,
                                   maskedoff);
      values.push_back(add->getResult(0));
    } else if (plan->hasVector()) {
      auto increment = arith::ConstantOp::create(
          builder, loc, start->getType(),
          builder.getIntegerAttr(start->getType(), i));
      Value newStart = arith::AddIOp::create(builder, loc, *start, increment);
      Operation *splat =
          vector::BroadcastOp::create(builder, loc, vectorType, newStart);
      splat = vp::predicateOperation(builder, splat, plan->evl, mask, nullptr,
                                     maskedoff);
      values.push_back(splat->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value increment = arith::ConstantOp::create(
                b, loc, start->getType(),
                b.getIntegerAttr(start->getType(), i));
            Value newStart = arith::AddIOp::create(b, loc, *start, increment);
            return newStart;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, elementType);
          });
      values.push_back(val);
    }
  }

  state.tileMap[stepOp.getResult()] = values;
  return success();
}

Value createCastOp(OpBuilder &b, Location loc, CastKind kind, Type outType,
                   Value source) {
  Value casted;
  switch (kind) {
  case CastKind::BITCAST:
    casted = arith::BitcastOp::create(b, loc, outType, source);
    break;
  case CastKind::EXTF:
    casted = arith::ExtFOp::create(b, loc, outType, source);
    break;
  case CastKind::EXTSI:
    casted = arith::ExtSIOp::create(b, loc, outType, source);
    break;
  case CastKind::EXTUI:
    casted = arith::ExtUIOp::create(b, loc, outType, source);
    break;
  case CastKind::TRUNCI:
    casted = arith::TruncIOp::create(b, loc, outType, source);
    break;
  case CastKind::TRUNCF:
    casted = arith::TruncFOp::create(b, loc, outType, source);
    break;
  case CastKind::FPTOSI:
    casted = arith::FPToSIOp::create(b, loc, outType, source);
    break;
  case CastKind::FPTOUI:
    casted = arith::FPToUIOp::create(b, loc, outType, source);
    break;
  case CastKind::SITOFP:
    casted = arith::SIToFPOp::create(b, loc, outType, source);
    break;
  case CastKind::UITOFP:
    casted = arith::UIToFPOp::create(b, loc, outType, source);
    break;
  case CastKind::INDEXCAST:
    casted = arith::IndexCastOp::create(b, loc, outType, source);
    break;
  case CastKind::INDEXCASTUI:
    casted = arith::IndexCastOp::create(b, loc, outType, source);
    break;
  }
  return casted;
}

static LogicalResult handleCastOp(OpBuilder &b, CastOp castOp,
                                  VPConversionState &state) {
  auto loc = castOp->getLoc();
  auto source = castOp.getTile();
  auto result = castOp.getResult();
  auto outType = result.getType().getElementType();
  auto kind = castOp.getKind();

  if (failed(ensureTileMapped(b, source, state))) {
    return failure();
  }
  auto shape = reifyZuanShape(b, result);
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  SmallVector<Value> values;

  for (unsigned i = 0; i < *rows; ++i) {
    Value sourceRow = lookupTileComponent(source, i, state);
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector()) {
      auto outVectorType = getVectorType(outType, state);
      auto casted = createCastOp(b, loc, kind, outVectorType, sourceRow);
      auto predOp = vp::predicateOperation(b, casted.getDefiningOp(), plan->evl,
                                           mask, nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    } else {
      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value casted = createCastOp(b, loc, kind, outType, sourceRow);
            return casted;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, outType);
          });
      values.push_back(res);
    }
  }

  state.tileMap[result] = values;
  return success();
}

static LogicalResult handleSelectOp(OpBuilder &b, SelectOp selectOp,
                                    VPConversionState &state) {
  auto loc = selectOp->getLoc();

  auto cond = selectOp.getCond();
  auto lhs = selectOp.getLhs();
  auto rhs = selectOp.getRhs();

  if (failed(ensureTileMapped(b, cond, state)) ||
      failed(ensureTileMapped(b, lhs, state)) ||
      failed(ensureTileMapped(b, rhs, state))) {
    return failure();
  }
  auto shape = reifyZuanShape(b, selectOp.getResult());
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  SmallVector<Value> values;

  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    Value condRow = lookupTileComponent(cond, i, state);
    Value lhsRow = lookupTileComponent(lhs, i, state);
    Value rhsRow = lookupTileComponent(rhs, i, state);

    if (plan->hasVector()) {
      auto selectOp = arith::SelectOp::create(b, loc, condRow, lhsRow, rhsRow);
      auto predOp = vp::predicateOperation(b, selectOp, plan->evl, mask,
                                           nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    } else {
      // Scalar select
      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value selectOp =
                arith::SelectOp::create(b, loc, condRow, lhsRow, rhsRow);
            return selectOp;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, lhsRow.getType());
          });
      values.push_back(res);
    }
  }
  state.tileMap[selectOp.getResult()] = values;
  return success();
}

static LogicalResult handleOuterOp(OpBuilder &b, OuterOp outerOp,
                                   VPConversionState &state) {
  auto loc = outerOp->getLoc();

  auto lhs = outerOp.getLhs();
  auto rhs = outerOp.getRhs();

  if (failed(ensureTileMapped(b, lhs, state)) ||
      failed(ensureTileMapped(b, rhs, state))) {
    return failure();
  }
  auto lhsShape = reifyZuanShape(b, lhs);
  auto rhsShape = reifyZuanShape(b, rhs);
  auto resultShape = reifyZuanShape(b, outerOp.getResult());
  if (failed(lhsShape) || failed(rhsShape) || failed(resultShape)) {
    return failure();
  }

  auto lhsPlan = planVPShape(b, loc, *lhsShape, state);
  auto rhsPlan = planVPShape(b, loc, *rhsShape, state);
  auto resultPlan = planVPShape(b, loc, *resultShape, state);
  if (failed(lhsPlan) || failed(rhsPlan) || failed(resultPlan) ||
      lhsPlan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector ||
      rhsPlan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector ||
      resultPlan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto resultRows = getStaticRowCount(*resultPlan);
  if (failed(resultRows)) {
    return failure();
  }

  auto kind = outerOp.getKind();
  auto &lhsValues = state.tileMap[lhs];
  auto &rhsValues = state.tileMap[rhs];

  auto getLhsForRow = [&](unsigned row) -> FailureOr<Value> {
    if (lhsValues.size() == *resultRows) {
      return lhsValues[row];
    }
    if (lhsValues.size() != 1) {
      return failure();
    }
    Value lhsValue = lhsValues.front();
    if (resultPlan->kind == VPShapePlan::Kind::RowPack2D &&
        lhs.getType().getRank() == 1 && isa<VectorType>(lhsValue.getType())) {
      // `vector x vector -> row-pack` outer products keep the lhs as a single
      // vector value in `tileMap`, so materialize one scalar lane per result
      // row on demand.
      return vector::ExtractOp::create(
                 b, loc, lhsValue, ArrayRef<int64_t>{static_cast<int64_t>(row)})
          .getResult();
    }
    return lhsValue;
  };

  auto getRhsForRow = [&](unsigned row) -> FailureOr<Value> {
    if (rhsValues.size() == *resultRows) {
      return rhsValues[row];
    }
    if (rhsValues.size() == 1) {
      return rhsValues.front();
    }
    return failure();
  };

  SmallVector<Value> values;

  if (resultPlan->hasVector()) {
    for (unsigned i = 0; i < *resultRows; ++i) {
      auto maskPair = getMaskPair(state, i);
      auto mask = maskPair.first;
      auto maskedoff = maskPair.second;

      auto lhsRow = getLhsForRow(i);
      auto rhsRow = getRhsForRow(i);
      if (failed(lhsRow) || failed(rhsRow)) {
        return failure();
      }

      Value lhsVector = *lhsRow;
      if (!isa<VectorType>(lhsVector.getType())) {
        auto splat =
            vector::BroadcastOp::create(b, loc, (*rhsRow).getType(), lhsVector);
        lhsVector = vp::predicateOperation(b, splat, resultPlan->evl, mask,
                                           nullptr, maskedoff)
                        ->getResult(0);
      }
      auto outer = createCombiningOp(b, loc, kind, lhsVector, *rhsRow);
      auto predOp = vp::predicateOperation(
          b, outer.getDefiningOp(), resultPlan->evl, mask, nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    }
  } else {
    for (unsigned i = 0; i < *resultRows; ++i) {
      auto maskPair = getMaskPair(state, i);
      auto mask = maskPair.first;
      auto maskedoff = maskPair.second;

      auto lhsRow = getLhsForRow(i);
      auto rhsRow = getRhsForRow(i);
      if (failed(lhsRow) || failed(rhsRow)) {
        return failure();
      }

      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &inner, Location innerLoc) {
            return createCombiningOp(inner, innerLoc, kind, *lhsRow, *rhsRow);
          },
          [&](OpBuilder &inner, Location innerLoc) {
            return buildZero(inner, innerLoc, (*lhsRow).getType());
          });
      values.push_back(res);
    }
  }

  state.tileMap[outerOp.getResult()] = values;
  return success();
}

static LogicalResult handleScatterOp(OpBuilder &b, ScatterOp scatterOp,
                                     VPConversionState &state) {
  auto loc = scatterOp->getLoc();
  auto base = ensureScalarValue(b, scatterOp.getBase(), state);
  if (failed(base)) {
    return failure();
  }

  auto value = scatterOp.getValue();
  auto indices = scatterOp.getIndices();

  if (failed(ensureTileMapped(b, value, state))) {
    return failure();
  }
  for (Value index : indices) {
    if (failed(ensureTileMapped(b, index, state))) {
      return failure();
    }
  }
  auto shape = reifyZuanShape(b, value);
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    Value valueRow = lookupTileComponent(value, i, state);
    SmallVector<Value> indicesRow = llvm::map_to_vector(
        indices, [&](Value idx) { return lookupTileComponent(idx, i, state); });

    if (plan->hasVector()) {
      auto scatter = vp::ScatterOp::create(b, loc, valueRow, *base, indicesRow);
      vp::predicateOperation(b, scatter, plan->evl, mask, nullptr, maskedoff);
    } else {
      if (mask) {
        // Scalar stores
        scf::IfOp::create(
            b, loc, mask,
            [&](OpBuilder &b, Location loc) {
              memref::StoreOp::create(b, loc, valueRow, *base, indicesRow);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                memref::StoreOp::create(b, loc, maskedoff, *base, indicesRow);
              }
            });
      } else {
        memref::StoreOp::create(b, loc, valueRow, *base, indicesRow);
      }
    }
  }
  return success();
}

static LogicalResult handleGatherOp(OpBuilder &b, GatherOp gatherOp,
                                    VPConversionState &state) {
  auto loc = gatherOp->getLoc();
  auto base = ensureScalarValue(b, gatherOp.getBase(), state);
  if (failed(base)) {
    return failure();
  }

  auto indices = gatherOp.getIndices();
  auto result = gatherOp.getResult();

  auto vectorType = getVectorType(result.getType().getElementType(), state);

  for (Value index : indices) {
    if (failed(ensureTileMapped(b, index, state))) {
      return failure();
    }
  }
  auto shape = reifyZuanShape(b, result);
  if (failed(shape)) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) ||
      plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }

  SmallVector<Value> values;

  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    SmallVector<Value> indicesRow = llvm::map_to_vector(
        indices, [&](Value idx) { return lookupTileComponent(idx, i, state); });

    if (plan->hasVector()) {

      auto gather = vp::GatherOp::create(b, loc, vectorType, *base, indicesRow);
      auto predOp = vp::predicateOperation(b, gather, plan->evl, mask, nullptr,
                                           maskedoff);
      values.push_back(predOp->getResult(0));
    } else {
      // Scalar loads
      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value gather = memref::LoadOp::create(b, loc, *base, indicesRow);
            return gather;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, result.getType().getElementType());
          });
      values.push_back(res);
    }
  }

  state.tileMap[result] = values;
  return success();
}

static LogicalResult handleSCFForOp(OpBuilder &builder, scf::ForOp forOp,
                                    VPConversionState &state) {
  OpBuilder::InsertionGuard g(builder);

  auto inits = forOp.getInitArgs();
  auto iterArgs = forOp.getRegionIterArgs();
  auto results = forOp.getResults();

  SmallVector<Value> newInits;
  // Map all init values to the new values.
  for (auto init : inits) {
    if (state.tileMap.contains(init)) {
      newInits.append(state.tileMap[init]);
    } else if (isa<TileType>(init.getType())) {
      // Loop-carried tiles are flattened into the per-row VP representation,
      // so the replacement loop must expand a single tile init into multiple
      // scalar/vector iter operands before cloning the body.
      if (failed(ensureTileMapped(builder, init, state))) {
        return failure();
      }
      newInits.append(state.tileMap[init]);
    } else {
      auto mappedInit = ensureScalarValue(builder, init, state);
      if (failed(mappedInit)) {
        return failure();
      }
      newInits.push_back(*mappedInit);
    }
  }

  auto lb = ensureScalarValue(builder, forOp.getLowerBound(), state);
  auto ub = ensureScalarValue(builder, forOp.getUpperBound(), state);
  auto step = ensureScalarValue(builder, forOp.getStep(), state);
  if (failed(lb) || failed(ub) || failed(step)) {
    return failure();
  }

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), *lb, *ub, *step, newInits);

  size_t newIdx = 0;
  for (auto [init, iterArg, result] : llvm::zip(inits, iterArgs, results)) {
    if (state.tileMap.contains(init)) {
      auto size = state.tileMap[init].size();
      SmallVector<Value> iterArgVectors;
      SmallVector<Value> resultVectors;

      for (size_t i = 0; i < size; ++i) {
        iterArgVectors.push_back(newForOp.getRegionIterArgs()[newIdx]);
        resultVectors.push_back(newForOp.getResults()[newIdx]);
        newIdx++;
      }

      state.tileMap[iterArg] = iterArgVectors;
      state.tileMap[result] = resultVectors;
    } else {
      state.valueMap.map(iterArg, newForOp.getRegionIterArgs()[newIdx]);
      state.valueMap.map(result, newForOp.getResults()[newIdx]);
      newIdx++;
    }
  }

  // Still need to handle the induction vars, as they are not iter args.
  state.valueMap.map(forOp.getInductionVar(), newForOp.getInductionVar());
  builder.setInsertionPointToStart(newForOp.getBody());

  for (auto &op : forOp.getBodyRegion(0).getOps()) {
    if (failed(convertToVP(builder, &op, state))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult handleSCFWhileOp(OpBuilder &builder, scf::WhileOp whileOp,
                                      VPConversionState &state) {
  OpBuilder::InsertionGuard g(builder);

  auto inits = whileOp.getInits();

  SmallVector<Value> newInits;
  // Map all init values to the new values.
  for (auto init : inits) {
    if (state.tileMap.contains(init)) {
      newInits.append(state.tileMap[init]);
    } else if (isa<TileType>(init.getType())) {
      // Keep while-loop tile state explicit in the flattened VP form for the
      // same reason as `scf.for`: one tile may become multiple carried values.
      if (failed(ensureTileMapped(builder, init, state))) {
        return failure();
      }
      newInits.append(state.tileMap[init]);
    } else {
      auto mappedInit = ensureScalarValue(builder, init, state);
      if (failed(mappedInit)) {
        return failure();
      }
      newInits.push_back(*mappedInit);
    }
  }

  SmallVector<Type> newTypes;
  for (auto init : newInits) {
    newTypes.push_back(init.getType());
  }

  auto newWhileOp = scf::WhileOp::create(
      builder, whileOp.getLoc(), newTypes, newInits,
      [&](OpBuilder &b, Location loc, ValueRange args) {},
      [&](OpBuilder &b, Location loc, ValueRange args) {});

  {
    OpBuilder::InsertionGuard g(builder);

    builder.setInsertionPointToStart(newWhileOp.getBeforeBody());
    auto beforeArgs = whileOp.getBeforeArguments();
    auto newArgs = newWhileOp.getBeforeArguments();
    size_t newIdx = 0;
    for (auto [init, oldArg] : llvm::zip(inits, beforeArgs)) {
      if (state.tileMap.contains(init)) {
        auto size = state.tileMap[init].size();
        SmallVector<Value> values;
        for (size_t i = 0; i < size; ++i) {
          values.push_back(newArgs[newIdx]);
          newIdx++;
        }
        state.tileMap[oldArg] = values;
      } else {
        state.valueMap.map(oldArg, newArgs[newIdx]);
        newIdx++;
      }
    }

    for (auto &op : whileOp.getBefore().getOps()) {
      if (failed(convertToVP(builder, &op, state))) {
        return failure();
      }
    }
  }

  {
    OpBuilder::InsertionGuard g(builder);

    builder.setInsertionPointToStart(newWhileOp.getAfterBody());
    auto bodyArgs = whileOp.getAfterArguments();
    auto newArgs = newWhileOp.getAfterArguments();
    auto results = whileOp.getResults();
    auto newResults = newWhileOp.getResults();

    size_t newIdx = 0;
    for (auto [init, oldArg, oldResult] : llvm::zip(inits, bodyArgs, results)) {
      if (state.tileMap.contains(init)) {
        auto size = state.tileMap[init].size();
        SmallVector<Value> argValues;
        SmallVector<Value> resValues;
        for (size_t i = 0; i < size; ++i) {
          argValues.push_back(newArgs[newIdx]);
          resValues.push_back(newResults[newIdx]);

          newIdx++;
        }
        state.tileMap[oldArg] = argValues;
        state.tileMap[oldResult] = resValues;
      } else {
        state.valueMap.map(oldArg, newArgs[newIdx]);
        state.valueMap.map(oldResult, newResults[newIdx]);
        newIdx++;
      }
    }

    for (auto &op : whileOp.getAfter().getOps()) {
      if (failed(convertToVP(builder, &op, state))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult convertToVP(OpBuilder &builder, Operation *op,
                          VPConversionState &state) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](LoadOp loadOp) {
        auto base = ensureScalarValue(builder, loadOp.getBase(), state);
        if (failed(base) ||
            failed(handleLoad(builder, loadOp->getLoc(), loadOp.getResult(),
                              *base, state))) {
          return failure();
        }
        return success();
      })
      .Case([&](StoreOp storeOp) {
        return handleStoreOp(builder, storeOp, state);
      })
      .Case([&](SplatOp splatOp) {
        return handleSplatOp(builder, splatOp, state);
      })
      .Case<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp,
            arith::DivSIOp, arith::DivUIOp, arith::DivFOp, arith::SubIOp,
            arith::SubFOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
            arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::MaxSIOp,
            arith::MaxUIOp, arith::MaximumFOp, arith::MinimumFOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp>(
          [&](auto arithOp) -> LogicalResult {
            return handleArithOp<decltype(arithOp)>(builder, op, state);
          })
      .Case<arith::CmpIOp, arith::CmpFOp>([&](auto cmpOp) -> LogicalResult {
        return handleCmpOp(builder, cmpOp, state);
      })
      .Case([&](ReductionOp reductionOp) {
        return handleReductionOp(builder, reductionOp, state);
      })
      .Case([&](StepOp stepOp) { return handleStepOp(builder, stepOp, state); })
      .Case([&](CastOp castOp) { return handleCastOp(builder, castOp, state); })
      .Case([&](MaskOp maskOp) {
        if (failed(ensureTileMapped(builder, maskOp.getMask(), state))) {
          return failure();
        }
        if (auto maskedoff = maskOp.getMaskedoff()) {
          if (failed(ensureTileMapped(builder, maskedoff, state))) {
            return failure();
          }
        }
        auto mask = maskOp.getMask();
        auto maskedoff = maskOp.getMaskedoff();
        ScopedVPMaskState scopedMask(state, mask, maskedoff);
        auto maskedOp = maskOp.getMaskedOp();
        if (maskedOp) {
          if (failed(convertToVP(builder, maskedOp, state))) {
            return failure();
          }
        }
        auto yieldOp =
            cast<MaskYieldOp>(maskOp.getBody().front().getTerminator());
        auto operands = yieldOp.getOperands();
        auto results = maskOp->getResults();
        for (auto [opd, result] : llvm::zip(operands, results)) {
          state.tileMap[result] = state.tileMap[opd];
        }
        return success();
      })
      .Case([&](SelectOp selectOp) {
        return handleSelectOp(builder, selectOp, state);
      })
      .Case([&](OuterOp outerOp) {
        return handleOuterOp(builder, outerOp, state);
      })
      .Case([&](ScatterOp scatterOp) {
        return handleScatterOp(builder, scatterOp, state);
      })
      .Case([&](GatherOp gatherOp) {
        return handleGatherOp(builder, gatherOp, state);
      })
      .Case([&](ExtractOp extractOp) {
        if (failed(ensureTileMapped(builder, extractOp.getTile(), state))) {
          return failure();
        }
        state.valueMap.map(extractOp.getResult(),
                           lookupTileComponent(extractOp.getTile(), 0, state));
        return success();
      })
      .Case([&](scf::ForOp forOp) {
        return handleSCFForOp(builder, forOp, state);
      })
      .Case([&](scf::YieldOp yieldOp) {
        SmallVector<Value> newOperands;
        for (auto operand : op->getOperands()) {
          if (state.tileMap.contains(operand)) {
            newOperands.append(state.tileMap[operand]);
          } else {
            auto mapped = ensureScalarValue(builder, operand, state);
            if (failed(mapped)) {
              return failure();
            }
            newOperands.push_back(*mapped);
          }
        }
        scf::YieldOp::create(builder, op->getLoc(), newOperands);
        return success();
      })
      .Case([&](scf::WhileOp whileOp) {
        return handleSCFWhileOp(builder, whileOp, state);
      })
      .Case([&](scf::ConditionOp) {
        SmallVector<Value> newOperands;
        for (auto operand : op->getOperands()) {
          if (state.tileMap.contains(operand)) {
            newOperands.append(state.tileMap[operand]);
          } else {
            auto mapped = ensureScalarValue(builder, operand, state);
            if (failed(mapped)) {
              return failure();
            }
            newOperands.push_back(*mapped);
          }
        }
        scf::ConditionOp::create(builder, op->getLoc(), op->getResultTypes(),
                                 newOperands);
        return success();
      })
      .Case<math::RsqrtOp, math::ExpOp>([&](auto unaryOp) -> LogicalResult {
        return handleElementwise<decltype(unaryOp)>(builder, op, state);
      })
      .Default([&](Operation *op) -> LogicalResult {
        LLVM_DEBUG(DBGS() << "Fallback to clone: " << op->getName() << "\n");
        cloneAndMap(builder, op, state);
        return success();
      });
}

void VPConversionState::initialize(Operation *root) {
  BackwardSliceOptions options;
  options.inclusive = true;

  SetVector<Operation *> slice;
  (void)getBackwardSlice(root, &slice, options);
  llvm::SmallDenseSet<Operation *> inSlice(slice.begin(), slice.end());

  for (Operation *op : slice) {
    for (Value operand : op->getOperands()) {
      auto *def = operand.getDefiningOp();
      if (!def || !inSlice.contains(def)) {
        valueMap.map(operand, operand);
      }
    }
  }
}

} // namespace zuan
} // namespace mlir
