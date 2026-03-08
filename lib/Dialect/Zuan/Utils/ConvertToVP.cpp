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
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

#include "VP/IR/VP.h"
#include "Zuan/IR/Zuan.h"
#include "Zuan/Utils/ConvertToVP.h"
#include "Zuan/Utils/Unrolling.h"

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
  return cloned->getResult(result.getResultNumber());
}

static FailureOr<Value> materializeDimValue(OpBuilder &builder, Location loc,
                                            DimSize dim,
                                            VPConversionState &state) {
  if (auto value = dim.getValue()) {
    return cloneMappedValue(builder, *value, state);
  }
  return dim.getOrCreateValue(builder, loc);
}

static FailureOr<VPShapePlan> planVPShape(OpBuilder &builder, Location loc,
                                          const TileShape &shape,
                                          VPConversionState &state) {
  VPShapePlan plan;
  plan.rank = shape.rank();
  if (shape.empty()) {
    plan.kind = VPShapePlan::Kind::Scalar;
    return plan;
  }
  if (shape.rank() == 1) {
    auto evl = materializeDimValue(builder, loc, shape[0], state);
    if (failed(evl)) {
      return failure();
    }
    plan.kind = VPShapePlan::Kind::Vector1D;
    plan.evl = *evl;
    return plan;
  }
  if (shape.rank() == 2) {
    if (auto rows = shape[0].getInteger()) {
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

static FailureOr<Value> lookupScalarValue(OpBuilder &builder, Value value,
                                          VPConversionState &state) {
  if (state.valueMap.contains(value)) {
    return state.valueMap.lookup(value);
  }
  if (!value.getDefiningOp()) {
    return failure();
  }
  return cloneMappedValue(builder, value, state);
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
    auto ifOp = scf::IfOp::create(builder, 
        loc, mask,
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
                                Value base, ShapeInfo &shapeInfo,
                                VPConversionState &state) {
  auto elementType = cast<MemRefType>(base.getType()).getElementType();

  auto *shape = shapeInfo.lookupShape(result);
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    // The current VP emitter stores tiles as a statically-sized row vector list
    // in `VPConversionState::tileMap`. A 2-D tile with a dynamic outer
    // dimension must therefore be removed earlier by `-zuan-stripmining`
    // and `ZuanTilingPattern`, or lowered through the loop-based path instead of
    // the VP path.
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }
  auto rank = shape->rank();

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

      Value subview =
          memref::SubViewOp::create(builder, loc, base, offsets, sizes, strides);
      // Reduce the rank for the first dimension.
      auto reducedSubview = memref::SubViewOp::rankReduceIfNeeded(
          builder, loc, subview, {ShapedType::kDynamic});
      auto vecType = getVectorType(elementType, state);
      auto rowLoadOp = vp::LoadOp::create(builder, loc, vecType, *reducedSubview,
                                                  ValueRange{zero});

      auto predOp = vp::predicateOperation(builder, rowLoadOp, plan->evl, mask,
                                           nullptr, maskedoff);
      vectors.push_back(predOp->getResult(0));
    } else {
      // scalar load
      Value offset = arith::ConstantIndexOp::create(builder, loc, i);
      SmallVector<Value> offsets{};
      if (rank == 1) {
        offsets.push_back(offset);
      }
      if (mask) {
        auto ifOp = scf::IfOp::create(builder, 
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              Value loaded = memref::LoadOp::create(b, loc, base, offsets);
              scf::YieldOp::create(b, loc, loaded);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                scf::YieldOp::create(b, loc, maskedoff);
              } else {
                // Random value, default to 0
                Value zero = arith::ConstantOp::create(b, 
                    loc, elementType, builder.getZeroAttr(elementType));
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
                                   ShapeInfo &shapeInfo,
                                   VPConversionState &state) {
  auto loc = storeOp.getLoc();
  auto *shape = shapeInfo.lookupShape(storeOp.getValue());
  if (!shape) {
    return failure();
  }
  auto base = state.valueMap.lookup(storeOp.getBase());
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    // See `handleLoad`: the current VP tile representation cannot materialize a
    // runtime-sized row list.
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }
  auto rank = shape->rank();

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

      Value subview =
          memref::SubViewOp::create(builder, loc, base, offsets, sizes, strides);
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
        scf::IfOp::create(builder, 
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              memref::StoreOp::create(b, loc, row, base, offsets);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                // TODO: Check if this is needed.
                memref::StoreOp::create(b, loc, maskedoff, base, offsets);
              }
            });
      } else {
        memref::StoreOp::create(builder, loc, row, base, offsets);
      }
    }
  }
  return success();
}

static LogicalResult handleSplatOp(OpBuilder &builder, SplatOp splatOp,
                                   ShapeInfo &shapeInfo,
                                   VPConversionState &state) {

  auto loc = splatOp->getLoc();

  auto result = splatOp.getResult();
  auto *shape = shapeInfo.lookupShape(result);
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto rows = getStaticRowCount(*plan);
  if (failed(rows)) {
    return failure();
  }
  Value source = splatOp.getValue();
  bool sourceIsTile = isa<TileType>(source.getType());

  SmallVector<Value> values;

  for (unsigned i = 0; i < *rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (plan->hasVector()) {
      if (sourceIsTile) {
        auto &sourceRows = state.tileMap[source];
        Value sourceRow = sourceRows[std::min<size_t>(i, sourceRows.size() - 1)];
        if (isa<VectorType>(sourceRow.getType())) {
          values.push_back(sourceRow);
        } else {
          auto vecType = getVectorType(result.getType().getElementType(), state);
          auto splat = vector::BroadcastOp::create(builder, loc, vecType, sourceRow);
          auto predOp = vp::predicateOperation(builder, splat, plan->evl, mask,
                                               nullptr, maskedoff);
          values.push_back(predOp->getResult(0));
        }
      } else {
        auto scalar = lookupScalarValue(builder, source, state);
        if (failed(scalar)) {
          return failure();
        }
        auto vecType = getVectorType(result.getType().getElementType(), state);
        auto splat = vector::BroadcastOp::create(builder, loc, vecType, *scalar);
        auto predOp = vp::predicateOperation(builder, splat, plan->evl, mask,
                                             nullptr, maskedoff);
        values.push_back(predOp->getResult(0));
      }
    } else {
      Value scalar = nullptr;
      if (sourceIsTile) {
        scalar = lookupTileComponent(source, 0, state);
      } else {
        auto mappedScalar = lookupScalarValue(builder, source, state);
        if (failed(mappedScalar)) {
          return failure();
        }
        scalar = *mappedScalar;
      }
      if (mask) {
        auto ifOp = scf::IfOp::create(builder, 
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              scf::YieldOp::create(b, loc, scalar);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                scf::YieldOp::create(b, loc, maskedoff);
              } else {
                // Random value, default to 0
                auto type = result.getType().getElementType();
                Value zero = arith::ConstantOp::create(b, 
                    loc, type, builder.getZeroAttr(type));
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
                                       ShapeInfo &shapeInfo,
                                       VPConversionState &state) {
  auto loc = op->getLoc();
  auto *shape = shapeInfo.lookupShape(op->getResult(0));
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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
      auto predOp = vp::predicateOperation(builder, elementwiseOp, plan->evl, mask,
                                           nullptr, maskedoff);
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
                                   ShapeInfo &shapeInfo,
                                   VPConversionState &state) {
  LLVM_DEBUG(DBGS() << "Handling arith op: " << *op << "\n");

  if (!isa<TileType>(op->getResult(0).getType())) {
    LLVM_DEBUG(DBGS() << "Result is not a tile type, cloning op\n");
    builder.clone(*op, state.valueMap);
    return success();
  }

  Type elementType =
      cast<TileType>(op->getResult(0).getType()).getElementType();

  bool hasPassthru = op->hasAttr("zuan_passthru_operand");
  auto passthruIdxAttr =
      op->getAttrOfType<IntegerAttr>("zuan_passthru_operand");
  auto passthruIdx = passthruIdxAttr ? passthruIdxAttr.getInt() : 0;

  auto loc = op->getLoc();

  auto *shape = shapeInfo.lookupShape(op->getOperand(passthruIdx));
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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
                                 ShapeInfo &shapeInfo,
                                 VPConversionState &state) {
  LLVM_DEBUG(DBGS() << "Handling cmp op: " << *op << "\n");

  if (!isa<TileType>(op->getResult(0).getType())) {
    LLVM_DEBUG(DBGS() << "Result is not a tile type, cloning op\n");
    builder.clone(*op, state.valueMap);
    return success();
  }

  Location loc = op->getLoc();
  auto *shape = shapeInfo.lookupShape(op->getOperand(0));
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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
      Operation *predOp = vp::predicateOperation(builder, cmpOp, plan->evl, mask,
                                                 nullptr, maskedoff);
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

static LogicalResult handleReductionOp(OpBuilder &builder, ReductionOp reductionOp,
                                       ShapeInfo &shapeInfo,
                                       VPConversionState &state) {
  auto loc = reductionOp->getLoc();
  auto *shape = shapeInfo.lookupShape(reductionOp.getTile());
  if (!shape) {
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
    init = lookupTileComponent(init, 0, state);
  }

  if (plan->kind == VPShapePlan::Kind::Scalar) {
    Value scalar = lookupTileComponent(reductionOp.getTile(), 0, state);
    Value reduced = init ? createCombiningOp(builder, loc, reductionOp.getKind(),
                                             scalar, init)
                         : scalar;
    if (mask) {
      reduced = createSCFIfOp(builder, loc, mask, maskedoff,
                              [&](OpBuilder &, Location) { return reduced; },
                              [&](OpBuilder &b, Location scalarLoc) {
                                return maskedoff ? maskedoff
                                                 : buildZero(b, scalarLoc,
                                                             reduced.getType());
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
                                  ShapeInfo &shapeInfo,
                                  VPConversionState &state) {
  auto loc = stepOp->getLoc();
  auto dim = stepOp.getDim().getZExtValue();
  auto start = state.valueMap.lookup(stepOp.getStart());
  auto *shape = shapeInfo.lookupShape(stepOp.getResult());
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(builder, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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
          vector::BroadcastOp::create(builder, loc, vectorType, start);
      Operation *add =
          arith::AddIOp::create(builder, loc, startSplat, step->getResult(0));
      add =
          vp::predicateOperation(builder, add, plan->evl, mask, nullptr, maskedoff);
      values.push_back(add->getResult(0));
    } else if (plan->hasVector()) {
      auto increment = arith::ConstantOp::create(builder, 
          loc, start.getType(), builder.getIntegerAttr(start.getType(), i));
      Value newStart = arith::AddIOp::create(builder, loc, start, increment);
      Operation *splat =
          vector::BroadcastOp::create(builder, loc, vectorType, newStart);
      splat = vp::predicateOperation(builder, splat, plan->evl, mask, nullptr,
                                     maskedoff);
      values.push_back(splat->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value increment = arith::ConstantOp::create(b, 
                loc, start.getType(), b.getIntegerAttr(start.getType(), i));
            Value newStart = arith::AddIOp::create(b, loc, start, increment);
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

Value createCastOp(OpBuilder &b, Location loc, CastKind kind,
                          Type outType, Value source) {
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
                                  ShapeInfo &shapeInfo,
                                  VPConversionState &state) {
  auto loc = castOp->getLoc();
  auto source = castOp.getTile();
  auto result = castOp.getResult();
  auto outType = result.getType().getElementType();
  auto kind = castOp.getKind();

  auto *shape = shapeInfo.lookupShape(result);
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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
                                    ShapeInfo &shapeInfo,
                                    VPConversionState &state) {
  auto loc = selectOp->getLoc();

  auto cond = selectOp.getCond();
  auto lhs = selectOp.getLhs();
  auto rhs = selectOp.getRhs();

  auto *shape = shapeInfo.lookupShape(selectOp.getResult());
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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
      auto predOp =
          vp::predicateOperation(b, selectOp, plan->evl, mask, nullptr, maskedoff);
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
                                   ShapeInfo &shapeInfo,
                                   VPConversionState &state) {
  auto loc = outerOp->getLoc();

  auto lhs = outerOp.getLhs();
  auto rhs = outerOp.getRhs();

  auto *lhsShape = shapeInfo.lookupShape(lhs);
  auto *rhsShape = shapeInfo.lookupShape(rhs);
  if (!lhsShape || !rhsShape) {
    return failure();
  }

  auto lhsPlan = planVPShape(b, loc, *lhsShape, state);
  auto rhsPlan = planVPShape(b, loc, *rhsShape, state);
  if (failed(lhsPlan) || failed(rhsPlan) ||
      lhsPlan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector ||
      rhsPlan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
    return failure();
  }
  auto lhsRows = getStaticRowCount(*lhsPlan);
  auto rhsRows = getStaticRowCount(*rhsPlan);
  if (failed(lhsRows) || failed(rhsRows)) {
    return failure();
  }
  if (*rhsRows != 1) {
    return failure();
  }

  auto kind = outerOp.getKind();

  // 3 cases:
  // 1. 1-D x 1-D: const & dynamic
  // 2. 1-D x 0-D: const & scalar
  // 3. 0-D x 1-D: scalar & dynamic

  SmallVector<Value> values;

  if (rhsPlan->hasVector()) {
    for (unsigned i = 0; i < *lhsRows; ++i) {
      auto maskPair = getMaskPair(state, i);
      auto mask = maskPair.first;
      auto maskedoff = maskPair.second;

      Value lhsRow = lookupTileComponent(lhs, i, state);
      Value rhsRow = lookupTileComponent(rhs, 0, state);

      Operation *lhsSplat =
          vector::BroadcastOp::create(b, loc, rhsRow.getType(), lhsRow);
      lhsSplat = vp::predicateOperation(b, lhsSplat, rhsPlan->evl, mask, nullptr,
                                        maskedoff);
      auto outer =
          createCombiningOp(b, loc, kind, lhsSplat->getResult(0), rhsRow);
      auto predOp = vp::predicateOperation(b, outer.getDefiningOp(), rhsPlan->evl,
                                           mask, nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    }
  } else {
    // 1-D x 0-D
    for (unsigned i = 0; i < *lhsRows; ++i) {
      auto maskPair = getMaskPair(state, i);
      auto mask = maskPair.first;
      auto maskedoff = maskPair.second;

      Value lhsRow = lookupTileComponent(lhs, i, state);
      Value rhsRow = lookupTileComponent(rhs, 0, state);

      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value outer = createCombiningOp(b, loc, kind, lhsRow, rhsRow);
            return outer;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, lhsRow.getType());
          });

      values.push_back(res);
    }
  }

  state.tileMap[outerOp.getResult()] = values;
  return success();
}

static LogicalResult handleScatterOp(OpBuilder &b, ScatterOp scatterOp,
                                     ShapeInfo &shapeInfo,
                                     VPConversionState &state) {
  auto loc = scatterOp->getLoc();
  auto base = state.valueMap.lookup(scatterOp.getBase());

  auto value = scatterOp.getValue();
  auto indices = scatterOp.getIndices();

  auto *shape = shapeInfo.lookupShape(value);
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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
      auto scatter = vp::ScatterOp::create(b, loc, valueRow, base, indicesRow);
      vp::predicateOperation(b, scatter, plan->evl, mask, nullptr, maskedoff);
    } else {
      if (mask) {
        // Scalar stores
        scf::IfOp::create(b, 
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              memref::StoreOp::create(b, loc, valueRow, base, indicesRow);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                memref::StoreOp::create(b, loc, maskedoff, base, indicesRow);
              }
            });
      } else {
        memref::StoreOp::create(b, loc, valueRow, base, indicesRow);
      }
    }
  }
  return success();
}

static LogicalResult handleGatherOp(OpBuilder &b, GatherOp gatherOp,
                                    ShapeInfo &shapeInfo,
                                    VPConversionState &state) {
  auto loc = gatherOp->getLoc();
  auto base = state.valueMap.lookup(gatherOp.getBase());

  auto indices = gatherOp.getIndices();
  auto result = gatherOp.getResult();

  auto vectorType = getVectorType(result.getType().getElementType(), state);

  auto *shape = shapeInfo.lookupShape(result);
  if (!shape) {
    return failure();
  }
  auto plan = planVPShape(b, loc, *shape, state);
  if (failed(plan) || plan->kind == VPShapePlan::Kind::DynamicOuterLoopAndVector) {
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

      auto gather = vp::GatherOp::create(b, loc, vectorType, base, indicesRow);
      auto predOp =
          vp::predicateOperation(b, gather, plan->evl, mask, nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    } else {
      // Scalar loads
      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value gather = memref::LoadOp::create(b, loc, base, indicesRow);
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

static LogicalResult handleSCFForOp(RewriterBase &rewriter, scf::ForOp forOp,
                                    ShapeInfo &shapeInfo,
                                    VPConversionState &state) {
  OpBuilder::InsertionGuard g(rewriter);

  auto inits = forOp.getInitArgs();
  auto iterArgs = forOp.getRegionIterArgs();
  auto results = forOp.getResults();

  SmallVector<Value> newInits;
  // Map all init values to the new values.
  for (auto init : inits) {
    if (state.tileMap.contains(init)) {
      newInits.append(state.tileMap[init]);
    } else if (state.valueMap.contains(init)) {
      newInits.push_back(state.valueMap.lookup(init));
    } else {
      newInits.push_back(init);
    }
  }

  auto lb = state.valueMap.lookup(forOp.getLowerBound());
  auto ub = state.valueMap.lookup(forOp.getUpperBound());
  auto step = state.valueMap.lookup(forOp.getStep());

  auto newForOp =
      scf::ForOp::create(rewriter, forOp.getLoc(), lb, ub, step, newInits);

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
  rewriter.setInsertionPointToStart(newForOp.getBody());

  for (auto &op : forOp.getBodyRegion(0).getOps()) {
    if (failed(convertToVP(rewriter, &op, shapeInfo, state))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult handleSCFWhileOp(RewriterBase &rewriter, scf::WhileOp whileOp,
                                      ShapeInfo &shapeInfo,
                                      VPConversionState &state) {
  OpBuilder::InsertionGuard g(rewriter);

  auto inits = whileOp.getInits();

  SmallVector<Value> newInits;
  // Map all init values to the new values.
  for (auto init : inits) {
    if (state.tileMap.contains(init)) {
      newInits.append(state.tileMap[init]);
    } else if (state.valueMap.contains(init)) {
      newInits.push_back(state.valueMap.lookup(init));
    } else {
      newInits.push_back(init);
    }
  }

  SmallVector<Type> newTypes;
  for (auto init : newInits) {
    newTypes.push_back(init.getType());
  }

  auto newWhileOp = scf::WhileOp::create(rewriter, 
      whileOp.getLoc(), newTypes, newInits,
      [&](OpBuilder &b, Location loc, ValueRange args) {},
      [&](OpBuilder &b, Location loc, ValueRange args) {});

  {
    OpBuilder::InsertionGuard g(rewriter);

    rewriter.setInsertionPointToStart(newWhileOp.getBeforeBody());
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
      if (failed(convertToVP(rewriter, &op, shapeInfo, state))) {
        return failure();
      }
    }
  }

  {
    OpBuilder::InsertionGuard g(rewriter);

    rewriter.setInsertionPointToStart(newWhileOp.getAfterBody());
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
      if (failed(convertToVP(rewriter, &op, shapeInfo, state))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult convertToVP(RewriterBase &rewriter, Operation *op,
                          ShapeInfo &shapeInfo, VPConversionState &state) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](DynamicOp dynamicOp) {
        auto inits = dynamicOp.getInits();
        state.valueMap.map(inits, inits);
        auto args = dynamicOp.getBody().getArguments();
        for (auto [init, arg] : llvm::zip(inits, args)) {
          if (cast<TileType>(arg.getType()).getRank() > 2) {
            // This is something not canonicalized, no need to handle.
            if (!arg.getUses().empty()) {
              return rewriter.notifyMatchFailure(
                  dynamicOp,
                  "rank > 2 dynamic arguments must be dead before VP lowering");
            }
            return success();
          }
          if (failed(handleLoad(rewriter, arg.getLoc(), arg,
                                state.valueMap.lookup(init), shapeInfo,
                                state))) {
            return rewriter.notifyMatchFailure(dynamicOp,
                                               "failed to lower dynamic init");
          }
        }
        for (auto &op : dynamicOp.getBody().getOps()) {
          if (failed(convertToVP(rewriter, &op, shapeInfo, state))) {
            return failure();
          }
        }
        return success();
      })
      .Case([&](LoadOp loadOp) {
        if (failed(handleLoad(rewriter, loadOp->getLoc(), loadOp.getResult(),
                              state.valueMap.lookup(loadOp.getBase()), shapeInfo,
                              state))) {
          return rewriter.notifyMatchFailure(loadOp, "failed to lower load to VP");
        }
        return success();
      })
      .Case([&](StoreOp storeOp) {
        if (failed(handleStoreOp(rewriter, storeOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(storeOp,
                                             "failed to lower store to VP");
        }
        return success();
      })
      .Case([&](SplatOp splatOp) {
        if (failed(handleSplatOp(rewriter, splatOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(splatOp,
                                             "failed to lower splat to VP");
        }
        return success();
      })
      .Case([&](YieldOp yieldOp) {
        auto operands = yieldOp.getOperands();
        auto dynamicOp = yieldOp->getParentOfType<DynamicOp>();

        auto results = dynamicOp->getResults();
        for (auto [opd, result] : llvm::zip(operands, results)) {
          auto mappedOperand = state.tileMap[opd][0];
          rewriter.replaceAllUsesWith(result, mappedOperand);
        }
        auto region = &yieldOp.getBody();
        for (auto &op : region->getOps()) {
          if (failed(convertToVP(rewriter, &op, shapeInfo, state))) {
            return failure();
          }
        }
        return success();
      })
      .Case<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp,
            arith::DivSIOp, arith::DivUIOp, arith::DivFOp, arith::SubIOp,
            arith::SubFOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
            arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::MaxSIOp,
            arith::MaxUIOp, arith::MaximumFOp, arith::MinimumFOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp>(
          [&](auto arithOp) -> LogicalResult {
            if (failed(handleArithOp<decltype(arithOp)>(rewriter, op, shapeInfo,
                                                        state))) {
              return rewriter.notifyMatchFailure(op,
                                                 "failed to lower arith op to VP");
            }
            return success();
          })
      .Case<arith::CmpIOp, arith::CmpFOp>(
          [&](auto cmpOp) -> LogicalResult {
            if (failed(handleCmpOp(rewriter, cmpOp, shapeInfo, state))) {
              return rewriter.notifyMatchFailure(op,
                                                 "failed to lower cmp op to VP");
            }
            return success();
          })
      .Case([&](ReductionOp reductionOp) {
        if (failed(handleReductionOp(rewriter, reductionOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(
              reductionOp, "only scalar and 1-D vector reductions are supported");
        }
        return success();
      })
      .Case([&](StepOp stepOp) {
        if (failed(handleStepOp(rewriter, stepOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(stepOp, "failed to lower step to VP");
        }
        return success();
      })
      .Case([&](CastOp castOp) {
        if (failed(handleCastOp(rewriter, castOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(castOp, "failed to lower cast to VP");
        }
        return success();
      })
      .Case([&](MaskOp maskOp) {
        auto mask = maskOp.getMask();
        auto maskedoff = maskOp.getMaskedoff();
        ScopedVPMaskState scopedMask(state, mask, maskedoff);
        auto maskedOp = maskOp.getMaskedOp();
        if (maskedOp) {
          if (failed(convertToVP(rewriter, maskedOp, shapeInfo, state))) {
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
        if (failed(handleSelectOp(rewriter, selectOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(selectOp,
                                             "failed to lower select to VP");
        }
        return success();
      })
      .Case([&](OuterOp outerOp) {
        if (failed(handleOuterOp(rewriter, outerOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(outerOp,
                                             "failed to lower outer op to VP");
        }
        return success();
      })
      .Case([&](ScatterOp scatterOp) {
        if (failed(handleScatterOp(rewriter, scatterOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(
              scatterOp, "failed to lower scatter to VP");
        }
        return success();
      })
      .Case([&](GatherOp gatherOp) {
        if (failed(handleGatherOp(rewriter, gatherOp, shapeInfo, state))) {
          return rewriter.notifyMatchFailure(gatherOp,
                                             "failed to lower gather to VP");
        }
        return success();
      })
      .Case([&](scf::ForOp forOp) {
        return handleSCFForOp(rewriter, forOp, shapeInfo, state);
      })
      .Case([&](scf::YieldOp yieldOp) {
        SmallVector<Value> newOperands;
        for (auto operand : op->getOperands()) {
          if (state.tileMap.contains(operand)) {
            newOperands.append(state.tileMap[operand]);
          } else if (state.valueMap.contains(operand)) {
            newOperands.push_back(state.valueMap.lookup(operand));
          } else {
            newOperands.push_back(operand);
          }
        }
        scf::YieldOp::create(rewriter, op->getLoc(), newOperands);
        return success();
      })
      .Case([&](scf::WhileOp whileOp) {
        return handleSCFWhileOp(rewriter, whileOp, shapeInfo, state);
      })
      .Case([&](scf::ConditionOp) {
        SmallVector<Value> newOperands;
        for (auto operand : op->getOperands()) {
          if (state.tileMap.contains(operand)) {
            newOperands.append(state.tileMap[operand]);
          } else if (state.valueMap.contains(operand)) {
            newOperands.push_back(state.valueMap.lookup(operand));
          } else {
            newOperands.push_back(operand);
          }
        }
        scf::ConditionOp::create(rewriter, op->getLoc(), op->getResultTypes(),
                                          newOperands);
        return success();
      })
      .Case<math::RsqrtOp, math::ExpOp>([&](auto unaryOp) -> LogicalResult {
        if (failed(handleElementwise<decltype(unaryOp)>(rewriter, op, shapeInfo,
                                                        state))) {
          return rewriter.notifyMatchFailure(op,
                                             "failed to lower unary op to VP");
        }
        return success();
      })
      .Default([&](Operation *op) -> LogicalResult {
        LLVM_DEBUG(DBGS() << "Fallback to clone: " << op->getName() << "\n");
        rewriter.clone(*op, state.valueMap);
        return success();
      });
}

void VPConversionState::initialize(DynamicOp op) {
  SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(op.getBody(), valuesDefinedAbove);
  this->valueMap.map(valuesDefinedAbove.getArrayRef(),
                     valuesDefinedAbove.getArrayRef());
}

} // namespace zuan
} // namespace mlir
