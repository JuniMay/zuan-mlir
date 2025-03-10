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

static std::variant<int64_t, Value> getDim(OpFoldResult ofr) {
  if (auto val = dyn_cast<Value>(ofr)) {
    if (isa<vp::GetVLOp>(val.getDefiningOp())) {
      // Only setvl is allowed to define the value in the RVV backend.
      return val;
    } else if (auto op = dyn_cast<arith::ConstantOp>(val.getDefiningOp())) {
      // Canonicalize constant op to the corresponding integer value.
      return cast<IntegerAttr>(op.getValue()).getInt();
    }
  } else if (auto attr = ofr.dyn_cast<Attribute>()) {
    return cast<IntegerAttr>(attr).getInt();
  }
  LLVM_DEBUG(DBGS() << "Invalid dim value." << ofr);
  llvm_unreachable("Invalid dim value.");
}

/// Major Dim, Minor Dim, Rank
static std::tuple<int64_t, std::optional<Value>, int64_t>
getRankedShape(OpBuilder &builder, Location loc, ArrayRef<DimSize> shape,
               VPConversionState &state) {
  if (shape.size() == 2) {
    auto dim0 = getDim(shape[0].getOrCreateOpFoldResult(builder, loc));
    auto dim1 = getDim(shape[1].getOrCreateOpFoldResult(builder, loc));

    auto dim1Val = std::get<Value>(dim1);
    Value cloned = builder.clone(*dim1Val.getDefiningOp(), state.valueMap)
                       ->getResult(0); // XXX: idx 0 is not always correct

    return {std::get<int64_t>(dim0), cloned, 2};
  } else if (shape.size() == 1) {
    auto dim = getDim(shape[0].getOrCreateOpFoldResult(builder, loc));
    if (auto val = std::get_if<int64_t>(&dim)) {
      return {*val, std::nullopt, 1};
    } else if (auto val = std::get_if<Value>(&dim)) {
      Value cloned = builder.clone(*val->getDefiningOp(), state.valueMap)
                         ->getResult(0); // XXX: idx 0 is not always correct
      return {1, cloned, 1};
    } else {
      // unreachable
      llvm_unreachable("Invalid dim value.");
    }
  } else if (shape.empty()) {
    // 0-D Vector, this is just a scalar.
    return {1, std::nullopt, 0};
  } else {
    LLVM_DEBUG(DBGS() << "Invalid shape size." << shape.size());
    llvm_unreachable("Invalid shape size.");
  }
}

/// Get the mapped mask and maskedoff values.
static std::pair<Value, Value> getMaskPair(VPConversionState &state,
                                           unsigned idx) {
  if (auto pair = state.getMasks()) {
    auto [mask, maskedoff] = *pair;
    Value mappedMask = state.tileMap[mask][idx];
    Value mappedMaskedoff = nullptr;
    if (maskedoff) {
      mappedMaskedoff = state.tileMap[maskedoff][idx];
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
    auto ifOp = builder.create<scf::IfOp>(
        loc, mask,
        [&](OpBuilder &b, Location loc) {
          auto res = thenBuilder(b, loc);
          b.create<scf::YieldOp>(loc, res);
        },
        [&](OpBuilder &b, Location loc) {
          if (maskedoff) {
            b.create<scf::YieldOp>(loc, maskedoff);
          } else {
            Value fallback = fallbackBuilder(b, loc);
            b.create<scf::YieldOp>(loc, fallback);
          }
        });
    return ifOp->getResult(0);
  } else {
    return thenBuilder(builder, loc);
  }
}

static Value buildZero(OpBuilder &builder, Location loc, Type type) {
  return builder.create<arith::ConstantOp>(loc, type,
                                           builder.getZeroAttr(type));
}

static void handleLoad(OpBuilder &builder, Location loc, Value result,
                       Value base, ShapeInfo &shapeInfo,
                       VPConversionState &state) {
  auto elementType = cast<MemRefType>(base.getType()).getElementType();

  auto shape = shapeInfo.getShape(result);
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  SmallVector<Value> vectors;
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);

  for (int64_t i = 0; i < rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (cols.has_value()) {
      SmallVector<OpFoldResult> offsets;
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

      if (rank == 2) {
        offsets.push_back(builder.getIndexAttr(i));
        sizes.push_back(builder.getIndexAttr(1));
      }
      offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(*cols);

      Value subview =
          builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides);
      // Reduce the rank for the first dimension.
      auto reducedSubview = memref::SubViewOp::rankReduceIfNeeded(
          builder, loc, subview, {ShapedType::kDynamic});
      auto vecType = getVectorType(elementType, state);
      auto rowLoadOp = builder.create<vp::LoadOp>(loc, vecType, *reducedSubview,
                                                  ValueRange{zero});

      auto predOp = vp::predicateOperation(builder, rowLoadOp, *cols, mask,
                                           nullptr, maskedoff);
      vectors.push_back(predOp->getResult(0));
    } else {
      // scalar load
      Value offset = builder.create<arith::ConstantIndexOp>(loc, i);
      SmallVector<Value> offsets{};
      if (rank == 1) {
        offsets.push_back(offset);
      }
      if (mask) {
        auto ifOp = builder.create<scf::IfOp>(
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              Value loaded = b.create<memref::LoadOp>(loc, base, offsets);
              b.create<scf::YieldOp>(loc, loaded);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                b.create<scf::YieldOp>(loc, maskedoff);
              } else {
                // Random value, default to 0
                Value zero = b.create<arith::ConstantOp>(
                    loc, elementType, builder.getZeroAttr(elementType));
                b.create<scf::YieldOp>(loc, zero);
              }
            });
        vectors.push_back(ifOp->getResult(0));
      } else {
        Value loaded = builder.create<memref::LoadOp>(loc, base, offsets);
        vectors.push_back(loaded);
      }
    }
  }
  // Map the generated values.
  state.tileMap[result] = vectors;
}

static void handleStoreOp(OpBuilder &builder, StoreOp storeOp,
                          ShapeInfo &shapeInfo, VPConversionState &state) {
  auto loc = storeOp.getLoc();
  auto shape = shapeInfo.getShape(storeOp.getValue());
  auto base = state.valueMap.lookup(storeOp.getBase());
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (int64_t i = 0; i < rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    auto row = state.tileMap[storeOp.getValue()][i];
    if (cols.has_value()) {
      SmallVector<OpFoldResult> offsets;
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

      if (rank == 2) {
        offsets.push_back(builder.getIndexAttr(i));
        sizes.push_back(builder.getIndexAttr(1));
      }
      offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(*cols);

      Value subview =
          builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides);
      // Reduce the rank for the first dimension.
      auto reducedSubview = memref::SubViewOp::rankReduceIfNeeded(
          builder, loc, subview, {ShapedType::kDynamic});
      auto rowStoreOp = builder.create<vp::StoreOp>(loc, row, *reducedSubview,
                                                    ValueRange{zero});
      vp::predicateOperation(builder, rowStoreOp, *cols, mask, nullptr,
                             maskedoff);
    } else {
      // scalar store
      Value offset = builder.create<arith::ConstantIndexOp>(loc, i);
      SmallVector<Value> offsets{};
      if (rank == 1) {
        offsets.push_back(offset);
      }
      if (mask) {
        builder.create<scf::IfOp>(
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              b.create<memref::StoreOp>(loc, row, base, offsets);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                // TODO: Check if this is needed.
                b.create<memref::StoreOp>(loc, maskedoff, base, offsets);
              }
            });
      } else {
        builder.create<memref::StoreOp>(loc, row, base, offsets);
      }
    }
  }
}

static void handleSplatOp(OpBuilder &builder, SplatOp splatOp,
                          ShapeInfo &shapeInfo, VPConversionState &state) {

  auto loc = splatOp->getLoc();

  auto result = splatOp.getResult();
  auto shape = shapeInfo.getShape(result);
  auto source = splatOp.getValue();
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  SmallVector<Value> values;

  for (int64_t i = 0; i < rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (cols.has_value()) {
      if (isa<TileType>(source.getType())) {
        values.push_back(state.tileMap[source][0]);
      } else {
        auto vecType = getVectorType(result.getType().getElementType(), state);
        auto splatOp = builder.create<vector::SplatOp>(loc, vecType, source);
        auto predOp = vp::predicateOperation(builder, splatOp, *cols, mask,
                                             nullptr, maskedoff);
        values.push_back(predOp->getResult(0));
      }
    } else {
      // The source is a scalar, and the length is a constant (rows), so
      // we can just splat the scalar.
      auto scalar = state.valueMap.lookup(source);
      if (mask) {
        auto ifOp = builder.create<scf::IfOp>(
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              b.create<scf::YieldOp>(loc, scalar);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                b.create<scf::YieldOp>(loc, maskedoff);
              } else {
                // Random value, default to 0
                auto type = result.getType().getElementType();
                Value zero = b.create<arith::ConstantOp>(
                    loc, type, builder.getZeroAttr(type));
                b.create<scf::YieldOp>(loc, zero);
              }
            });
        values.push_back(ifOp->getResult(0));
      } else {
        values.push_back(scalar);
      }
    }
  }
  state.tileMap[result] = values;
}

template <typename ElementwiseOp>
static void handleElementwise(OpBuilder &builder, Operation *op,
                              ShapeInfo &shapeInfo, VPConversionState &state) {
  auto loc = op->getLoc();
  auto shape = shapeInfo.getShape(op->getResult(0));
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  SmallVector<Value> resultValues;
  for (int64_t i = 0; i < rows; ++i) {
    SmallVector<Value> operands;

    for (auto operand : op->getOperands()) {
      operands.push_back(state.tileMap[operand][i]);
    }

    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (cols.has_value()) {
      auto elementwiseOp = builder.create<ElementwiseOp>(loc, operands);
      auto predOp = vp::predicateOperation(builder, elementwiseOp, *cols, mask,
                                           nullptr, maskedoff);
      resultValues.push_back(predOp->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value res = b.create<ElementwiseOp>(loc, operands);
            return res;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, op->getResult(0).getType());
          });
      resultValues.push_back(val);
    }
  }

  state.tileMap[op->getResult(0)] = resultValues;
}

template <typename ArithOp>
static void handleArithOp(OpBuilder &builder, Operation *op,
                          ShapeInfo &shapeInfo, VPConversionState &state) {
  LLVM_DEBUG(DBGS() << "Handling arith op: " << *op << "\n");

  if (!isa<TileType>(op->getResult(0).getType())) {
    LLVM_DEBUG(DBGS() << "Result is not a tile type, cloning op\n");
    builder.clone(*op, state.valueMap);
    return;
  }

  Type elementType =
      cast<TileType>(op->getResult(0).getType()).getElementType();

  bool hasPassthru = op->hasAttr("zuan_passthru_operand");
  auto passthruIdxAttr =
      op->getAttrOfType<IntegerAttr>("zuan_passthru_operand");
  auto passthruIdx = passthruIdxAttr ? passthruIdxAttr.getInt() : 0;

  auto loc = op->getLoc();

  auto shape = shapeInfo.getShape(op->getOperand(passthruIdx));
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  auto lhsValues = state.tileMap[op->getOperand(0)];
  auto rhsValues = state.tileMap[op->getOperand(1)];

  SmallVector<Value> resultValues;

  for (int64_t i = 0; i < rows; ++i) {
    Value lhs = lhsValues[i];
    Value rhs = rhsValues[i];

    Value passthru = nullptr;
    if (hasPassthru) {
      passthru = passthruIdx == 0 ? lhs : rhs;
    }

    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (cols.has_value()) {
      auto arithOp = builder.create<ArithOp>(loc, lhs, rhs);
      auto predOp = vp::predicateOperation(builder, arithOp, *cols, mask,
                                           passthru, maskedoff);
      resultValues.push_back(predOp->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value res = b.create<ArithOp>(loc, lhs, rhs);
            return res;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, elementType);
          });
      resultValues.push_back(val);
    }
  }
  state.tileMap[op->getResult(0)] = resultValues;
}

template <typename CmpOp>
static void handleCmpOp(OpBuilder &builder, CmpOp op, ShapeInfo shapeInfo,
                        VPConversionState &state) {
  LLVM_DEBUG(DBGS() << "Handling cmp op: " << *op << "\n");

  if (!isa<TileType>(op->getResult(0).getType())) {
    LLVM_DEBUG(DBGS() << "Result is not a tile type, cloning op\n");
    builder.clone(*op, state.valueMap);
    return;
  }

  Location loc = op->getLoc();
  auto shape = shapeInfo.getShape(op->getOperand(0));
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  auto lhsValues = state.tileMap[op->getOperand(0)];
  auto rhsValues = state.tileMap[op->getOperand(1)];

  SmallVector<Value> resultValues;

  for (int64_t i = 0; i < rows; ++i) {
    Value lhs = lhsValues[i];
    Value rhs = rhsValues[i];

    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (cols.has_value()) {
      auto cmpOp = builder.create<CmpOp>(loc, op.getPredicate(), lhs, rhs);
      Operation *predOp = vp::predicateOperation(builder, cmpOp, *cols, mask,
                                                 nullptr, maskedoff);
      resultValues.push_back(predOp->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value res = b.create<CmpOp>(loc, op.getPredicate(), lhs, rhs);
            return res;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, builder.getI1Type());
          });
      resultValues.push_back(val);
    }
  }
  state.tileMap[op->getResult(0)] = resultValues;
}

static void handleReductionOp(OpBuilder &builder, ReductionOp reductionOp,
                              ShapeInfo &shapeInfo, VPConversionState &state) {
  auto loc = reductionOp->getLoc();
  auto shape = shapeInfo.getShape(reductionOp.getTile());
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  assert(rows == 1 && "ReductionOp only supports 1 row.");

  auto maskPair = getMaskPair(state, 0);
  auto mask = maskPair.first;
  auto maskedoff = maskPair.second;

  auto vector = state.tileMap[reductionOp.getTile()][0];
  auto kind = static_cast<vector::CombiningKind>(reductionOp.getKind());
  Value init = reductionOp.getInit();
  if (init) {
    init = state.tileMap[init][0];
  }

  auto reduction = builder.create<vector::ReductionOp>(
      loc, kind, vector, init, arith::FastMathFlags::reassoc);
  auto predOp = vp::predicateOperation(builder, reduction, *cols, mask, nullptr,
                                       maskedoff);
  /// The old reduction generates a tile, so map in tileMap.
  state.tileMap[reductionOp.getResult()] = {predOp->getResult(0)};
  // TODO: what if this is a col vector.
}

static void handleStepOp(OpBuilder &builder, StepOp stepOp,
                         ShapeInfo &shapeInfo, VPConversionState &state) {
  auto loc = stepOp->getLoc();
  auto dim = stepOp.getDim().getZExtValue();
  auto start = state.valueMap.lookup(stepOp.getStart());
  auto shape = shapeInfo.getShape(stepOp.getResult());
  auto [rows, cols, rank] = getRankedShape(builder, loc, *shape, state);

  SmallVector<Value> values;
  auto elementType = stepOp.getResult().getType().getElementType();
  auto vectorType = getVectorType(elementType, state);

  for (int64_t i = 0; i < rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (dim == shape->size() - 1 && cols.has_value()) {
      Operation *step = builder.create<vector::StepOp>(loc, vectorType);
      step = vp::predicateOperation(builder, step, *cols, mask, nullptr,
                                    maskedoff);
      auto startSplat = builder.create<vector::SplatOp>(loc, vectorType, start);
      Operation *add =
          builder.create<arith::AddIOp>(loc, startSplat, step->getResult(0));
      add =
          vp::predicateOperation(builder, add, *cols, mask, nullptr, maskedoff);
      values.push_back(add->getResult(0));
    } else if (cols.has_value()) {
      auto increment = builder.create<arith::ConstantOp>(
          loc, start.getType(), builder.getIntegerAttr(start.getType(), i));
      Value newStart = builder.create<arith::AddIOp>(loc, start, increment);
      Operation *splat =
          builder.create<vector::SplatOp>(loc, vectorType, newStart);
      splat = vp::predicateOperation(builder, splat, *cols, mask, nullptr,
                                     maskedoff);
      values.push_back(splat->getResult(0));
    } else {
      auto val = createSCFIfOp(
          builder, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value increment = b.create<arith::ConstantOp>(
                loc, start.getType(), b.getIntegerAttr(start.getType(), i));
            Value newStart = b.create<arith::AddIOp>(loc, start, increment);
            return newStart;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, elementType);
          });
      values.push_back(val);
    }
  }

  state.tileMap[stepOp.getResult()] = values;
}

static Value createCastOp(OpBuilder &b, Location loc, CastKind kind,
                          Type outType, Value source) {
  Value casted;
  switch (kind) {
  case CastKind::BITCAST:
    casted = b.create<arith::BitcastOp>(loc, outType, source);
    break;
  case CastKind::EXTF:
    casted = b.create<arith::ExtFOp>(loc, outType, source);
    break;
  case CastKind::EXTSI:
    casted = b.create<arith::ExtSIOp>(loc, outType, source);
    break;
  case CastKind::EXTUI:
    casted = b.create<arith::ExtUIOp>(loc, outType, source);
    break;
  case CastKind::TRUNCI:
    casted = b.create<arith::TruncIOp>(loc, outType, source);
    break;
  case CastKind::TRUNCF:
    casted = b.create<arith::TruncFOp>(loc, outType, source);
    break;
  case CastKind::FPTOSI:
    casted = b.create<arith::FPToSIOp>(loc, outType, source);
    break;
  case CastKind::FPTOUI:
    casted = b.create<arith::FPToUIOp>(loc, outType, source);
    break;
  case CastKind::SITOFP:
    casted = b.create<arith::SIToFPOp>(loc, outType, source);
    break;
  case CastKind::UITOFP:
    casted = b.create<arith::UIToFPOp>(loc, outType, source);
    break;
  case CastKind::INDEXCAST:
    casted = b.create<arith::IndexCastOp>(loc, outType, source);
    break;
  case CastKind::INDEXCASTUI:
    casted = b.create<arith::IndexCastOp>(loc, outType, source);
    break;
  }
  return casted;
}

static void handleCastOp(OpBuilder &b, CastOp castOp, ShapeInfo &shapeInfo,
                         VPConversionState &state) {
  auto loc = castOp->getLoc();
  auto source = castOp.getTile();
  auto result = castOp.getResult();
  auto outType = result.getType().getElementType();
  auto kind = castOp.getKind();

  auto shape = shapeInfo.getShape(result);
  auto [rows, cols, rank] = getRankedShape(b, loc, *shape, state);

  SmallVector<Value> values;

  for (int64_t i = 0; i < rows; ++i) {
    Value sourceRow = state.tileMap[source][i];
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    if (cols.has_value()) {
      auto outVectorType = getVectorType(outType, state);
      auto casted = createCastOp(b, loc, kind, outVectorType, sourceRow);
      auto predOp = vp::predicateOperation(b, casted.getDefiningOp(), *cols,
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
}

static void handleSelectOp(OpBuilder &b, SelectOp selectOp,
                           ShapeInfo &shapeInfo, VPConversionState &state) {
  auto loc = selectOp->getLoc();

  auto cond = selectOp.getCond();
  auto lhs = selectOp.getLhs();
  auto rhs = selectOp.getRhs();

  auto shape = shapeInfo.getShape(selectOp.getResult());
  auto [rows, cols, rank] = getRankedShape(b, loc, *shape, state);

  SmallVector<Value> values;

  for (int64_t i = 0; i < rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    Value condRow = state.tileMap[cond][i];
    Value lhsRow = state.tileMap[lhs][i];
    Value rhsRow = state.tileMap[rhs][i];

    if (cols.has_value()) {
      auto selectOp = b.create<arith::SelectOp>(loc, condRow, lhsRow, rhsRow);
      auto predOp =
          vp::predicateOperation(b, selectOp, *cols, mask, nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    } else {
      // Scalar select
      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value selectOp =
                b.create<arith::SelectOp>(loc, condRow, lhsRow, rhsRow);
            return selectOp;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, lhsRow.getType());
          });
      values.push_back(res);
    }
  }
  state.tileMap[selectOp.getResult()] = values;
}

static void handleOuterOp(OpBuilder &b, OuterOp outerOp, ShapeInfo &shapeInfo,
                          VPConversionState &state) {
  auto loc = outerOp->getLoc();

  auto lhs = outerOp.getLhs();
  auto rhs = outerOp.getRhs();

  auto lhsShape = shapeInfo.getShape(lhs);
  auto rhsShape = shapeInfo.getShape(rhs);

  auto [lhsRows, lhsCols, lhsRank] = getRankedShape(b, loc, *lhsShape, state);
  auto [rhsRows, rhsCols, rhsRank] = getRankedShape(b, loc, *rhsShape, state);

  auto kind = outerOp.getKind();

  // 3 cases:
  // 1. 1-D x 1-D: const & dynamic
  // 2. 1-D x 0-D: const & scalar
  // 3. 0-D x 1-D: scalar & dynamic

  SmallVector<Value> values;

  if (rhsCols.has_value()) {
    for (int64_t i = 0; i < lhsRows; ++i) {
      auto maskPair = getMaskPair(state, i);
      auto mask = maskPair.first;
      auto maskedoff = maskPair.second;

      Value lhsRow = state.tileMap[lhs][i];
      Value rhsRow = state.tileMap[rhs][0];

      Operation *lhsSplat =
          b.create<vector::SplatOp>(loc, rhsRow.getType(), lhsRow);
      lhsSplat = vp::predicateOperation(b, lhsSplat, *rhsCols, mask, nullptr,
                                        maskedoff);
      auto outer =
          createCombiningOp(b, loc, kind, lhsSplat->getResult(0), rhsRow);
      auto predOp = vp::predicateOperation(b, outer.getDefiningOp(), *rhsCols,
                                           mask, nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    }
  } else {
    // 1-D x 0-D
    for (int64_t i = 0; i < lhsRows; ++i) {
      auto maskPair = getMaskPair(state, i);
      auto mask = maskPair.first;
      auto maskedoff = maskPair.second;

      Value lhsRow = state.tileMap[lhs][i];
      Value rhsRow = state.tileMap[rhs][0];

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
}

static void handleScatterOp(OpBuilder &b, ScatterOp scatterOp,
                            ShapeInfo &shapeInfo, VPConversionState &state) {
  auto loc = scatterOp->getLoc();
  auto base = state.valueMap.lookup(scatterOp.getBase());

  auto value = scatterOp.getValue();
  auto indices = scatterOp.getIndices();

  auto shape = shapeInfo.getShape(value);
  auto [rows, cols, rank] = getRankedShape(b, loc, *shape, state);

  for (int64_t i = 0; i < rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    Value valueRow = state.tileMap[value][i];
    SmallVector<Value> indicesRow = llvm::map_to_vector(
        indices, [&](Value idx) { return state.tileMap[idx][i]; });

    if (cols.has_value()) {
      auto scatter = b.create<vp::ScatterOp>(loc, valueRow, base, indicesRow);
      vp::predicateOperation(b, scatter, *cols, mask, nullptr, maskedoff);
    } else {
      if (mask) {
        // Scalar stores
        b.create<scf::IfOp>(
            loc, mask,
            [&](OpBuilder &b, Location loc) {
              b.create<memref::StoreOp>(loc, valueRow, base, indicesRow);
            },
            [&](OpBuilder &b, Location loc) {
              if (maskedoff) {
                b.create<memref::StoreOp>(loc, maskedoff, base, indicesRow);
              }
            });
      } else {
        b.create<memref::StoreOp>(loc, valueRow, base, indicesRow);
      }
    }
  }
}

static void handleGatherOp(OpBuilder &b, GatherOp gatherOp,
                           ShapeInfo &shapeInfo, VPConversionState &state) {
  auto loc = gatherOp->getLoc();
  auto base = state.valueMap.lookup(gatherOp.getBase());

  auto indices = gatherOp.getIndices();
  auto result = gatherOp.getResult();

  auto vectorType = getVectorType(result.getType().getElementType(), state);

  auto shape = shapeInfo.getShape(result);
  auto [rows, cols, rank] = getRankedShape(b, loc, *shape, state);

  SmallVector<Value> values;

  for (int64_t i = 0; i < rows; ++i) {
    auto maskPair = getMaskPair(state, i);
    auto mask = maskPair.first;
    auto maskedoff = maskPair.second;

    SmallVector<Value> indicesRow = llvm::map_to_vector(
        indices, [&](Value idx) { return state.tileMap[idx][i]; });

    if (cols.has_value()) {

      auto gather = b.create<vp::GatherOp>(loc, vectorType, base, indicesRow);
      auto predOp =
          vp::predicateOperation(b, gather, *cols, mask, nullptr, maskedoff);
      values.push_back(predOp->getResult(0));
    } else {
      // Scalar loads
      auto res = createSCFIfOp(
          b, loc, mask, maskedoff,
          [&](OpBuilder &b, Location loc) {
            Value gather = b.create<memref::LoadOp>(loc, base, indicesRow);
            return gather;
          },
          [&](OpBuilder &b, Location loc) {
            return buildZero(b, loc, result.getType().getElementType());
          });
      values.push_back(res);
    }
  }

  state.tileMap[result] = values;
}

static void handleSCFForOp(RewriterBase &rewriter, scf::ForOp forOp,
                           ShapeInfo &shapeInfo, VPConversionState &state) {
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
      rewriter.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, newInits);

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
    convertToVP(rewriter, &op, shapeInfo, state);
  }
}

static void handleSCFWhileOp(RewriterBase &rewriter, scf::WhileOp whileOp,
                             ShapeInfo &shapeInfo, VPConversionState &state) {
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

  auto newWhileOp = rewriter.create<scf::WhileOp>(
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
      convertToVP(rewriter, &op, shapeInfo, state);
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
      convertToVP(rewriter, &op, shapeInfo, state);
    }
  }
}

void convertToVP(RewriterBase &rewriter, Operation *op, ShapeInfo &shapeInfo,
                 VPConversionState &state) {
  TypeSwitch<Operation *, void>(op)
      .Case([&](DynamicOp dynamicOp) {
        auto inits = dynamicOp.getInits();
        state.valueMap.map(inits, inits);
        auto args = dynamicOp.getBody().getArguments();
        for (auto [init, arg] : llvm::zip(inits, args)) {
          if (cast<TileType>(arg.getType()).getRank() > 2) {
            // This is something not canonicalized, no need to handle.
            assert(arg.getUses().empty() &&
                   "Unexpected no uses for rank > 2 args.");
            continue;
          }
          handleLoad(rewriter, arg.getLoc(), arg, state.valueMap.lookup(init),
                     shapeInfo, state);
        }
        for (auto &op : dynamicOp.getBody().getOps()) {
          convertToVP(rewriter, &op, shapeInfo, state);
        }
      })
      .Case([&](LoadOp loadOp) {
        handleLoad(rewriter, loadOp->getLoc(), loadOp.getResult(),
                   state.valueMap.lookup(loadOp.getBase()), shapeInfo, state);
      })
      .Case([&](StoreOp storeOp) {
        handleStoreOp(rewriter, storeOp, shapeInfo, state);
      })
      .Case([&](SplatOp splatOp) {
        handleSplatOp(rewriter, splatOp, shapeInfo, state);
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
          convertToVP(rewriter, &op, shapeInfo, state);
        }
      })
      .Case<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp,
            arith::DivSIOp, arith::DivUIOp, arith::DivFOp, arith::SubIOp,
            arith::SubFOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
            arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::MaxSIOp,
            arith::MaxUIOp, arith::MaximumFOp, arith::MinimumFOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp>(
          [&](auto arithOp) {
            handleArithOp<decltype(arithOp)>(rewriter, op, shapeInfo, state);
          })
      .Case<arith::CmpIOp, arith::CmpFOp>(
          [&](auto cmpOp) { handleCmpOp(rewriter, cmpOp, shapeInfo, state); })
      .Case([&](ReductionOp reductionOp) {
        handleReductionOp(rewriter, reductionOp, shapeInfo, state);
      })
      .Case([&](StepOp stepOp) {
        handleStepOp(rewriter, stepOp, shapeInfo, state);
      })
      .Case([&](CastOp castOp) {
        handleCastOp(rewriter, castOp, shapeInfo, state);
      })
      .Case([&](MaskOp maskOp) {
        auto mask = maskOp.getMask();
        auto maskedoff = maskOp.getMaskedoff();
        state.setMasks(mask, maskedoff);
        auto maskedOp = maskOp.getMaskedOp();
        if (maskedOp) {
          convertToVP(rewriter, maskedOp, shapeInfo, state);
        }
        auto yieldOp =
            cast<MaskYieldOp>(maskOp.getBody().front().getTerminator());
        auto operands = yieldOp.getOperands();
        auto results = maskOp->getResults();
        for (auto [opd, result] : llvm::zip(operands, results)) {
          state.tileMap[result] = state.tileMap[opd];
        }

        state.resetMasks();
      })
      .Case([&](SelectOp selectOp) {
        handleSelectOp(rewriter, selectOp, shapeInfo, state);
      })
      .Case([&](OuterOp outerOp) {
        handleOuterOp(rewriter, outerOp, shapeInfo, state);
      })
      .Case([&](ScatterOp scatterOp) {
        handleScatterOp(rewriter, scatterOp, shapeInfo, state);
      })
      .Case([&](GatherOp gatherOp) {
        handleGatherOp(rewriter, gatherOp, shapeInfo, state);
      })
      .Case([&](scf::ForOp forOp) {
        handleSCFForOp(rewriter, forOp, shapeInfo, state);
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
        rewriter.create<scf::YieldOp>(op->getLoc(), newOperands);
      })
      .Case([&](scf::WhileOp whileOp) {
        handleSCFWhileOp(rewriter, whileOp, shapeInfo, state);
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
        rewriter.create<scf::ConditionOp>(op->getLoc(), op->getResultTypes(),
                                          newOperands);
      })
      .Case<math::RsqrtOp, math::ExpOp>([&](auto unaryOp) {
        handleElementwise<decltype(unaryOp)>(rewriter, op, shapeInfo, state);
      })
      .Default([&](Operation *op) {
        LLVM_DEBUG(DBGS() << "Fallback to clone: " << op->getName() << "\n");
        rewriter.clone(*op, state.valueMap);
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
