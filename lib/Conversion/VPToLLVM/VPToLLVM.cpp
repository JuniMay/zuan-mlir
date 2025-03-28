//===- VPToLLVM.cpp - VP to LLVM dialect conversion -------------*- C++ -*-===//
//
// This file implements the VP to LLVM dialect conversion patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <cmath>
#include <optional>

#include "Conversion/VPToLLVM.h"
#include "VP/IR/VP.h"

using namespace mlir;

namespace {

template <typename OpT> struct ForwardOperands : OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes()) {
      return rewriter.notifyMatchFailure(op, "operand types already match");
    }
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

/// SFINAE helper to get the LLVM VP intrinsic op for an arith op.
template <typename OpT> struct LLVMVPIntrOpMapping;

template <> struct LLVMVPIntrOpMapping<arith::AddIOp> {
  using VPIntrT = LLVM::VPAddOp;
};

template <> struct LLVMVPIntrOpMapping<arith::AddFOp> {
  using VPIntrT = LLVM::VPFAddOp;
};

template <> struct LLVMVPIntrOpMapping<arith::MulIOp> {
  using VPIntrT = LLVM::VPMulOp;
};

template <> struct LLVMVPIntrOpMapping<arith::MulFOp> {
  using VPIntrT = LLVM::VPFMulOp;
};

template <> struct LLVMVPIntrOpMapping<arith::SubIOp> {
  using VPIntrT = LLVM::VPSubOp;
};

template <> struct LLVMVPIntrOpMapping<arith::SubFOp> {
  using VPIntrT = LLVM::VPFSubOp;
};

template <> struct LLVMVPIntrOpMapping<arith::DivSIOp> {
  using VPIntrT = LLVM::VPSDivOp;
};

template <> struct LLVMVPIntrOpMapping<arith::DivUIOp> {
  using VPIntrT = LLVM::VPUDivOp;
};

template <> struct LLVMVPIntrOpMapping<arith::DivFOp> {
  using VPIntrT = LLVM::VPFDivOp;
};

template <> struct LLVMVPIntrOpMapping<arith::AndIOp> {
  using VPIntrT = LLVM::VPAndOp;
};

template <> struct LLVMVPIntrOpMapping<arith::OrIOp> {
  using VPIntrT = LLVM::VPOrOp;
};

template <> struct LLVMVPIntrOpMapping<arith::XOrIOp> {
  using VPIntrT = LLVM::VPOrOp;
};

template <> struct LLVMVPIntrOpMapping<arith::ShLIOp> {
  using VPIntrT = LLVM::VPShlOp;
};

template <> struct LLVMVPIntrOpMapping<arith::ShRUIOp> {
  using VPIntrT = LLVM::VPLShrOp;
};

template <> struct LLVMVPIntrOpMapping<arith::ShRSIOp> {
  using VPIntrT = LLVM::VPAShrOp;
};

template <> struct LLVMVPIntrOpMapping<arith::MaxSIOp> {
  using VPIntrT = LLVM::VPSMaxOp;
};

template <> struct LLVMVPIntrOpMapping<arith::RemFOp> {
  using VPIntrT = LLVM::VPFRemOp;
};

template <> struct LLVMVPIntrOpMapping<arith::RemSIOp> {
  using VPIntrT = LLVM::VPSRemOp;
};

template <> struct LLVMVPIntrOpMapping<arith::RemUIOp> {
  using VPIntrT = LLVM::VPURemOp;
};

template <> struct LLVMVPIntrOpMapping<arith::MaxUIOp> {
  using VPIntrT = LLVM::VPUMaxOp;
};

template <> struct LLVMVPIntrOpMapping<arith::TruncFOp> {
  using VPIntrT = LLVM::VPFPTruncOp;
};

template <> struct LLVMVPIntrOpMapping<arith::TruncIOp> {
  using VPIntrT = LLVM::VPTruncOp;
};

template <> struct LLVMVPIntrOpMapping<arith::ExtUIOp> {
  using VPIntrT = LLVM::VPZExtOp;
};

template <> struct LLVMVPIntrOpMapping<arith::ExtSIOp> {
  using VPIntrT = LLVM::VPSExtOp;
};

template <> struct LLVMVPIntrOpMapping<arith::ExtFOp> {
  using VPIntrT = LLVM::VPFPExtOp;
};

template <> struct LLVMVPIntrOpMapping<arith::FPToSIOp> {
  using VPIntrT = LLVM::VPFPToSIOp;
};

template <> struct LLVMVPIntrOpMapping<arith::FPToUIOp> {
  using VPIntrT = LLVM::VPFPToUIOp;
};

template <> struct LLVMVPIntrOpMapping<arith::SIToFPOp> {
  using VPIntrT = LLVM::VPSIToFPOp;
};

template <> struct LLVMVPIntrOpMapping<arith::UIToFPOp> {
  using VPIntrT = LLVM::VPUIToFPOp;
};

template <> struct LLVMVPIntrOpMapping<arith::IndexCastOp> {
  // Both index_cast and index_castui uses vptrunc when the target width is
  // smaller, The difference is when the target width is larger, index_castui
  // uses zext while index_cast uses sext.
  using VPIntrT = LLVM::VPSExtOp;
};

template <> struct LLVMVPIntrOpMapping<arith::IndexCastUIOp> {
  using VPIntrT = LLVM::VPZExtOp;
};

/// Some VP intrinsics are either not implemented in the LLVM dialect, or
/// does not support certain feature (e.g., fast-math flags). This trait
/// provides the intrinsic name for the arith op.
template <typename OpT> struct LLVMVPIntrNameMapping;

template <> struct LLVMVPIntrNameMapping<arith::MaximumFOp> {
  [[maybe_unused]]
  static constexpr llvm::StringRef name = "llvm.vp.maximum";
};

template <> struct LLVMVPIntrNameMapping<arith::MinimumFOp> {
  [[maybe_unused]]
  static constexpr llvm::StringRef name = "llvm.vp.minimum";
};

template <> struct LLVMVPIntrNameMapping<arith::MaxNumFOp> {
  [[maybe_unused]]
  static constexpr llvm::StringRef name = "llvm.vp.maxnum";
};

template <> struct LLVMVPIntrNameMapping<arith::MinNumFOp> {
  [[maybe_unused]]
  static constexpr llvm::StringRef name = "llvm.vp.minnum";
};

/// Materialize the given operand if needed.
[[maybe_unused]]
static Value materializeOperand(OpBuilder &b,
                                const LLVMTypeConverter *typeConverter,
                                Value operand) {
  auto targetType = typeConverter->convertType(operand.getType());
  if (targetType == operand.getType()) {
    return operand;
  }
  return typeConverter->materializeSourceConversion(b, operand.getLoc(),
                                                    targetType, {operand});
}

/// Build an all-true mask with the given shape in the vector type.
[[maybe_unused]]
static Value buildAllTrueMask(ConversionPatternRewriter &rewriter, Location loc,
                              VectorType vecType) {
  auto maskType = vecType.cloneWith(std::nullopt, rewriter.getI1Type());
  return rewriter.create<LLVM::ConstantOp>(
      loc, maskType, DenseIntElementsAttr::get(maskType, true));
}

/// Build the passthru value and the policy for the RVV intrinsic.
[[maybe_unused]]
static std::pair<Value, unsigned>
buildRVVPassthru(ConversionPatternRewriter &rewriter, Location loc, Value evl,
                 Value mask, Value passthru, Value maskedoff,
                 Type targetResType) {
  // Policy:
  //   tumu -> 00
  //   tama -> 11
  //   tamu -> 01
  //   tuma -> 10
  unsigned policy = 0; // tail & mask undisturbed

  if (!passthru && !maskedoff) {
    // Both not provided, use undef
    passthru = rewriter.create<LLVM::UndefOp>(loc, targetResType);
    policy = 3; // tail & mask agnostic
  } else if (passthru && !maskedoff) {
    // Do nothing.
    policy = 2; // tail-undisturbed & mask-agnostic
  } else if (!passthru && maskedoff) {
    // Use maskedoff as passthru.
    passthru = maskedoff;
    policy = 1; // mask-undisturbed & tail-agnostic
  } else if (passthru != maskedoff) {
    // Merge first to get the final passthru.
    auto allTrue =
        buildAllTrueMask(rewriter, loc, cast<VectorType>(targetResType));
    auto notmask = rewriter.create<LLVM::VPXorOp>(loc, mask.getType(), mask,
                                                  allTrue, allTrue, evl);
    auto passthruMerged = rewriter.create<LLVM::VPMergeMinOp>(
        loc, targetResType, notmask, maskedoff, passthru, evl);
    passthru = passthruMerged.getResult();
  }
  return {passthru, policy};
}

static Value buildPtrVec(Location loc, MemRefType type,
                         const LLVMTypeConverter *typeConverter,
                         Value memrefDescValue, ValueRange indices, Value mask,
                         Value evl, VectorType vectorType,
                         ConversionPatternRewriter &rewriter) {
  auto [strides, offset] = type.getStridesAndOffset();
  MemRefDescriptor memrefDesc(memrefDescValue);

  Type indexType = typeConverter->getIndexType();
  auto indexVecType = vectorType.cloneWith(std::nullopt, indexType);
  // The start address pointer of the memref.
  Value bufferPtr = memrefDesc.bufferPtr(rewriter, loc, *typeConverter, type);
  // Bitcast the buffer pointer to the index type.
  Value bufferPtrCasted =
      rewriter.create<LLVM::PtrToIntOp>(loc, indexType, bufferPtr);
  auto splatIntrName = rewriter.getStringAttr("llvm.experimental.vp.splat");
  // Splats the offset value to the vector type.
  Value indexVec =
      rewriter
          .create<LLVM::CallIntrinsicOp>(loc, indexVecType, splatIntrName,
                                         ValueRange{bufferPtrCasted, mask, evl})
          ->getResult(0);

  auto elementType = typeConverter->convertType(type.getElementType());
  // XXX: All element types should be byte-aligned.
  auto elementSize = elementType.getIntOrFloatBitWidth() / 8;
  Value elementSizeValue = rewriter.create<LLVM::ConstantOp>(
      loc, indexType, rewriter.getIntegerAttr(indexType, elementSize));
  // Splat the element size to the vector type.
  Value elementSizeVec = rewriter
                             .create<LLVM::CallIntrinsicOp>(
                                 loc, indexVecType, splatIntrName,
                                 ValueRange{elementSizeValue, mask, evl})
                             ->getResult(0);

  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) {
      Value stride = memrefDesc.stride(rewriter, loc, i);
      Value strideSplat = rewriter
                              .create<LLVM::CallIntrinsicOp>(
                                  loc, increment.getType(), splatIntrName,
                                  ValueRange{stride, mask, evl})
                              ->getResult(0);
      increment = rewriter.create<LLVM::VPMulOp>(
          loc, increment.getType(), increment, strideSplat, mask, evl);
    }
    // Multiply the increment with the element size.
    increment = rewriter.create<LLVM::VPMulOp>(
        loc, increment.getType(), increment, elementSizeVec, mask, evl);
    indexVec = rewriter.create<LLVM::VPAddOp>(loc, indexVec.getType(), indexVec,
                                              increment, mask, evl);
  }
  Type ptrType = rewriter.getType<LLVM::LLVMPointerType>();
  Type ptrVecType = rewriter.getType<LLVM::LLVMScalableVectorType>(
      ptrType, vectorType.getNumElements());
  Value ptrVec =
      rewriter.create<LLVM::VPIntToPtrOp>(loc, ptrVecType, indexVec, mask, evl);
  return ptrVec;
}

struct PredicateOpLowering : ConvertOpToLLVMPattern<vp::PredicateOp> {
private:
  bool enableRVV;
  bool enableMathEstimation;

public:
  PredicateOpLowering(LLVMTypeConverter &typeConverter, bool enableRVV,
                      bool enableMathEstimation)
      : ConvertOpToLLVMPattern<vp::PredicateOp>(typeConverter),
        enableRVV(enableRVV), enableMathEstimation(enableMathEstimation) {}

  LogicalResult
  matchAndRewrite(vp::PredicateOp op, vp::PredicateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicatedOp = &op.getBody().front().front();

    auto loc = op->getLoc();
    auto evlxlen = adaptor.getEvl();
    auto mask = adaptor.getMask();
    auto passthru = adaptor.getPassthru();
    auto maskedoff = adaptor.getMaskedoff();

    auto evl =
        rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), evlxlen);

    auto typeConverter = getTypeConverter();
    bool doMerge = true;
    /// The rewriting result.
    FailureOr<Operation *> loweredOp;

    TypeSwitch<Operation *, void>(predicatedOp)
        .Case<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp,
              arith::SubIOp, arith::SubFOp, arith::DivSIOp, arith::DivUIOp,
              arith::DivFOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
              arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::MaxSIOp,
              arith::MaxUIOp, arith::RemFOp, arith::RemSIOp, arith::RemUIOp>(
            [&](auto predicatedOp) {
              using VPIntrOp =
                  typename LLVMVPIntrOpMapping<decltype(predicatedOp)>::VPIntrT;

              Value lhs = predicatedOp.getLhs();
              Value rhs = predicatedOp.getRhs();
              if (!mask) {
                auto vectorType = cast<VectorType>(lhs.getType());
                mask = buildAllTrueMask(rewriter, loc, vectorType);
              }
              Type targetResType = typeConverter->convertType(
                  predicatedOp.getResult().getType());
              lhs = materializeOperand(rewriter, typeConverter, lhs);
              rhs = materializeOperand(rewriter, typeConverter, rhs);
              auto intrOp = rewriter.create<VPIntrOp>(loc, targetResType, lhs,
                                                      rhs, mask, evl);
              loweredOp = intrOp.getOperation();
            })
        .Case<arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
              arith::MinNumFOp>([&](auto predicatedOp) {
          using IntrName = LLVMVPIntrNameMapping<decltype(predicatedOp)>;
          // No need to conver the floating point types.
          Value lhs = predicatedOp.getLhs();
          Value rhs = predicatedOp.getRhs();
          // vp.maximum is not supported in llvm dialect yet, use call_intrinsic
          // as an alternative.
          StringAttr intrName = rewriter.getStringAttr(IntrName::name);
          if (!mask) {
            auto vectorType = cast<VectorType>(lhs.getType());
            mask = buildAllTrueMask(rewriter, loc, vectorType);
          }
          auto intrOp = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, lhs.getType(), intrName,
              ArrayRef<Value>({lhs, rhs, mask, evl}));
          loweredOp = intrOp.getOperation();
        })
        .Case([&](vector::ReductionOp redOp) {
          Value vec = redOp.getVector();
          Value acc = redOp.getAcc();

          vec = materializeOperand(rewriter, typeConverter, vec);
          auto targetResType =
              typeConverter->convertType(redOp.getResult().getType());
          if (acc) {
            acc = materializeOperand(rewriter, typeConverter, acc);
          } else {
            acc = rewriter.create<LLVM::ZeroOp>(loc, targetResType);
          }
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc, redOp.getSourceVectorType());
          }

          StringRef intrName;
          switch (redOp.getKind()) {
          case vector::CombiningKind::ADD:
            if (targetResType.isInteger()) {
              intrName = "llvm.vp.reduce.add";
            } else {
              intrName = "llvm.vp.reduce.fadd";
            }
            break;
          case vector::CombiningKind::MUL:
            if (targetResType.isInteger()) {
              intrName = "llvm.vp.reduce.mul";
            } else {
              intrName = "llvm.vp.reduce.fmul";
            }
            break;
          case vector::CombiningKind::MAXNUMF:
            intrName = "llvm.vp.reduce.fmax";
            break;
          case vector::CombiningKind::MAXIMUMF:
            intrName = "llvm.vp.reduce.fmaximum";
            break;
          case vector::CombiningKind::MINNUMF:
            intrName = "llvm.vp.reduce.fmin";
            break;
          case vector::CombiningKind::MINIMUMF:
            intrName = "llvm.vp.reduce.fminimum";
            break;
          case vector::CombiningKind::AND:
            intrName = "llvm.vp.reduce.and";
            break;
          case vector::CombiningKind::OR:
            intrName = "llvm.vp.reduce.or";
            break;
          case vector::CombiningKind::XOR:
            intrName = "llvm.vp.reduce.xor";
            break;
          case vector::CombiningKind::MAXSI:
            intrName = "llvm.vp.reduce.smax";
            break;
          case vector::CombiningKind::MAXUI:
            intrName = "llvm.vp.reduce.umax";
            break;
          case vector::CombiningKind::MINSI:
            intrName = "llvm.vp.reduce.smin";
            break;
          case vector::CombiningKind::MINUI:
            intrName = "llvm.vp.reduce.umin";
            break;
          }
          // Propaget fastmath flags.
          auto llvmFmf = mlir::arith::convertArithFastMathAttrToLLVM(
              redOp.getFastmathAttr());

          // VPIntrinsics does not support fast-math flags. Using call_intrinsic
          // here instead.
          auto intrOp = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, targetResType, rewriter.getStringAttr(intrName),
              ValueRange{acc, vec, mask, evl}, llvmFmf);
          loweredOp = intrOp.getOperation();
          doMerge = false;
        })
        .Case([&](vector::SplatOp splatOp) {
          auto src = splatOp.getOperand();
          src = materializeOperand(rewriter, typeConverter, src);
          auto targetResType =
              typeConverter->convertType(splatOp.getResult().getType());

          if (!mask) {
            mask =
                buildAllTrueMask(rewriter, loc, splatOp.getResult().getType());
          }

          auto vpSplat = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, targetResType,
              rewriter.getStringAttr("llvm.experimental.vp.splat"),
              ValueRange{src, mask, evl});
          loweredOp = vpSplat.getOperation();
        })
        .Case([&](vector::StepOp stepOp) {
          auto targetResType =
              typeConverter->convertType(stepOp.getResult().getType());
          if (enableRVV && (passthru || mask)) {
            // Only use RVV intrinsic if the passthru is provided or this is a
            // maked operation. Otherwise, the vid can be hoisted and cse-ed.
            auto [passthruMerged, policy] = buildRVVPassthru(
                rewriter, loc, evl, mask, passthru, maskedoff, targetResType);
            if (!mask) {
              // This passthru is either an undef or just the original passthru.
              auto intrOp = rewriter.create<vp::RVVIntrVidOp>(
                  loc, targetResType, passthruMerged, evlxlen);
              loweredOp = intrOp.getOperation();
            } else {
              auto policyValue = rewriter.create<LLVM::ConstantOp>(
                  loc, rewriter.getI32Type(),
                  rewriter.getI32IntegerAttr(policy));
              auto intrOp = rewriter.create<vp::RVVIntrVidMaskedOp>(
                  loc, targetResType, passthruMerged, mask, evlxlen,
                  policyValue);
              loweredOp = intrOp.getOperation();
            }
            doMerge = false; // Already merged in the intrinsic.
          } else {
            // We cannot lower the exact operation. `stepvector` is used to
            // mimic the behavior. The mask and passthru will be merged later
            // with the result.
            auto intrOp =
                rewriter.create<LLVM::StepVectorOp>(loc, targetResType);
            loweredOp = intrOp.getOperation();
          }
        })
        .Case([&](vp::StepOp stepOp) {
          // Same as the vector.step operation.
          auto targetResType =
              typeConverter->convertType(stepOp.getResult().getType());
          if (enableRVV && (passthru || mask)) {
            // Only use RVV intrinsic if the passthru is provided or this is a
            // maked operation. Otherwise, the vid can be hoisted and cse-ed.
            auto [passthruMerged, policy] = buildRVVPassthru(
                rewriter, loc, evl, mask, passthru, maskedoff, targetResType);
            if (!mask) {
              // This passthru is either an undef or just the original passthru.
              auto intrOp = rewriter.create<vp::RVVIntrVidOp>(
                  loc, targetResType, passthruMerged, evlxlen);
              loweredOp = intrOp.getOperation();
            } else {
              auto policyValue = rewriter.create<LLVM::ConstantOp>(
                  loc, rewriter.getI32Type(),
                  rewriter.getI32IntegerAttr(policy));
              auto intrOp = rewriter.create<vp::RVVIntrVidMaskedOp>(
                  loc, targetResType, passthruMerged, mask, evlxlen,
                  policyValue);
              loweredOp = intrOp.getOperation();
            }
            doMerge = false; // Already merged in the intrinsic.
          } else {
            // We cannot lower the exact operation. `stepvector` is used to
            // mimic the behavior. The mask and passthru will be merged later
            // with the result.
            auto intrOp =
                rewriter.create<LLVM::StepVectorOp>(loc, targetResType);
            loweredOp = intrOp.getOperation();
          }
        })
        .Case([&](vector::FMAOp fmaOp) {
          Value lhs = fmaOp.getLhs();
          Value rhs = fmaOp.getRhs();
          Value acc = fmaOp.getAcc();
          if (!mask) {
            auto vectorType = cast<VectorType>(lhs.getType());
            mask = buildAllTrueMask(rewriter, loc, vectorType);
          }
          auto targetResType =
              typeConverter->convertType(fmaOp.getResult().getType());
          lhs = materializeOperand(rewriter, typeConverter, lhs);
          rhs = materializeOperand(rewriter, typeConverter, rhs);
          acc = materializeOperand(rewriter, typeConverter, acc);
          auto intrOp = rewriter.create<LLVM::VPFmaOp>(loc, targetResType, lhs,
                                                       rhs, acc, mask, evl);
          loweredOp = intrOp.getOperation();
        })
        .Case([&](vp::LoadOp loadOp) {
          auto vectorType = loadOp.getVectorType();
          vectorType = cast<VectorType>(typeConverter->convertType(vectorType));

          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc, vectorType);
          }

          auto baseStruct =
              materializeOperand(rewriter, typeConverter, loadOp.getBase());
          SmallVector<Value> indices = loadOp.getIndices();
          for (auto &index : indices) {
            index = materializeOperand(rewriter, typeConverter, index);
          }

          auto llvmPtrType = rewriter.getType<LLVM::LLVMPointerType>();
          auto dataPtr = getStridedElementPtr(loc, loadOp.getMemRefType(),
                                              baseStruct, indices, rewriter);
          auto dataPtrCasted =
              rewriter.create<LLVM::BitcastOp>(loc, llvmPtrType, dataPtr);

          auto [strides, offset] = loadOp.getMemRefType().getStridesAndOffset();
          auto rank = loadOp.getMemRefType().getRank();
          if (strides[rank - 1] != 1) {
            if (strides[rank - 1] == 0) {
              // 0-stride load, make it scalar load & splat
              Value load = rewriter.create<LLVM::LoadOp>(
                  loc, vectorType.getElementType(), dataPtrCasted);
              auto splatOp = rewriter.create<LLVM::CallIntrinsicOp>(
                  loc, vectorType,
                  rewriter.getStringAttr("llvm.experimental.vp.splat"),
                  ValueRange{load, mask, evl});
              loweredOp = splatOp.getOperation();
            } else {
              // strided load
              MemRefDescriptor memrefDesc(baseStruct);
              Value stride = memrefDesc.stride(rewriter, loc, rank - 1);
              // Need to multiply the stride with the byte size of the element.
              auto bytewidth =
                  vectorType.getElementType().getIntOrFloatBitWidth() / 8;
              Value scale = rewriter.create<LLVM::ConstantOp>(
                  loc, stride.getType(), rewriter.getI64IntegerAttr(bytewidth));
              Value strideScaled =
                  rewriter.create<LLVM::MulOp>(loc, stride, scale);
              auto intrOp = rewriter.create<LLVM::VPStridedLoadOp>(
                  loc, vectorType, dataPtrCasted, strideScaled, mask, evl);
              loweredOp = intrOp.getOperation();
            }
          } else {
            if (vectorType.getElementType().isInteger(1) && enableRVV) {
              // XXX: Ignore mask here
              // XXX: RISC-V intrinsic only support xlen vl
              auto intrOp = rewriter.create<vp::RVVIntrVlmOp>(
                  loc, vectorType, dataPtrCasted, evlxlen);
              loweredOp = intrOp.getOperation();
            } else {
              auto intrOp = rewriter.create<LLVM::VPLoadOp>(
                  loc, vectorType, dataPtrCasted, mask, evl);
              loweredOp = intrOp.getOperation();
            }
          }
        })
        .Case([&](vp::StoreOp storeOp) {
          auto value =
              materializeOperand(rewriter, typeConverter, storeOp.getValue());
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc, storeOp.getVectorType());
          }

          auto baseStruct =
              materializeOperand(rewriter, typeConverter, storeOp.getBase());
          SmallVector<Value> indices = storeOp.getIndices();
          for (auto &index : indices) {
            index = materializeOperand(rewriter, typeConverter, index);
          }

          auto llvmPtrType = rewriter.getType<LLVM::LLVMPointerType>();
          auto dataPtr = getStridedElementPtr(loc, storeOp.getMemRefType(),
                                              baseStruct, indices, rewriter);
          auto dataPtrCasted =
              rewriter.create<LLVM::BitcastOp>(loc, llvmPtrType, dataPtr);

          auto [strides, offset] =
              storeOp.getMemRefType().getStridesAndOffset();
          auto rank = storeOp.getMemRefType().getRank();
          if (strides[rank - 1] != 1) {
            // strided load
            MemRefDescriptor memrefDesc(baseStruct);
            unsigned pos = storeOp.getMemRefType().getRank() - 1;
            Value stride = memrefDesc.stride(rewriter, loc, pos);
            // Need to multiply the stride with the byte size of the element.
            auto bytewidth = storeOp.getMemRefType()
                                 .getElementType()
                                 .getIntOrFloatBitWidth() /
                             8;
            Value scale = rewriter.create<LLVM::ConstantOp>(
                loc, stride.getType(), rewriter.getI64IntegerAttr(bytewidth));
            Value strideScaled =
                rewriter.create<LLVM::MulOp>(loc, stride, scale);
            auto intrOp = rewriter.create<LLVM::VPStridedStoreOp>(
                loc, value, dataPtrCasted, strideScaled, mask, evl);
            loweredOp = intrOp.getOperation();
          } else {
            if (storeOp.getVectorType().getElementType().isInteger(1) &&
                enableRVV) {
              // XXX: Ignore mask here
              // XXX: RISC-V intrinsic only support xlen vl
              auto intrOp = rewriter.create<vp::RVVIntrVsmOp>(
                  loc, value, dataPtrCasted, evlxlen);
              loweredOp = intrOp.getOperation();
            } else {
              auto intrOp = rewriter.create<LLVM::VPStoreOp>(
                  loc, value, dataPtrCasted, mask, evl);
              loweredOp = intrOp.getOperation();
            }
          }
        })
        .Case([&](arith::SelectOp selectOp) {
          auto vcond = materializeOperand(rewriter, typeConverter,
                                          selectOp.getCondition());
          auto vtrue = materializeOperand(rewriter, typeConverter,
                                          selectOp.getTrueValue());
          auto vfalse = materializeOperand(rewriter, typeConverter,
                                           selectOp.getFalseValue());
          auto targetResType =
              typeConverter->convertType(selectOp.getResult().getType());

          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc,
                                    cast<VectorType>(vtrue.getType()));
          }
          auto intrOp = rewriter.create<LLVM::VPSelectMinOp>(
              loc, targetResType, vcond, vtrue, vfalse, evl);
          loweredOp = intrOp.getOperation();
        })
        .Case([&](vp::GatherOp gatherOp) {
          auto memrefType = gatherOp.getBase().getType();
          auto baseStruct =
              materializeOperand(rewriter, typeConverter, gatherOp.getBase());
          SmallVector<Value> indices = gatherOp.getIndices();
          for (auto &index : indices) {
            index = materializeOperand(rewriter, typeConverter, index);
          }
          auto resultType = gatherOp.getResult().getType();
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc, resultType);
          }
          auto ptrVec = buildPtrVec(loc, memrefType, typeConverter, baseStruct,
                                    indices, mask, evl, resultType, rewriter);

          auto targetResType = typeConverter->convertType(resultType);
          auto gatherIntrName = rewriter.getStringAttr("llvm.vp.gather");
          auto intrOp = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, targetResType, gatherIntrName,
              ValueRange{ptrVec, mask, evl});
          loweredOp = intrOp.getOperation();
        })
        .Case([&](vp::ScatterOp scatterOp) {
          auto memrefType = scatterOp.getBase().getType();
          auto baseStruct =
              materializeOperand(rewriter, typeConverter, scatterOp.getBase());
          SmallVector<Value> indices = scatterOp.getIndices();
          for (auto &index : indices) {
            index = materializeOperand(rewriter, typeConverter, index);
          }
          auto valueType = scatterOp.getValue().getType();
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc, valueType);
          }
          auto ptrVec = buildPtrVec(loc, memrefType, typeConverter, baseStruct,
                                    indices, mask, evl, valueType, rewriter);
          auto gatherIntrName = rewriter.getStringAttr("llvm.vp.scatter");
          auto value =
              materializeOperand(rewriter, typeConverter, scatterOp.getValue());
          auto intrOp = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, gatherIntrName, ValueRange{value, ptrVec, mask, evl});
          loweredOp = intrOp.getOperation();
        })
        .Case<arith::TruncFOp, arith::TruncIOp, arith::ExtUIOp, arith::ExtSIOp,
              arith::ExtFOp, arith::FPToSIOp, arith::FPToUIOp, arith::UIToFPOp,
              arith::SIToFPOp>([&](auto castOp) {
          using VPIntrOp =
              typename LLVMVPIntrOpMapping<decltype(castOp)>::VPIntrT;

          Value src = castOp->getOperand(0);
          Value res = castOp->getResult(0);
          src = materializeOperand(rewriter, typeConverter, src);
          Type targetResType = typeConverter->convertType(res.getType());
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc,
                                    cast<VectorType>(src.getType()));
          }
          auto intrOp =
              rewriter.create<VPIntrOp>(loc, targetResType, src, mask, evl);
          loweredOp = intrOp.getOperation();
        })
        .Case<arith::IndexCastOp, arith::IndexCastUIOp>([&](auto indexcastOp) {
          Value src = indexcastOp.getOperand();
          src = materializeOperand(rewriter, typeConverter, src);

          unsigned srcElemBits = cast<VectorType>(src.getType())
                                     .getElementType()
                                     .getIntOrFloatBitWidth();
          auto targetResType =
              typeConverter->convertType(indexcastOp.getResult().getType());
          unsigned dstElemBits = cast<VectorType>(targetResType)
                                     .getElementType()
                                     .getIntOrFloatBitWidth();
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc,
                                    cast<VectorType>(src.getType()));
          }
          if (srcElemBits > dstElemBits) {
            auto intrOp = rewriter.create<LLVM::VPTruncOp>(loc, targetResType,
                                                           src, mask, evl);
            loweredOp = intrOp.getOperation();
          } else if (srcElemBits == dstElemBits) {
            loweredOp = src.getDefiningOp();
          } else {
            // Select between zext and sext.
            using VPIntrOp =
                typename LLVMVPIntrOpMapping<decltype(indexcastOp)>::VPIntrT;
            auto intrOp =
                rewriter.create<VPIntrOp>(loc, targetResType, src, mask, evl);
            loweredOp = intrOp.getOperation();
          }
        })
        .Case([&](arith::CmpFOp cmpOp) {
          auto lhs =
              materializeOperand(rewriter, typeConverter, cmpOp.getLhs());
          auto rhs =
              materializeOperand(rewriter, typeConverter, cmpOp.getRhs());
          auto targetResType =
              typeConverter->convertType(cmpOp.getResult().getType());
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc,
                                    cast<VectorType>(lhs.getType()));
          }
          auto predicate = cmpOp.getPredicate();
          StringRef prefStr = arith::stringifyCmpFPredicate(predicate);

          auto llvmPred = rewriter.getStringAttr(prefStr);
          auto intrOp = rewriter.create<vp::VPIntrFCmpOp>(loc, targetResType,
                                                          lhs, rhs, mask, evl);
          intrOp->setAttr("predicate", llvmPred);
          loweredOp = intrOp.getOperation();
        })
        .Case([&](arith::CmpIOp cmpOp) {
          auto lhs =
              materializeOperand(rewriter, typeConverter, cmpOp.getLhs());
          auto rhs =
              materializeOperand(rewriter, typeConverter, cmpOp.getRhs());
          auto targetResType =
              typeConverter->convertType(cmpOp.getResult().getType());
          if (!mask) {
            mask = buildAllTrueMask(rewriter, loc,
                                    cast<VectorType>(lhs.getType()));
          }
          auto predicate = cmpOp.getPredicate();
          StringRef prefStr = arith::stringifyCmpIPredicate(predicate);

          auto llvmPred = rewriter.getStringAttr(prefStr);
          auto intrOp = rewriter.create<vp::VPIntrICmpOp>(loc, targetResType,
                                                          lhs, rhs, mask, evl);
          intrOp->setAttr("predicate", llvmPred);
          loweredOp = intrOp.getOperation();
        })
        .Case([&](math::RsqrtOp rsqrtOp) {
          auto src = rsqrtOp.getOperand();
          src = materializeOperand(rewriter, typeConverter, src);
          auto targetResType =
              typeConverter->convertType(rsqrtOp.getResult().getType());
          auto targetVecType = cast<VectorType>(targetResType);
          if (enableRVV && enableMathEstimation) {
            auto [passthruMerged, policy] = buildRVVPassthru(
                rewriter, loc, evl, mask, passthru, maskedoff, targetResType);
            if (!mask) {
              auto intrOp = rewriter.create<vp::RVVIntrFRsqrt7Op>(
                  loc, targetResType, passthruMerged, src, evlxlen);
              loweredOp = intrOp.getOperation();
            } else {
              auto policyValue = rewriter.create<LLVM::ConstantOp>(
                  loc, rewriter.getI32Type(),
                  rewriter.getI32IntegerAttr(policy));
              auto intrOp = rewriter.create<vp::RVVIntrFRsqrt7MaskedOp>(
                  loc, targetResType, passthruMerged, src, mask, evlxlen,
                  policyValue);
              loweredOp = intrOp.getOperation();
            }
            doMerge = false; // Already merged in the intrinsic.
          } else {
            if (!mask) {
              mask = buildAllTrueMask(rewriter, loc, targetVecType);
            }
            auto sqrtIntrName = "llvm.vp.sqrt";
            auto sqrtOp = rewriter.create<LLVM::CallIntrinsicOp>(
                loc, targetResType, rewriter.getStringAttr(sqrtIntrName),
                ValueRange{src, mask, evl});
            auto elementType = targetVecType.getElementType();
            auto one = rewriter.create<LLVM::ConstantOp>(
                loc, elementType, rewriter.getFloatAttr(elementType, 1.0));
            auto splatIntrName =
                rewriter.getStringAttr("llvm.experimental.vp.splat");
            auto oneSplat = rewriter.create<LLVM::CallIntrinsicOp>(
                loc, targetResType, splatIntrName, ValueRange{one, mask, evl});
            auto divOp = rewriter.create<LLVM::VPFDivOp>(
                loc, targetResType, oneSplat.getResult(0), sqrtOp->getResult(0),
                mask, evl);
            loweredOp = divOp.getOperation();
          }
        })
        .Case([&](math::ExpOp expOp) {
          // TODO: There are no usable math libraries for dynamic vector,
          // fallback to the scalar version.
          auto src = expOp.getOperand();
          src = materializeOperand(rewriter, typeConverter, src);
          auto newOp = rewriter.create<math::ExpOp>(loc, src);
          loweredOp = newOp.getOperation();
        })
        .Default([&](Operation *) {
          loweredOp = rewriter.notifyMatchFailure(op, "unsupported operation");
        });

    if (failed(loweredOp)) {
      return failure();
    }

    if (doMerge && passthru && (*loweredOp)->getNumResults() != 0) {
      Value result = (*loweredOp)->getResult(0);
      if (!maskedoff) {
        // No maskedoff value provided, only merge passthru.
        result = rewriter.create<LLVM::VPMergeMinOp>(
            loc, result.getType(), mask, result, passthru, evl);
      } else {
        // Also merge maskedoff value into the result where the mask is false.
        auto maskMerged = rewriter.create<LLVM::VPSelectMinOp>(
            loc, result.getType(), mask, result, maskedoff, evl);
        // Create a mask with all true values to merge the tail from passthru.
        auto vectorType = cast<VectorType>(passthru.getType());
        auto maskType =
            vectorType.cloneWith(std::nullopt, rewriter.getI1Type());
        mask = rewriter.create<LLVM::ConstantOp>(
            loc, maskType, DenseIntElementsAttr::get(maskType, true));
        result = rewriter.create<LLVM::VPMergeMinOp>(
            loc, result.getType(), mask, maskMerged, passthru, evl);
      }
    }

    rewriter.replaceOp(op, *loweredOp);
    return success();
  }
};

struct GetVLOpLowering : ConvertOpToLLVMPattern<vp::GetVLOp> {
  GetVLOpLowering(LLVMTypeConverter &converter, bool enableRVV)
      : ConvertOpToLLVMPattern<vp::GetVLOp>(converter), enableRVV(enableRVV) {}

  LogicalResult
  matchAndRewrite(vp::GetVLOp op, vp::GetVLOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto typeConverter = getTypeConverter();

    unsigned vlBitwidth;
    if (enableRVV) {
      /// RVV intrinsics only support xlen-bit VL.
      vlBitwidth = typeConverter->getIndexType().getIntOrFloatBitWidth();
    } else {
      /// VP intrinsics support 32-bit VL.
      vlBitwidth = 32;
    }

    auto cnt = adaptor.getCnt();
    auto resultBitwidth = cnt.getType().getIntOrFloatBitWidth();
    auto resultType = cnt.getType();

    auto vlType = rewriter.getIntegerType(vlBitwidth);
    if (resultBitwidth > vlBitwidth) {
      cnt = rewriter.create<LLVM::TruncOp>(loc, vlType, cnt);
    } else if (resultBitwidth < vlBitwidth) {
      cnt = rewriter.create<LLVM::ZExtOp>(loc, vlType, cnt);
    }

    auto vf = adaptor.getVf();
    auto computeType = adaptor.getComputeType();
    auto scalable = adaptor.getScalable();

    computeType = typeConverter->convertType(computeType);

    if (!computeType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported compute type for GetVLOp");
    }

    auto sew = computeType.getIntOrFloatBitWidth();
    auto sewBytes = llvm::divideCeil(sew, 8);

    Value vl;

    if (enableRVV) {
      // Compute the LMUL adn SEW values.
      // https://llvm.org/docs/RISCV/RISCVVectorExtension.html#mapping-to-llvm-ir-types
      // LMUL = 64 / (VF * SEWbits) = 8 / (VF * SEWBytes)
      // SEW = log2(SEWbits)
      unsigned sew = std::log2(sewBytes);
      auto totalBytes = vf * sewBytes;
      unsigned lmul;
      if (totalBytes < 8) {
        auto factor = 8 / totalBytes;
        lmul = 8 - std::log2(factor);
      } else {
        auto factor = totalBytes / 8;
        lmul = std::log2(factor);
      }

      Value lmulValue = rewriter.create<LLVM::ConstantOp>(
          loc, vlType, rewriter.getIntegerAttr(vlType, lmul));
      Value sewValue = rewriter.create<LLVM::ConstantOp>(
          loc, vlType, rewriter.getIntegerAttr(vlType, sew));
      vl = rewriter.create<vp::RVVIntrSetVliOp>(loc, vlType, cnt, sewValue,
                                                lmulValue);
    } else {
      Value vfValue = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(sewBytes * vf));
      Value scalableValue = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI1Type(), rewriter.getBoolAttr(scalable));
      vl =
          rewriter
              .create<LLVM::CallIntrinsicOp>(
                  loc, rewriter.getI32Type(),
                  rewriter.getStringAttr("llvm.experimental.get.vector.length"),
                  ValueRange({cnt, vfValue, scalableValue}))
              ->getResult(0);
    }

    if (resultBitwidth > vlBitwidth) {
      vl = rewriter.create<LLVM::ZExtOp>(loc, resultType, vl);
    } else if (resultBitwidth < vlBitwidth) {
      vl = rewriter.create<LLVM::TruncOp>(loc, resultType, vl);
    }

    rewriter.replaceOp(op, vl);
    return success();
  }

private:
  bool enableRVV;
};

} // namespace

namespace mlir {
namespace vp {

void populateVPToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns,
                                        bool enableRVV,
                                        bool enableMathEstimation) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<PredicateOpLowering>(converter, enableRVV, enableMathEstimation);
  patterns.add<GetVLOpLowering>(converter, enableRVV);
}

void configureVPToLLVMConversionLegality(LLVMConversionTarget &target) {
  target.addIllegalOp<PredicateOp, GetVLOp, LoadOp, StoreOp, GatherOp,
                      ScatterOp>();
  target.addLegalOp<VPIntrFCmpOp, VPIntrICmpOp>();
  target.addLegalOp<RVVIntrSetVliOp, RVVIntrSetVliMaxOp, RVVIntrVidOp,
                    RVVIntrVidMaskedOp, RVVIntrVlmOp, RVVIntrVsmOp,
                    RVVIntrFRsqrt7Op, RVVIntrFRsqrt7MaskedOp>();
  // Later convert to llvm/libm
  target.addLegalOp<math::ExpOp>();
}

void ConvertVPToLLVMPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp moduleOp = getOperation();

  LowerToLLVMOptions options(ctx);
  if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout) {
    options.overrideIndexBitwidth(indexBitwidth);
  }
  LLVMTypeConverter typeConverter(ctx, options);

  LLVMConversionTarget target(*ctx);
  configureVPToLLVMConversionLegality(target);

  RewritePatternSet patterns(ctx);
  populateVPToLLVMConversionPatterns(typeConverter, patterns, enableRVV,
                                     enableMathEstimation);
  vp::PredicateOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

void ConvertVPToLLVMPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect, vp::VPDialect>();
}

void registerConvertVPToLLVMPass() { PassRegistration<ConvertVPToLLVMPass>(); }

} // namespace vp
} // namespace mlir
