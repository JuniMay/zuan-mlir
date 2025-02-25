//===- VPOps.cpp - VP Operation Implementations -----------------*- C++ -*-===//
//
// This file implements the operations of the VP dialect.
//
//===----------------------------------------------------------------------===//

#include "VP/IR/VP.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

namespace mlir {
namespace vp {

void PredicateOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    Value evl, Operation *predicatedOp,
    function_ref<void(OpBuilder &, Operation *)> bodyBuilder, Value mask,
    Value passthru, Value maskedoff) {

  assert(bodyBuilder &&
         "builder callback for the predicated region must be present");

  result.addOperands(evl);
  if (mask) {
    result.addOperands(mask);
  }
  if (passthru) {
    result.addOperands(passthru);
  }
  if (maskedoff) {
    result.addOperands(maskedoff);
  }

  llvm::copy(
      ArrayRef<int32_t>(
          {1, (mask ? 1 : 0), (passthru ? 1 : 0), (maskedoff ? 1 : 0)}),
      result.getOrAddProperties<Properties>().operandSegmentSizes.begin());

  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  bodyBuilder(builder, predicatedOp);

  result.addTypes(resultTypes);
}

void PredicateOp::build(
    OpBuilder &builder, OperationState &result, Value evl,
    Operation *predicatedOp,
    function_ref<void(OpBuilder &, Operation *)> bodyBuilder, Value mask,
    Value passthru, Value maskedoff) {
  build(builder, result, {}, evl, predicatedOp, bodyBuilder, mask, passthru,
        maskedoff);
}

LogicalResult PredicateOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  // TODO: We can check if evl equals to vlmax and mask is all true, then we can
  // replace the predicate with the predicated operation inside.
  return failure();
}

struct ElideEmptyPredicateOp : OpRewritePattern<PredicateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PredicateOp predOp,
                                PatternRewriter &rewriter) const override {
    Block *block = &predOp.getBody().front();
    if (block->getOperations().size() >= 2) {
      // There is another operation apart from the terminator.
      return failure();
    }
    auto terminator = cast<vector::YieldOp>(block->front());
    if (terminator->getNumOperands() == 0) {
      rewriter.eraseOp(predOp);
    } else {
      rewriter.replaceOp(predOp, terminator.getOperands());
    }
    return success();
  }
};

void PredicateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<ElideEmptyPredicateOp>(context);
}

void createPredicateOpRegion(OpBuilder &builder, Operation *predicatedOp) {
  assert(predicatedOp->getBlock() &&
         "PredicatedOp must be inserted into a block");
  Block *insBlock = builder.getInsertionBlock();
  // Move the predicated operation into the region of the predicate.
  insBlock->getOperations().splice(insBlock->begin(),
                                   predicatedOp->getBlock()->getOperations(),
                                   predicatedOp);
  builder.create<vector::YieldOp>(predicatedOp->getLoc(),
                                  predicatedOp->getResults());
}

Operation *predicateOperation(OpBuilder &builder, Operation *predicatedOp,
                              Value evl, Value mask, Value passthru,
                              Value maskedoff) {
  auto predication = builder.create<PredicateOp>(
      predicatedOp->getLoc(), predicatedOp->getResultTypes(), evl, predicatedOp,
      createPredicateOpRegion, mask, passthru, maskedoff);
  return predication;
}

LogicalResult PredicateOp::verify() {
  Block &block = getBody().front();
  if (block.empty()) {
    return emitOpError("expects a terminator within the predication block");
  }

  if (block.getOperations().size() > 2) {
    return emitOpError("expects only one operation to be predicated");
  }

  auto terminator = dyn_cast<vector::YieldOp>(block.getTerminator());
  if (!terminator) {
    return emitOpError("expects the terminator to be a vector.yield");
  }

  if (terminator->getNumOperands() != getNumResults()) {
    return emitOpError("expects the number of operands in the terminator to "
                       "match the number of results in the predication");
  }

  if (!llvm::equal(terminator->getOperandTypes(), getResultTypes())) {
    return emitOpError("expects the types of the operands in the terminator to "
                       "match the types of the results in the predication");
  }

  return success();
}

} // namespace vp
} // namespace mlir
