//===- VPDialect.cpp - VP Dialect Implementations ---------------*- C++ -*-===//
//
// This file implements the initialization of VP dialect.
//
//===----------------------------------------------------------------------===//

#include "VP/IR/VP.h"

using namespace mlir;
using namespace mlir::vp;

#include "VP/IR/VPOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "VP/IR/VPOps.cpp.inc"

namespace mlir {
namespace vp {

void VPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VP/IR/VPOps.cpp.inc"
      >();
}

} // namespace vp
} // namespace mlir
