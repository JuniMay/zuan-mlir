//===- ZuanDialect.cpp - Zuan dialect ---------------------------*- C++ -*-===//
//
// This file implements the Zuan dialect.
//
//===----------------------------------------------------------------------===//

#include "Zuan/IR/Zuan.h"

#include "Zuan/IR/ZuanEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Zuan/IR/ZuanAttributes.cpp.inc"
#include "Zuan/IR/ZuanOpsDialect.cpp.inc"

namespace mlir {
namespace zuan {

void ZuanDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Zuan/IR/ZuanAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Zuan/IR/ZuanOps.cpp.inc"
      >();
  registerTypes();
}

} // namespace zuan
} // namespace mlir
