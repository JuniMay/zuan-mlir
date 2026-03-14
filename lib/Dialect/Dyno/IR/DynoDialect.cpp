//===- DynoDialect.cpp - Dyno dialect ---------------------------*- C++ -*-===//
//
// This file implements the Dyno dialect.
//
//===----------------------------------------------------------------------===//

#include "Dyno/IR/Dyno.h"

#include "Dyno/IR/DynoEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Dyno/IR/DynoAttributes.cpp.inc"
#include "Dyno/IR/DynoOpsDialect.cpp.inc"

namespace mlir {
namespace dyno {

void DynoDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dyno/IR/DynoAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dyno/IR/DynoOps.cpp.inc"
      >();
  registerTypes();
}

} // namespace dyno
} // namespace mlir
