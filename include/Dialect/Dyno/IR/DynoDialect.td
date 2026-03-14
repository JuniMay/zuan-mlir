//===- DynoDialect.td - Dyno Dialect -----------------------*- tablegen -*-===//
//
// This file declares the Dyno dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_IR_DYNODIALECT
#define DYNO_IR_DYNODIALECT

include "mlir/IR/OpBase.td"

def Dyno_Dialect : Dialect {
  let name = "dyno";
  let cppNamespace = "::mlir::dyno";
  let summary = "Dyno dialect";
  let description = [{
    Dyno dialect is a high level representation of high dimensional dynamic
    vector operations.
  }];

  let extraClassDeclaration = [{
    void registerTypes();  
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base class of Dyno dialect operations.
class Dyno_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dyno_Dialect, mnemonic, traits> {}

class Dyno_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Dyno_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // DYNO_IR_DYNODIALECT
