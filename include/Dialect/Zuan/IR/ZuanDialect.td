//===- ZuanDialect.td - Zuan Dialect -----------------------*- tablegen -*-===//
//
// This file declares the Zuan dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_IR_ZUANDIALECT
#define ZUAN_IR_ZUANDIALECT

include "mlir/IR/OpBase.td"

def Zuan_Dialect : Dialect {
  let name = "zuan";
  let cppNamespace = "::mlir::zuan";
  let summary = "Zuan dialect";
  let description = [{
    Zuan dialect is a high level representation of high dimensional dynamic
    vector operations.
  }];

  let extraClassDeclaration = [{
    void registerTypes();  
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base class of Zuan dialect operations.
class Zuan_Op<string mnemonic, list<Trait> traits = []> :
    Op<Zuan_Dialect, mnemonic, traits> {}

class Zuan_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Zuan_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // ZUAN_IR_ZUANDIALECT
