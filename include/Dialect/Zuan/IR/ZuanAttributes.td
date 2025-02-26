#ifndef ZUAN_IR_ZUANATTRIBUTES
#define ZUAN_IR_ZUANATTRIBUTES

include "Zuan/IR/ZuanDialect.td"

include "mlir/IR/EnumAttr.td"

// XXX: This is kept the same as vector dialect in purpose.
def COMBINING_KIND_ADD : I32EnumAttrCase<"ADD", 0, "add">;
def COMBINING_KIND_MUL : I32EnumAttrCase<"MUL", 1, "mul">;
def COMBINING_KIND_MINUI : I32EnumAttrCase<"MINUI", 2, "minui">;
def COMBINING_KIND_MINSI : I32EnumAttrCase<"MINSI", 3, "minsi">;
def COMBINING_KIND_MINNUMF : I32EnumAttrCase<"MINNUMF", 4, "minnumf">;
def COMBINING_KIND_MAXUI : I32EnumAttrCase<"MAXUI", 5, "maxui">;
def COMBINING_KIND_MAXSI : I32EnumAttrCase<"MAXSI", 6, "maxsi">;
def COMBINING_KIND_MAXNUMF : I32EnumAttrCase<"MAXNUMF", 7, "maxnumf">;
def COMBINING_KIND_AND : I32EnumAttrCase<"AND", 8, "and">;
def COMBINING_KIND_OR  : I32EnumAttrCase<"OR", 9, "or">;
def COMBINING_KIND_XOR : I32EnumAttrCase<"XOR", 10, "xor">;
def COMBINING_KIND_MINIMUMF : I32EnumAttrCase<"MINIMUMF", 11, "minimumf">;
def COMBINING_KIND_MAXIMUMF : I32EnumAttrCase<"MAXIMUMF", 12, "maximumf">;

def CombiningKind : I32EnumAttr<"CombiningKind", "Kind of outer operations",
  [
    COMBINING_KIND_ADD, COMBINING_KIND_MUL, COMBINING_KIND_MINUI,
    COMBINING_KIND_MINSI, COMBINING_KIND_MINNUMF, COMBINING_KIND_MAXUI,
    COMBINING_KIND_MAXSI, COMBINING_KIND_MAXNUMF, COMBINING_KIND_AND,
    COMBINING_KIND_OR, COMBINING_KIND_XOR,
    COMBINING_KIND_MAXIMUMF, COMBINING_KIND_MINIMUMF
  ]> {
  
  let cppNamespace = "::mlir::zuan";
  let genSpecializedAttr = 0;
}

def Zuan_CombiningKindAttr : EnumAttr<Zuan_Dialect, CombiningKind, "combining_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def CAST_KIND_EXTF : I32EnumAttrCase<"EXTF", 0, "extf">;
def CAST_KIND_EXTSI : I32EnumAttrCase<"EXTSI", 1, "extsi">;
def CAST_KIND_EXTUI : I32EnumAttrCase<"EXTUI", 2, "extui">;
def CAST_KIND_FPTOSI : I32EnumAttrCase<"FPTOSI", 3, "fptosi">;
def CAST_KIND_FPTOUI : I32EnumAttrCase<"FPTOUI", 4, "fptoui">;
def CAST_KIND_INDEXCAST : I32EnumAttrCase<"INDEXCAST", 5, "indexcast">;
def CAST_KIND_INDEXCASTUI : I32EnumAttrCase<"INDEXCASTUI", 6, "indexcastui">;
def CAST_KIND_SITOFP : I32EnumAttrCase<"SITOFP", 7, "sitofp">;
def CAST_KIND_UITOFP : I32EnumAttrCase<"UITOFP", 8, "uitofp">;
def CAST_KIND_TRUNCF : I32EnumAttrCase<"TRUNCF", 9, "truncf">;
def CAST_KIND_TRUNCI : I32EnumAttrCase<"TRUNCI", 10, "trunci">;
def CAST_KIND_BITCAST : I32EnumAttrCase<"BITCAST", 11, "bitcast">;

def CastKind : I32EnumAttr<"CastKind", "Kind of type conversion",
  [
    CAST_KIND_EXTF,
    CAST_KIND_EXTSI,
    CAST_KIND_EXTUI,
    CAST_KIND_FPTOSI,
    CAST_KIND_FPTOUI,
    CAST_KIND_INDEXCAST,
    CAST_KIND_INDEXCASTUI,
    CAST_KIND_SITOFP,
    CAST_KIND_UITOFP,
    CAST_KIND_TRUNCF,
    CAST_KIND_TRUNCI,
    CAST_KIND_BITCAST
  ]> {
  
  let cppNamespace = "::mlir::zuan";
  let genSpecializedAttr = 0;
}

def Zuan_CastKindAttr : EnumAttr<Zuan_Dialect, CastKind, "cast_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ZUAN_IR_ZUANATTRIBUTES
