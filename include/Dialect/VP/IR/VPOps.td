//===- VPOps.td - VP Operations ----------------------------*- tablegen -*-===//
//
// This file declares the Vector Predication operations.
//
//===----------------------------------------------------------------------===//

#ifndef VP_IR_VPOPS
#define VP_IR_VPOPS

include "VPDialect.td"

include "mlir/IR/CommonTypeConstraints.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def VPScalar : AnyTypeOf<[
  F16, F32, F64, F128, BF16, I1, I8, I16, I32, I64, I128, Index]>;

def VPVector : VectorOfLengthAndType<
  [64, 32, 16, 8, 4, 2, 1], [VPScalar]>;

def VPIntVector : VectorOfLengthAndType<
  [64, 32, 16, 8, 4, 2, 1], [I1, I8, I16, I32, I64, I128]>;

def VPIndexVector : VectorOfLengthAndType<
  [64, 32, 16, 8, 4, 2, 1], [Index]>;

def VPMask : VectorOfLengthAndType<
  [64, 32, 16, 8, 4, 2, 1], [I1]>;

def VP_PredicateOp : VP_Op<"predicate", [
  SingleBlockImplicitTerminator<"vector::YieldOp">,
  RecursiveMemoryEffects, NoRegionArguments, AttrSizedOperandSegments
]> {
  let summary = "VP predicate operation";
  let description = [{
    VP predicate operation is a wrapper around arith or vector dialect
    operations. It specifies the explicit vector length, optional mask and
    optional passthru values.
  }];
  
  let arguments = (ins AnySignlessIntegerOrIndex:$evl,
                       Optional<VPMask>:$mask,
                       Optional<VPVector>:$passthru,
                       Optional<VPVector>:$maskedoff);
  let results = (outs Variadic<AnyType>:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $evl `:` type($evl)
    `,` `mask` `=` ($mask^ `:` type($mask)):(`none`)? 
    `,` `passthru` `=` ($passthru^ `:` type($passthru)):(`none`)?
    `,` `maskedoff` `=` ($maskedoff^ `:` type($maskedoff)):(`none`)?
    $body attr-dict (`:` type($result)^)?
  }];

  let builders = [
    // Builder with EVL, mask, passthru, maskedoff, and operation-to-be-predicated.
    OpBuilder<(ins "Value":$evl,
                   "Operation *":$predicatedOp,
                   CArg<"function_ref<void(OpBuilder &, Operation *)>">:$bodyBuilder,
                   CArg<"Value", "nullptr">:$mask,
                   CArg<"Value", "nullptr">:$passthru,
                   CArg<"Value", "nullptr">:$maskedoff)>,

    // Builder with result types, EVL, mask, passthru, maskedoff, and operation-to-be-predicated.
    OpBuilder<(ins "TypeRange":$resultTypes,
                   "Value":$evl,
                   "Operation *":$predicatedOp,
                   CArg<"function_ref<void(OpBuilder &, Operation *)>">:$bodyBuilder,
                   CArg<"Value", "nullptr">:$mask,
                   CArg<"Value", "nullptr">:$passthru,
                   CArg<"Value", "nullptr">:$maskedoff)>
  ];

  let skipDefaultBuilders = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def VP_GetVLOp : VP_Op<"getvl", [
  AllTypesMatch<["cnt", "result"]>
]> {
  let summary = "Get Vector Length Operation.";
  let description = [{
    The MLIR version of `llvm.experimental.get.vector.lenth` intrinsic.

    See: https://llvm.org/docs/LangRef.html#llvm-experimental-get-vector-length-intrinsic
  }];
  
  let arguments = (ins AnySignlessIntegerOrIndex:$cnt,
                       I32Attr:$vf,
                       TypeAttr:$computeType,
                       I1Attr:$scalable);
  let results = (outs AnySignlessIntegerOrIndex:$result);

  let assemblyFormat = [{
    $cnt `,` $vf `x` $computeType `,` $scalable attr-dict `:` type($cnt)
  }];
}

def VP_LoadOp : VP_Op<"load", []> {
  let summary = "reads an 1-D slice of memory into an 1-D vector";
  let description = [{
    This is a replacement of `vector.load` operation in the vector dialect,
    which does not support strided load at the most minor dimension.

    Different from vector dialect, this operation only suppotr 1-D vector
    loading from n-D memory.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
                       Variadic<Index>:$indices);
  let results = (outs VPVector:$result);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }

    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];

  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base) `,` type($result)
  }];
}

def VP_StoreOp : VP_Op<"store", []> {
  let summary = "writes an 1-D vector to an 1-D slice of memory";
  let description = [{
    This is a replacement of `vector.store` operation in the vector dialect,
    which does not support strided store at the most minor dimension.

    Different from vector dialect, this operation only suppotr 1-D vector
    storing to n-D memory.
  }];

  let arguments = (ins VPVector:$value,
                       Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
                       Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }

    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getValue().getType());
    }
  }];

  let assemblyFormat = [{
    $value `,` $base `[` $indices `]` attr-dict `:` type($value) `,` type($base)
  }];
}

def VP_GatherOp : VP_Op<"gather", []> {
  let summary = "reads an 1-D vector from memref with vector indices";
  let description = [{
    This operation support multiple indexing vectors to read from a N-D memref.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
                       Variadic<VPIndexVector>:$indices);
  let results = (outs VPVector:$result);

  let assemblyFormat = [{
    $base `[` $indices `:` type($indices) `]` attr-dict `:` type($base) `,` type($result)
  }];
}

def VP_ScatterOp : VP_Op<"scatter", []> {
  let summary = "writes an 1-D vector to memref with vector indices";
  let description = [{
    This operation support multiple indexing vectors to write to a N-D memref.
  }];

  let arguments = (ins VPVector:$value,
                       Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
                       Variadic<VPIndexVector>:$indices);

  let assemblyFormat = [{
    $value `,` $base `[` $indices `:` type($indices) `]` attr-dict `:` type($value) `,` type($base)
  }];
}

def VP_StepOp : VP_Op<"step", [Pure]> {
  let summary = "VP step operation";
  let description = [{
    `vp.step` is an alternative to `vector.step` which only supports index type.
  }];

  let results = (outs VPIntVector:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Target Specific Intrinsic Operations
//===----------------------------------------------------------------------===//

class VP_RVVIntrOp<string mnemonic, 
                list<int> overloadedResults = [],
                list<int> overloadedOperands = [],
                list<Trait> traits = [],
                int numResults = 1> :
  LLVM_IntrOpBase</*Dialect dialect=*/VP_Dialect, 
                  /*string opName=*/"intr.rvv." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/overloadedResults, 
                  /*list<int> overloadedOperands=*/overloadedOperands, 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/numResults>;

def VP_RVVIntrSetVliOp :
  VP_RVVIntrOp<"vsetvli", [0], [], [], 1>,
  Arguments<(ins AnyInteger, AnyInteger, AnyInteger)>;

def VP_RVVIntrSetVliMaxOp :
  VP_RVVIntrOp<"vsetvlimax", [0], [], [], 1>,
  Arguments<(ins AnyInteger, AnyInteger)>;

def VP_RVVIntrVidOp :
  VP_RVVIntrOp<"vid", [0], [1], [], 1>,
  Arguments<(ins VPVector, AnyInteger)>;

def VP_RVVIntrVidMaskedOp:
  VP_RVVIntrOp<"vid_mask", [0], [2], [], 1>,
  Arguments<(ins VPVector, VPMask, AnyInteger, AnyInteger)>;

def VP_RVVIntrVlmOp:
  VP_RVVIntrOp<"vlm", [0], [1], [], 1>,
  Arguments<(ins LLVM_AnyPointer, AnyInteger)>;

def VP_RVVIntrVsmOp:
  VP_RVVIntrOp<"vsm", [], [0, 2], [], 0>,
  Arguments<(ins VPMask, LLVM_AnyPointer, AnyInteger)>;

def VP_RVVIntrFRsqrt7Op:
  VP_RVVIntrOp<"vfrsqrt7", [0], [2], [], 1>,
  Arguments<(ins VPVector, VPVector, AnyInteger)>;

def VP_RVVIntrFRsqrt7MaskedOp:
  VP_RVVIntrOp<"vfrsqrt7_mask", [0], [3], [], 1>,
  Arguments<(ins VPVector, VPVector, VPMask, AnyInteger, AnyInteger)>;

//===----------------------------------------------------------------------===//
// Additional vp intrinsics
//===----------------------------------------------------------------------===//

def VP_VPIntrFCmpOp : VP_Op<"intr.fcmp", [Pure]> {
  let summary = "VP fcmp operation";
  let description = [{
    Workaround intrinsic for the vp.fcmp operation, which has a metadata in it,
    and cannot be directly lowered to LLVM dialect.
  }];

  let arguments = (ins VPVector:$lhs, VPVector:$rhs, VPMask:$mask, I32:$evl);
  let results = (outs VPMask:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask `,` $evl attr-dict `:` 
    type($lhs) `,` type($rhs) `,` type($mask) `,` type($evl) `,` type($result)
  }];
}

def VP_VPIntrICmpOp : VP_Op<"intr.icmp", [Pure]> {
  let summary = "VP icmp operation";
  let description = [{
    Workaround intrinsic for the vp.icmp operation, which has a metadata in it,
    and cannot be directly lowered to LLVM dialect.
  }];

  let arguments = (ins VPVector:$lhs, VPVector:$rhs, VPMask:$mask, I32:$evl);
  let results = (outs VPMask:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask `,` $evl attr-dict `:` 
    type($lhs) `,` type($rhs) `,` type($mask) `,` type($evl) `,` type($result)
  }];
}

#endif // VP_IR_VPOPS
