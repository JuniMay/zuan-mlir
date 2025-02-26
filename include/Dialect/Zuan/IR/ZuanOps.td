//===- ZuanOps.td - Zuan Operations ------------------------*- tablegen -*-===//
//
// This file declares the operations for the Zuan dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_IR_ZUANOPS
#define ZUAN_IR_ZUANOPS

include "Zuan/IR/ZuanAttributes.td"
include "Zuan/IR/ZuanDialect.td"
include "Zuan/IR/ZuanTypes.td"
include "Zuan/IR/ZuanAttributes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dynamic and Yield operations
//===----------------------------------------------------------------------===//

def Zuan_YieldOp : Zuan_Op<"yield", [
  ReturnLike, Terminator, AttrSizedOperandSegments
]> {
  let summary = "Yield operation in Zuan dialect";
  let description = [{
    `zuan.yield` operation brings the computed value out of the dynamic region.
    The scalars will be yielded as results, and tiles will be stored back into
    the corresponding memref arguments.
  }];

  let arguments = (ins Variadic<Zuan_TileType>:$tiles,
                       Variadic<ZuanTileElement>:$scalars);
  let builders = [
    OpBuilder<(ins), [{ /* do nothing */ }]>
  ];
  let assemblyFormat = [{
    `(` ($tiles^ `:` type($tiles))? `)` `,`
    `(` ($scalars^ `:` type($scalars))? `)` attr-dict
  }];
  let hasVerifier = 1;
}

def Zuan_DynamicOp : Zuan_Op<"dynamic", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DestinationStyleOpInterface,
  SingleBlockImplicitTerminator<"zuan::YieldOp">,
  RecursiveMemoryEffects
]> {
  let summary = "A region that contains size-agnostic tile operations.";
  let description = [{
    The `zuan.dynamic` operation is a region that contains size-agnostic tile
    operations. MemRefs that will be written should be provided as dps inits.

    The memory writing semantics are similar to `vector.transfer_write`. The
    elements of tiles are written with 1:1 correspondence to the memref
    positions. Memory readings are not restricted and are not presented in the
    operands, `memref.subview` can be used to slicing or broadcasting the
    memrefs. Note that if there are aliasing memrefs between the inits (i.e., 
    writes) and the reads, the behavior is undefined.

    Scalars produced with reduction can be yielded as results. To lower this
    operation, the reductions and stores will be separate first. And then the
    leading dimensions will be expanded into loops. If any init memrefs differ
    in shape, the dynamic region will be duplicated with each of their
    dependencies. The shape differences are computed by a shape inference.

    The dynamic regions cannot be nested.
  }];

  let arguments = (ins Variadic<AnyMemRef>:$inits);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<ZuanTileElement>:$results);

  let assemblyFormat = [{
    (`(` $inits^ `:` type($inits) `)`)? $body attr-dict (`:` type($results)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitsMutable(); }
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes,
                   "ValueRange":$inits,
                   CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>">:$bodyBuilder)>,
    OpBuilder<(ins "ValueRange":$inits,
                   CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>">:$bodyBuilder)>
  ];

  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// High-level tile operations
//===----------------------------------------------------------------------===//

def Zuan_MatmulOp : Zuan_Op<"matmul", [
  Pure, InferTypeOpAdaptor,
  AllRanksMatch<["lhs", "rhs", "result"]>,
]> {
  let summary = "Matrix multiplication operation in Zuan dialect";
  let description = [{
    `zuan.matmul` operation computes the matrix multiplication of two tiles in
    the most minor two dimensions. The leading should be compatible.
  }];

  let arguments = (ins Zuan_TileType:$lhs, Zuan_TileType:$rhs);
  let results = (outs Zuan_TileType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))
  }];
}

def Zuan_MultiReductionOp : Zuan_Op<"multi_reduction", [
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["tile", "result"]>,
  OptionalTypesMatchWith<"result and the init have the same type",
                          "result", "init", "$_self", "TileType::isCompatible">,
]> {
  let summary = "Multi-dimensional reduction operation in Zuan dialect";
  let description = [{
    `zuan.multi_reduction` operation computes the reduction of the input tile
    along the specified dimensions. The reduced dimensions will be removed.

    The init tile should be shape-compatible with the result.
  }];

  let arguments = (ins Zuan_CombiningKindAttr:$kind,
                       Zuan_TileType:$tile,
                       DenseI64ArrayAttr:$dims,
                       Optional<Zuan_TileType>:$init);
  let results = (outs Zuan_TileType:$result);

  let assemblyFormat = [{
    $kind $tile $dims (`,` $init^)? attr-dict 
    `:` qualified(type($tile)) `,` qualified(type($init))
  }];
}

//===----------------------------------------------------------------------===//
// General Operations
//===----------------------------------------------------------------------===//

def Zuan_LoadOp : Zuan_Op<"load", [
  InferTypeOpAdaptor,
  AllShapesMatch<["base", "result"]>,
  AllElementTypesMatch<["base", "result"]>
]> {
  let summary = "Load a tile from a memref";
  let description = [{
    `zuan.load` operation reads a tile from a memref. The shape of the tile
    is the same as the memref.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{ $base attr-dict `:` type($base) }];
}

def Zuan_SplatOp : Zuan_Op<"splat", [Pure, InferTypeOpAdaptor]> {
  let summary = "Extend the leading dimensions of a vector or scalar";
  let description = [{
    `zuan.splat` extends the leading dims of a `zuan.tile`, or broadcasts a
    scalar to a `zuan.tile` type.
  }];

  let arguments = (ins AnyTypeOf<[ZuanTileElement, Zuan_TileType]>:$value,
                       Variadic<Index>:$dims,
                       DenseI64ArrayAttr:$staticDims);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{
    $value custom<DynamicIndexList>($dims, $staticDims) attr-dict `:` type($value)
  }];
  
  let builders = [
    OpBuilder<(ins "Value":$value, "ArrayRef<int64_t>":$dims)>,
    OpBuilder<(ins "Value":$value, "ArrayRef<OpFoldResult>":$dims)>
  ];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedDims();
  }];
}

def Zuan_OuterOp : Zuan_Op<"outer", [Pure, InferTypeOpAdaptor]> {
  let summary = "Outer operation";
  let description = [{
    `zuan.outer` operation computes the outer product/sum/... of two tiles.
    The result equals to broadcasting and applying the element-wise operation
    to the two tiles.

    The leading dims of two tiles should be compatible. If the rank of one
    of the tiles is less than the other by 1, this equals to a vector-scalar
    operation at the most minor dimension:

    ```
    // Semantic: [m, n] op [m, k] -> [m, n, k]
    %res = zuan.outer <op> %a, %b : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    ```

    ```
    // Semantic: [m, n] op [m, _] -> [m, n]
    %res = zuan.outer <op> %a, %b : !zuan.tile<?x?xf32>, !zuan.tile<?xf32>
    ```
  }];

  let arguments = (ins Zuan_CombiningKindAttr:$kind,
                       Zuan_TileType:$lhs,
                       Zuan_TileType:$rhs);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{
    $kind $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))
  }];
}

def Zuan_CastOp : Zuan_Op<"cast", [
  Pure,
  AllShapesMatch<["tile", "result"]>
]> {
  let summary = "Cast all elements of a tile to another type";
  let description = [{
    `zuan.cast` operation casts all elements of a tile to another type. The cast
    ops in arith dialect only support `vector` and `tensor` types, other types
    are regarded as cast-incompatible. So here is a dedicated cast op for
    `zuan.tile`, implemented with `CastKind` for simplicity.
  }];
  let arguments = (ins Zuan_CastKindAttr:$kind, Zuan_TileType:$tile);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{
    $kind $tile attr-dict `:` qualified(type($tile)) `to` qualified(type($result))
  }];
}

def Zuan_StepOp : Zuan_Op<"step", [
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["result", "start"]>
]> {
  let summary = "Step operation";
  let description = [{
    `zuan.step` operation generates a tile with the specified shape and fills
    the elements with increasing values on a certain dimension from the given
    start value.
  }];
  let arguments = (ins AnySignlessIntegerOrIndex:$start,
                       IndexAttr:$dim,
                       Variadic<Index>:$sizes,
                       DenseI64ArrayAttr:$staticSizes);
  let results = (outs ZuanTileOfType<[AnySignlessIntegerOrIndex]>:$result);
  let assemblyFormat = [{
    $start `,` $dim `,` custom<DynamicIndexList>($sizes, $staticSizes) attr-dict
    `:` type($start)
  }];

  let builders = [
    OpBuilder<(ins "Value":$start, "int64_t":$dim, "ArrayRef<OpFoldResult>":$sizes)>,
    OpBuilder<(ins "Value":$start, "int64_t":$dim, "ArrayRef<int64_t>":$staticSizes)>,
  ];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

#endif // ZUAN_IR_ZUANOPS
