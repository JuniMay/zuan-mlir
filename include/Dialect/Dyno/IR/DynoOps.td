//===- DynoOps.td - Dyno Operations ------------------------*- tablegen -*-===//
//
// This file declares the operations for the Dyno dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_IR_DYNOOPS
#define DYNO_IR_DYNOOPS

include "Dyno/IR/DynoAttributes.td"
include "Dyno/IR/DynoDialect.td"
include "Dyno/IR/DynoTypes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// General region operations
//===----------------------------------------------------------------------===//

def Dyno_MaskYieldOp : Dyno_Op<"mask_yield", [
  Pure, ReturnLike, Terminator, HasParent<"dyno::MaskOp">,
]> {
  let summary = "Terminator for the mask region";
  let description = [{
    `dyno.mask_yield` operation is the terminator of the mask region. Only tile
    type can be yielded, including 0-D tiles.
  }];
  let arguments = (ins Variadic<Dyno_TileType>:$tiles);
  let builders = [
    OpBuilder<(ins), [{ /* do nothing */ }]>
  ];
  let assemblyFormat = [{
    ($tiles^ `:` type($tiles))? attr-dict
  }];
}

def Dyno_MaskOp : Dyno_Op<"mask", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"dyno::MaskYieldOp">
]> {
  let summary = "A mask region for conditional execution";
  let description = [{
    `dyno.mask` operation is a region that contains conditional tile operations.
    Only the elements whose corresponding mask is true will be computed. An
    optional passthru tile `maskedoff` can be provided to fill the masked-off
    elements.

    If no passthru is provided, using the masked-off values is undefined and
    the result can be arbitrary.

    The passthru and mask are up to operations to decide their semantics. The
    mask regions cannot be nested.

    At most one operation can be inside the mask operation, and the yielded
    value must have tile type.
  }];
  let arguments = (ins DynoTileOfType<[I1]>:$mask,
                       Optional<Dyno_TileType>:$maskedoff);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<Dyno_TileType>:$results);

  let assemblyFormat = [{
    $mask `:` type($mask)
    (`,` $maskedoff^ `:` type($maskedoff))? $body attr-dict
    (`:` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes,
                   "Value":$mask,
                   CArg<"function_ref<void(OpBuilder &, Location)>">:$bodyBuilder,
                   CArg<"Value", "nullptr">:$maskedoff)>,
    OpBuilder<(ins "Value":$mask,
                   CArg<"function_ref<void(OpBuilder &, Location)>">:$bodyBuilder,
                   CArg<"Value", "nullptr">:$maskedoff)>
  ];
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    Operation *getMaskedOp();
  }];
}

//===----------------------------------------------------------------------===//
// Reduction-like tile operations
//===----------------------------------------------------------------------===//

 def Dyno_ReductionOp : Dyno_Op<"reduction", [
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["tile", "result"]>,
  OptionalTypesMatchWith<"result and the init have the same type",
                          "result", "init", "$_self", "TileType::isCompatible">,
]> {
  let summary = "Multi-dimensional reduction operation in Dyno dialect";
  let description = [{
    `dyno.reduction` operation computes the reduction of the input tile along
    the specified dimensions. The reduced dimensions will be removed. The init
    tile should be shape-compatible with the result.
  }];

  let arguments = (ins Dyno_CombiningKindAttr:$kind,
                       Dyno_TileType:$tile,
                       DenseI64ArrayAttr:$dims,
                       Optional<Dyno_TileType>:$init);
  let results = (outs Dyno_TileType:$result);

  let assemblyFormat = [{
    $kind $tile $dims (`,` $init^)? attr-dict
    `:` qualified(type($tile)) (`,` qualified(type($init))^)?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// General tile operations
//===----------------------------------------------------------------------===//

def Dyno_LoadOp : Dyno_Op<"load", [
  InferTypeOpAdaptor,
  AllShapesMatch<["base", "result"]>,
  AllElementTypesMatch<["base", "result"]>
]> {
  let summary = "Load a tile from a memref";
  let description = [{
    `dyno.load` operation reads a tile from a memref. The shape of the tile
    is the same as the memref.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base);
  let results = (outs Dyno_TileType:$result);
  let assemblyFormat = [{ $base attr-dict `:` type($base) }];
  let hasVerifier = 1;
}

def Dyno_StoreOp : Dyno_Op<"store", [
  AllShapesMatch<["value", "base"]>, AllElementTypesMatch<["value", "base"]>
]> {
  let summary = "Store a tile to a memref";
  let description = [{
    `dyno.store` operation writes a tile to a memref. The shape of the tile
    should be the same as the memref.
  }];

  let arguments = (ins Dyno_TileType:$value,
                       Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base);
  let assemblyFormat = [{
    $value `,` $base attr-dict `:` type($value) `,` type($base)
  }];
}

def Dyno_DimOp : Dyno_Op<"dim", [Pure]> {
  let summary = "Query the runtime extent of a tile dimension";
  let description = [{
    `dyno.dim` materializes the extent of a tile value at a constant dimension
    index. Static dimensions fold to constants. Dynamic dimensions are forwarded
    through the producer semantics when possible.
  }];

  let arguments = (ins Dyno_TileType:$tile, I64Attr:$dim);
  let results = (outs Index:$result);

  let builders = [
    OpBuilder<(ins "Value":$tile, "int64_t":$dim)>
  ];

  let assemblyFormat = [{
    $tile `,` $dim attr-dict `:` qualified(type($tile))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Dyno_ExtractOp : Dyno_Op<"extract", [Pure, InferTypeOpAdaptor]> {
  let summary = "Extract the scalar element from a rank-0 tile";
  let description = [{
    `dyno.extract` converts a rank-0 tile into its scalar element.
  }];

  let arguments = (ins DynoTileOfRank<[0]>:$tile);
  let results = (outs DynoTileElement:$result);

  let assemblyFormat = [{
    $tile attr-dict `:` qualified(type($tile))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def Dyno_SplatOp : Dyno_Op<"splat", [
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["value", "result"]>
]> {
  let summary = "Extend the leading dimensions of a vector or scalar";
  let description = [{
    `dyno.splat` extends the leading dims of a `dyno.tile`, or broadcasts a
    scalar to a `dyno.tile` type.
  }];

  let arguments = (ins AnyTypeOf<[DynoTileElement, Dyno_TileType]>:$value,
                       Variadic<Index>:$dims,
                       DenseI64ArrayAttr:$staticDims);
  let results = (outs Dyno_TileType:$result);
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
  let hasVerifier = 1;
}

def Dyno_CastOp : Dyno_Op<"cast", [
  Pure, AllShapesMatch<["tile", "result"]>
]> {
  let summary = "Cast all elements of a tile to another type";
  let description = [{
    `dyno.cast` operation casts all elements of a tile to another type. The
    cast ops in arith dialect only support `vector` and `tensor` types, other
    types are regarded as cast-incompatible. So here is a dedicated cast op for
    `dyno.tile`, implemented with `CastKind` for simplicity.
  }];
  let arguments = (ins Dyno_CastKindAttr:$kind, Dyno_TileType:$tile);
  let results = (outs Dyno_TileType:$result);
  let assemblyFormat = [{
    $kind $tile attr-dict `:` qualified(type($tile)) `to` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Dyno_StepOp : Dyno_Op<"step", [
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["result", "start"]>
]> {
  let summary = "Step operation";
  let description = [{
    `dyno.step` operation generates a tile with the specified shape and fills
    the elements with increasing values on a certain dimension from the given
    start value.
  }];
  let arguments = (ins AnySignlessIntegerOrIndex:$start,
                       IndexAttr:$dim,
                       Variadic<Index>:$sizes,
                       DenseI64ArrayAttr:$staticSizes);
  let results = (outs DynoTileOfType<[AnySignlessIntegerOrIndex]>:$result);
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
  let hasVerifier = 1;
}

def Dyno_SelectOp : Dyno_Op<"select", [
  Pure, AllShapesMatch<["cond", "lhs", "rhs", "result"]>,
  AllTypesMatch<["lhs", "rhs", "result"]>
]> {
  let summary = "Select operation";
  let description = [{
    `dyno.select` operation selects elements from the two tiles based on the
    condition tile. The shape of the condition tile should be the same as the
    other two tiles.
  }];
  let arguments = (ins Dyno_TileType:$cond,
                       Dyno_TileType:$lhs,
                       Dyno_TileType:$rhs);
  let results = (outs Dyno_TileType:$result);
  let assemblyFormat = [{
    $cond `,` $lhs `,` $rhs attr-dict `:` type($cond) `,` type($lhs)
  }];
  let hasVerifier = 1;
}

def Dyno_GatherOp : Dyno_Op<"gather", [
  InferTypeOpAdaptor, AllElementTypesMatch<["base", "result"]>
]> {
  let summary = "Memory gather operation";
  let description = [{
    Given the source memref and indices, `dyno.gather` loads the values from
    the source with the following behavior:

    ```
    res[i_0, i_1, ... i_n] = source[indices_0[i_0, i_1, ... i_n],
                                    indices_1[i_0, i_1, ... i_n],
                                    ...
                                    indices_k[i_0, i_1, ... i_n]]
    ```

    The indices are expected to be the same shape as the result.
  }];

  let arguments = (ins Arg<AnyMemRef, "The memref to gather from", [MemRead]>:$base,
                       Variadic<DynoTileOfType<[Index]>>:$indices);
  let results = (outs Dyno_TileType:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base) (`,` type($indices)^)?
  }];
  let hasVerifier = 1;
}

def Dyno_ScatterOp : Dyno_Op<"scatter", [
  AllElementTypesMatch<["value", "base"]>
]> {
  let summary = "Memory scatter operation";
  let description = [{
    Given the source tile, indices, and the destination memref, `dyno.scatter`
    writes the values to the destination with the following behavior:

    ```
    dest[indices_0[i_0, i_1, ... i_n],
         indices_1[i_0, i_1, ... i_n],
         ...
         indices_k[i_0, i_1, ... i_n]] = source[i_0, i_1, ... i_n]
    ```

    The indices are expected to be the same shape as the value.
  }];

  let arguments = (ins Dyno_TileType:$value,
                       Arg<AnyMemRef, "The memref to scatter to", [MemWrite]>:$base,
                       Variadic<DynoTileOfType<[Index]>>:$indices);
  let assemblyFormat = [{
    $value `,` $base `[` $indices `]` attr-dict `:` type($value) `,` type($base) (`,` type($indices)^)?
  }];
  let hasVerifier = 1;
}

#endif // DYNO_IR_DYNOOPS
