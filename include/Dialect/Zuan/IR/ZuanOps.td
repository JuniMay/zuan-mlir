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
include "Zuan/Interfaces/ZuanUnrollingInterface.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// General region operations
//===----------------------------------------------------------------------===//

def Zuan_MaskYieldOp : Zuan_Op<"mask_yield", [
  Pure, ReturnLike, Terminator, HasParent<"zuan::MaskOp">,
]> {
  let summary = "Terminator for the mask region";
  let description = [{
    `zuan.mask_yield` operation is the terminator of the mask region. Only tile
    type can be yielded, including 0-D tiles.
  }];
  let arguments = (ins Variadic<Zuan_TileType>:$tiles);
  let builders = [
    OpBuilder<(ins), [{ /* do nothing */ }]>
  ];
  let assemblyFormat = [{
    ($tiles^ `:` type($tiles))? attr-dict
  }];
}

def Zuan_MaskOp : Zuan_Op<"mask", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"zuan::MaskYieldOp">
]> {
  let summary = "A mask region for conditional execution";
  let description = [{
    `zuan.mask` operation is a region that contains conditional tile operations.
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
  let arguments = (ins ZuanTileOfType<[I1]>:$mask,
                       Optional<Zuan_TileType>:$maskedoff);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<Zuan_TileType>:$results);

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
// High-level tile operations
//===----------------------------------------------------------------------===//

def Zuan_MatmulOp : Zuan_Op<"matmul", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  Pure, InferTypeOpAdaptor,
  AllRanksMatch<["lhs", "rhs", "result"]>,
]> {
  let summary = "Matrix multiplication operation in Zuan dialect";
  let description = [{
    `zuan.matmul` operation computes the matrix multiplication of two tiles. If
    the two operands differ in ranks, it will execute a matvec or vecmat mul.

    The leading dims of the two tiles should be compatible.
  }];

  let arguments = (ins Zuan_TileType:$lhs, Zuan_TileType:$rhs);
  let results = (outs Zuan_TileType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))
  }];

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Get the shared leading size of two operands.
    unsigned getLeadingSize();
  }];
}

def Zuan_ReductionOp : Zuan_Op<"reduction", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["tile", "result"]>,
  OptionalTypesMatchWith<"result and the init have the same type",
                          "result", "init", "$_self", "TileType::isCompatible">,
]> {
  let summary = "Multi-dimensional reduction operation in Zuan dialect";
  let description = [{
    `zuan.reduction` operation computes the reduction of the input tile along
    the specified dimensions. The reduced dimensions will be removed. The init
    tile should be shape-compatible with the result.
  }];

  let arguments = (ins Zuan_CombiningKindAttr:$kind,
                       Zuan_TileType:$tile,
                       DenseI64ArrayAttr:$dims,
                       Optional<Zuan_TileType>:$init);
  let results = (outs Zuan_TileType:$result);

  let assemblyFormat = [{
    $kind $tile $dims (`,` $init^)? attr-dict
    `:` qualified(type($tile)) (`,` qualified(type($init))^)?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// General tile operations
//===----------------------------------------------------------------------===//

def Zuan_LoadOp : Zuan_Op<"load", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
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
  let hasVerifier = 1;
}

def Zuan_StoreOp : Zuan_Op<"store", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  AllShapesMatch<["value", "base"]>, AllElementTypesMatch<["value", "base"]>
]> {
  let summary = "Store a tile to a memref";
  let description = [{
    `zuan.store` operation writes a tile to a memref. The shape of the tile
    should be the same as the memref.
  }];

  let arguments = (ins Zuan_TileType:$value,
                       Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base);
  let assemblyFormat = [{
    $value `,` $base attr-dict `:` type($value) `,` type($base)
  }];
}

def Zuan_DimOp : Zuan_Op<"dim", [Pure]> {
  let summary = "Query the runtime extent of a tile dimension";
  let description = [{
    `zuan.dim` materializes the extent of a tile value at a constant dimension
    index. Static dimensions fold to constants. Dynamic dimensions are forwarded
    through the producer semantics when possible.
  }];

  let arguments = (ins Zuan_TileType:$tile, I64Attr:$dim);
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

def Zuan_ExtractOp : Zuan_Op<"extract", [Pure, InferTypeOpAdaptor]> {
  let summary = "Extract the scalar element from a rank-0 tile";
  let description = [{
    `zuan.extract` converts a rank-0 tile into its scalar element.
  }];

  let arguments = (ins ZuanTileOfRank<[0]>:$tile);
  let results = (outs ZuanTileElement:$result);

  let assemblyFormat = [{
    $tile attr-dict `:` qualified(type($tile))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def Zuan_SplatOp : Zuan_Op<"splat", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["value", "result"]>
]> {
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
  let hasVerifier = 1;
}

def Zuan_OuterOp : Zuan_Op<"outer", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  Pure, InferTypeOpAdaptor,
  AllElementTypesMatch<["lhs", "rhs", "result"]>
]> {
  let summary = "Outer operation";
  let description = [{
    `zuan.outer` operation computes the outer product/sum/... of two tiles.
    The result equals broadcasting and applying the element-wise operation to
    the two tiles.

    The leading dims of two tiles should be compatible. If the rank of one of
    the tiles is less than the other by 1, this equals a vector-scalar
    operation at the most minor dimension.
  }];

  let arguments = (ins Zuan_CombiningKindAttr:$kind,
                       Zuan_TileType:$lhs,
                       Zuan_TileType:$rhs);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{
    $kind $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))
  }];
  let hasVerifier = 1;
}

def Zuan_CastOp : Zuan_Op<"cast", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  Pure, AllShapesMatch<["tile", "result"]>
]> {
  let summary = "Cast all elements of a tile to another type";
  let description = [{
    `zuan.cast` operation casts all elements of a tile to another type. The
    cast ops in arith dialect only support `vector` and `tensor` types, other
    types are regarded as cast-incompatible. So here is a dedicated cast op for
    `zuan.tile`, implemented with `CastKind` for simplicity.
  }];
  let arguments = (ins Zuan_CastKindAttr:$kind, Zuan_TileType:$tile);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{
    $kind $tile attr-dict `:` qualified(type($tile)) `to` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Zuan_StepOp : Zuan_Op<"step", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
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
  let hasVerifier = 1;
}

def Zuan_SelectOp : Zuan_Op<"select", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  Pure, AllShapesMatch<["cond", "lhs", "rhs", "result"]>,
  AllTypesMatch<["lhs", "rhs", "result"]>
]> {
  let summary = "Select operation";
  let description = [{
    `zuan.select` operation selects elements from the two tiles based on the
    condition tile. The shape of the condition tile should be the same as the
    other two tiles.
  }];
  let arguments = (ins Zuan_TileType:$cond,
                       Zuan_TileType:$lhs,
                       Zuan_TileType:$rhs);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{
    $cond `,` $lhs `,` $rhs attr-dict `:` type($cond) `,` type($lhs)
  }];
  let hasVerifier = 1;
}

def Zuan_GatherOp : Zuan_Op<"gather", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  InferTypeOpAdaptor, AllElementTypesMatch<["base", "result"]>
]> {
  let summary = "Memory gather operation";
  let description = [{
    Given the source memref and indices, `zuan.gather` loads the values from
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
                       Variadic<ZuanTileOfType<[Index]>>:$indices);
  let results = (outs Zuan_TileType:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base) (`,` type($indices)^)?
  }];
  let hasVerifier = 1;
}

def Zuan_ScatterOp : Zuan_Op<"scatter", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  AllElementTypesMatch<["value", "base"]>
]> {
  let summary = "Memory scatter operation";
  let description = [{
    Given the source tile, indices, and the destination memref, `zuan.scatter`
    writes the values to the destination with the following behavior:

    ```
    dest[indices_0[i_0, i_1, ... i_n],
         indices_1[i_0, i_1, ... i_n],
         ...
         indices_k[i_0, i_1, ... i_n]] = source[i_0, i_1, ... i_n]
    ```

    The indices are expected to be the same shape as the value.
  }];

  let arguments = (ins Zuan_TileType:$value,
                       Arg<AnyMemRef, "The memref to scatter to", [MemWrite]>:$base,
                       Variadic<ZuanTileOfType<[Index]>>:$indices);
  let assemblyFormat = [{
    $value `,` $base `[` $indices `]` attr-dict `:` type($value) `,` type($base) (`,` type($indices)^)?
  }];
  let hasVerifier = 1;
}

#endif // ZUAN_IR_ZUANOPS
