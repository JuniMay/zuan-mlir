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
include "Zuan/Interfaces/ZuanInferShapeInterface.td"
include "Zuan/Interfaces/ZuanUnrollingInterface.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dynamic and Yield operations
//===----------------------------------------------------------------------===//

def Zuan_YieldOp : Zuan_Op<"yield", [
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  // The yield op is the terminator of the dynamic region.
  ReturnLike, Terminator,
  // There might be memory effects in the store region.
  RecursiveMemoryEffects,
  // The store region does not have any arguments, nor terminators.
  NoRegionArguments, NoTerminator,
  HasParent<"zuan::DynamicOp">,
]> {
  let summary = "Yield operation in Zuan dialect";
  let description = [{
    `zuan.yield` operation brings the computed value out of the dynamic region.
    The scalars will be yielded as results. And all the stores should be done in
    the yield region.

    The yielded values corresponds to the results of the dynamic region. 0-D
    tiles can also be yielded as scalars.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[ZuanTileElement, ZuanTileOfRank<[0]>]>>:$scalars);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins)>,
    OpBuilder<(ins "ValueRange":$scalars,
                   CArg<"function_ref<void(OpBuilder &, Location)>">:$bodyBuilder)>
  ];

  let assemblyFormat = [{
    ($scalars^ `:` type($scalars))? $body attr-dict
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def Zuan_DynamicOp : Zuan_Op<"dynamic", [
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DestinationStyleOpInterface,
  RecursiveMemoryEffects // The memory effects are in the yield op region.
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
    in the dimension, the dynamic region will be duplicated with each of their
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
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Zuan_MaskYieldOp : Zuan_Op<"mask_yield", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, ReturnLike, Terminator, HasParent<"zuan::MaskOp">,
]> {
  let summary = "Terminator for the mask region";
  let description = [{
    `zuan.mask_yield` operation is the terminator of the mask region. Only tile
    type can be yielded, including 0-D tiles.
  }];
  let arguments = (ins Optional<Zuan_TileType>:$tile);
  let builders = [
    OpBuilder<(ins), [{ /* do nothing */ }]>
  ];
  let assemblyFormat = [{
    ($tile^ `:` type($tile))? attr-dict
  }];
}

def Zuan_MaskOp : Zuan_Op<"mask", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"zuan::MaskYieldOp">,
  ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp", "zuan::YieldOp"]>,
]> {
  let summary = "A mask region for conditional execution";
  let description = [{
    `zuan.mask` operation is a region that contains conditional tile operations.
    Only the elements whose corresponding mask is true will be computed. An
    Optional passthru tile `maskedoff` can be provided to fill the masked-off
    elements.

    If no passthru provided, using the masked-off values is undefined, the
    result can be arbitrary. The mask region can be nested, all the masks will
    be combined with logical AND.

    The passthru and mask are up to operations to decide their semantics.

    The mask op can also be used inside the yield op to implement masked stores.

    At most one operation can be inside the mask operation, and the result must
    be tile type.
  }];
  let arguments = (ins ZuanTileOfType<[I1]>:$mask,
                       Optional<Zuan_TileType>:$maskedoff);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Optional<Zuan_TileType>:$result);
  
  let assemblyFormat = [{
    $mask `:` type($mask) 
    (`,` $maskedoff^ `:` type($maskedoff))? $body attr-dict 
    (`:` type($result)^)?
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType,
                   "Value":$mask,
                   CArg<"function_ref<void(OpBuilder &, Location)>">:$bodyBuilder,
                   CArg<"Value", "nullptr">:$maskedoff)>,
    OpBuilder<(ins "Value":$mask,
                   CArg<"function_ref<void(OpBuilder &, Location)>">:$bodyBuilder,
                   CArg<"Value", "nullptr">:$maskedoff)>
  ];
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// High-level tile operations
//===----------------------------------------------------------------------===//

def Zuan_MatmulOp : Zuan_Op<"matmul", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, InferTypeOpAdaptor, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
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

  let hasVerifier = 1;
}

def Zuan_ReductionOp : Zuan_Op<"reduction", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, InferTypeOpAdaptor, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
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
// General Operations
//===----------------------------------------------------------------------===//

def Zuan_LoadOp : Zuan_Op<"load", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  InferTypeOpAdaptor, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
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
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  AllShapesMatch<["value", "base"]>, AllElementTypesMatch<["value", "base"]>,
  ParentOneOf<["zuan::YieldOp", "zuan::MaskOp"]>
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

def Zuan_SplatOp : Zuan_Op<"splat", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, InferTypeOpAdaptor, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
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
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, InferTypeOpAdaptor, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
  AllElementTypesMatch<["lhs", "rhs", "result"]>
]> {
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
  let hasVerifier = 1;
}

def Zuan_CastOp : Zuan_Op<"cast", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
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
  let hasVerifier = 1;
}

def Zuan_StepOp : Zuan_Op<"step", [
  DeclareOpInterfaceMethods<ZuanUnrollingInterface>,
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, InferTypeOpAdaptor, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
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
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  Pure, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
  AllShapesMatch<["cond", "lhs", "rhs", "result"]>,
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
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  InferTypeOpAdaptor, ParentOneOf<["zuan::DynamicOp", "zuan::MaskOp"]>,
  AllElementTypesMatch<["base", "result"]>
]> {
  let summary = "Memory gather operation";
  let description = [{
    Given the source memref and indices, `zuan.gather` loads the values
    from the source with the following behavior:

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
  DeclareOpInterfaceMethods<ZuanInferShapeInterface>,
  AllElementTypesMatch<["value", "base"]>,
  ParentOneOf<["zuan::YieldOp", "zuan::MaskOp"]>
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
