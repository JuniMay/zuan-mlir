//===- ZuanTypes.td - Zuan Types ---------------------------*- tablegen -*-===//
//
// This file declares the types for the Zuan dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZUAN_IR_ZUANTYPES
#define ZUAN_IR_ZUANTYPES

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "Zuan/IR/ZuanDialect.td"

def ZuanTileElement : AnyTypeOf<[AnySignlessIntegerOrIndex, AnyFloat]>;

def Zuan_TileType : Zuan_Type<"Tile", "tile", [
  ShapedTypeInterface, ValueSemantics
]> {
  let summary = "Zuan tile type";
  let description = [{
    Tiles in zuan are multi-dimensional vectors of elements. The dimension sizes
    can be dynamic or static.
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        ZuanTileElement:$elementType);
  let builders = [
    TypeBuilderWithInferredContext<
      (ins "ArrayRef<int64_t>":$shape, "Type":$elementType), 
      [{ return $_get(elementType.getContext(), shape, elementType); }]>
  ];

  let extraClassDeclaration = [{
    bool hasRank() const { return true; }

    TileType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const {
      return get(shape.value_or(getShape()), elementType);
    }

    unsigned getRank() const { return getShape().size(); }

    static bool isDimCompatible(int64_t a, int64_t b);
    static bool isShapeCompatible(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs);
    static bool isCompatible(Type a, Type b);

    /// Select the static sizes from the given shapes
    static SmallVector<int64_t> selectStaticShape(ArrayRef<int64_t> lhs,
                                                  ArrayRef<int64_t> rhs);
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

class IsZuanTileOfRankPref<list<int> allowedRanks> :
  Or<!foreach(allowed, allowedRanks,
              CPred<[{ ::llvm::cast<::mlir::zuan::TileType>($_self).getRank()
                       == }] # allowed>)>;

class ZuanTileOfRank<list<int> allowedRanks> : Type<
  IsZuanTileOfRankPref<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::zuan::TileType">;

class ZuanTileOfType<list<Type> allowedTypes> :
  ShapedContainerType<
    allowedTypes, 
    CPred<[{ ::llvm::isa<::mlir::zuan::TileType>($_self) }]>,
    "zuan.tile",
    "::mlir::zuan::TileType">;

#endif // ZUAN_IR_ZUANTYPES
