//===- DynoTypes.td - Dyno Types ---------------------------*- tablegen -*-===//
//
// This file declares the types for the Dyno dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNO_IR_DYNOTYPES
#define DYNO_IR_DYNOTYPES

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "Dyno/IR/DynoDialect.td"

def DynoTileElement : AnyTypeOf<[AnySignlessIntegerOrIndex, AnyFloat]>;

def Dyno_TileType : Dyno_Type<"Tile", "tile", [
  ShapedTypeInterface, ValueSemantics
]> {
  let summary = "Dyno tile type";
  let description = [{
    Tiles in dyno are multi-dimensional vectors of elements. The dimension sizes
    can be dynamic or static.
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        DynoTileElement:$elementType);
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

class IsDynoTileOfRankPref<list<int> allowedRanks> :
  Or<!foreach(allowed, allowedRanks,
              CPred<[{ ::llvm::cast<::mlir::dyno::TileType>($_self).getRank()
                       == }] # allowed>)>;

class DynoTileOfRank<list<int> allowedRanks> : Type<
  IsDynoTileOfRankPref<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::dyno::TileType">;

class DynoTileOfType<list<Type> allowedTypes> :
  ShapedContainerType<
    allowedTypes, 
    CPred<[{ ::llvm::isa<::mlir::dyno::TileType>($_self) }]>,
    "dyno.tile",
    "::mlir::dyno::TileType">;

#endif // DYNO_IR_DYNOTYPES
