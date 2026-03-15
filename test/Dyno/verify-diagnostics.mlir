// RUN: dyno-opt -split-input-file -verify-diagnostics %s

func.func @reduction_duplicate_dims(%arg0: !dyno.tile<?x?xf32>) {
  // expected-error@+1 {{expected reduction dims to be unique}}
  %0 = dyno.reduction <add> %arg0 [0, 0] : !dyno.tile<?x?xf32>
  return
}

// -----

func.func @reduction_out_of_range(%arg0: !dyno.tile<?x?xf32>) {
  // expected-error@+1 {{expected reduction dims to be in range [0, 2)}}
  %0 = dyno.reduction <add> %arg0 [2] : !dyno.tile<?x?xf32>
  return
}

// -----

func.func @reduction_bad_init(%arg0: !dyno.tile<?xf32>, %init: !dyno.tile<?xf32>) {
  // expected-error@+1 {{'dyno.reduction' op failed to verify that result and the init have the same type}}
  %0 = dyno.reduction <add> %arg0 [0], %init : !dyno.tile<?xf32>, !dyno.tile<?xf32>
  return
}

// -----

func.func @splat_bad_suffix(%arg0: !dyno.tile<2xf32>) {
  // expected-error@+1 {{expected result suffix shape to match the operand tile shape exactly}}
  %0 = "dyno.splat"(%arg0) {staticDims = array<i64: 4>} : (!dyno.tile<2xf32>) -> !dyno.tile<4x3xf32>
  return
}

// -----

func.func @step_dim_out_of_range(%start: index) {
  // expected-error@+1 {{expected dim to be in range [0, 1)}}
  %0 = "dyno.step"(%start) {dim = 2 : index, staticSizes = array<i64: 4>} : (index) -> !dyno.tile<4xindex>
  return
}

// -----

func.func @cast_bad_shape(%arg0: !dyno.tile<4xf16>) {
  // expected-error@+1 {{failed to verify that all of {tile, result} have same shape}}
  %0 = dyno.cast <extf> %arg0 : !dyno.tile<4xf16> to !dyno.tile<2xf32>
  return
}

// -----

func.func @cast_bad_kind(%arg0: !dyno.tile<?xf32>) {
  // expected-error@+1 {{invalid cast kind}}
  %0 = dyno.cast <indexcastui> %arg0 : !dyno.tile<?xf32> to !dyno.tile<?xindex>
  return
}

// -----

func.func @cast_bad_bitcast_index(%arg0: !dyno.tile<4xindex>) {
  // expected-error@+1 {{invalid cast kind}}
  %0 = dyno.cast <bitcast> %arg0 : !dyno.tile<4xindex> to !dyno.tile<4xi64>
  return
}

// -----

func.func @select_bad_cond_element(%cond: !dyno.tile<?xf32>,
                                   %lhs: !dyno.tile<?xf32>,
                                   %rhs: !dyno.tile<?xf32>) {
  // expected-error@+1 {{expected the condition tile element type to be i1}}
  %0 = dyno.select %cond, %lhs, %rhs : !dyno.tile<?xf32>, !dyno.tile<?xf32>
  return
}

// -----

func.func @mask_bad_maskedoff(%mask: !dyno.tile<?xi1>, %arg0: !dyno.tile<?xf32>,
                              %maskedoff: !dyno.tile<4xf32>) {
  // expected-error@+1 {{expected maskedoff tile type to match each result type exactly}}
  %0 = dyno.mask %mask : !dyno.tile<?xi1>, %maskedoff : !dyno.tile<4xf32> {
    dyno.mask_yield %arg0 : !dyno.tile<?xf32>
  } : !dyno.tile<?xf32>
  return
}

// -----

func.func @mask_effect_only_maskedoff(%mask: !dyno.tile<?xi1>,
                                      %tile: !dyno.tile<?xf32>,
                                      %maskedoff: !dyno.tile<?xf32>,
                                      %dst: memref<?xf32>) {
  // expected-error@+1 {{expected maskedoff only on value-producing masks}}
  dyno.mask %mask : !dyno.tile<?xi1>, %maskedoff : !dyno.tile<?xf32> {
    dyno.store %tile, %dst : !dyno.tile<?xf32>, memref<?xf32>
  }
  return
}

// -----

func.func @mask_effect_only_value_op(%mask: !dyno.tile<?xi1>,
                                     %lhs: !dyno.tile<?xf32>,
                                     %rhs: !dyno.tile<?xf32>) {
  // expected-error@+1 {{expected effect-only masks to contain a masked operation with no results}}
  dyno.mask %mask : !dyno.tile<?xi1> {
    %sum = arith.addf %lhs, %rhs : !dyno.tile<?xf32>
    dyno.mask_yield
  }
  return
}

// -----

func.func @gather_bad_index_count(%base: memref<?x?xf32>, %idx0: !dyno.tile<?xindex>) {
  // expected-error@+1 {{expected the number of indices to be the same as the rank of the base memref}}
  %0 = dyno.gather %base[%idx0] : memref<?x?xf32>, !dyno.tile<?xindex>
  return
}

// -----

func.func @gather_bad_index_shape(%base: memref<?x?xf32>, %idx0: !dyno.tile<?xindex>, %idx1: !dyno.tile<?x?xindex>) {
  // expected-error@+1 {{expected all index tile shapes to match}}
  %0 = dyno.gather %base[%idx0, %idx1] : memref<?x?xf32>, !dyno.tile<?xindex>, !dyno.tile<?x?xindex>
  return
}

// -----

func.func @scatter_bad_value_shape(%base: memref<?x?xf32>, %value: !dyno.tile<?x?xf32>, %idx0: !dyno.tile<?xindex>, %idx1: !dyno.tile<?xindex>) {
  // expected-error@+1 {{expected the result/value tile shape to match the index tile shape}}
  dyno.scatter %value, %base[%idx0, %idx1] : !dyno.tile<?x?xf32>, memref<?x?xf32>, !dyno.tile<?xindex>, !dyno.tile<?xindex>
  return
}

// -----

func.func @dim_out_of_range(%arg0: !dyno.tile<?x?xf32>) {
  // expected-error@+1 {{expected dim to be in range [0, 2)}}
  %0 = dyno.dim %arg0, 2 : !dyno.tile<?x?xf32>
  return
}

// -----

func.func @extract_non_scalar(%arg0: !dyno.tile<?xf32>) {
  // expected-error@+1 {{operand #0 must be  of ranks 0}}
  %0 = dyno.extract %arg0 : !dyno.tile<?xf32>
  return
}
