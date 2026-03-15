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

func.func @matmul_bad_contract(%lhs: !dyno.tile<2x4x3xf32>, %rhs: !dyno.tile<2x5x7xf32>) {
  // expected-error@+1 {{expected the inner dimensions of the lhs and rhs to be compatible}}
  %0 = dyno.matmul %lhs, %rhs : !dyno.tile<2x4x3xf32>, !dyno.tile<2x5x7xf32>
  return
}

// -----

func.func @outer_bad_leading(%lhs: !dyno.tile<2x4xf32>, %rhs: !dyno.tile<3x7xf32>) {
  // expected-error@+1 {{expected the leading dimensions of the lhs and rhs to be compatible}}
  %0 = dyno.outer <add> %lhs, %rhs : !dyno.tile<2x4xf32>, !dyno.tile<3x7xf32>
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
