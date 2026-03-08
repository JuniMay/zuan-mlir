// RUN: zuan-opt -split-input-file -verify-diagnostics %s

func.func @reduction_duplicate_dims(%arg0: !zuan.tile<?x?xf32>) {
  // expected-error@+1 {{expected reduction dims to be unique}}
  %0 = zuan.reduction <add> %arg0 [0, 0] : !zuan.tile<?x?xf32>
  return
}

// -----

func.func @reduction_out_of_range(%arg0: !zuan.tile<?x?xf32>) {
  // expected-error@+1 {{expected reduction dims to be in range [0, 2)}}
  %0 = zuan.reduction <add> %arg0 [2] : !zuan.tile<?x?xf32>
  return
}

// -----

func.func @matmul_bad_contract(%lhs: !zuan.tile<2x4x3xf32>, %rhs: !zuan.tile<2x5x7xf32>) {
  // expected-error@+1 {{expected the inner dimensions of the lhs and rhs to be compatible}}
  %0 = zuan.matmul %lhs, %rhs : !zuan.tile<2x4x3xf32>, !zuan.tile<2x5x7xf32>
  return
}

// -----

func.func @outer_bad_leading(%lhs: !zuan.tile<2x4xf32>, %rhs: !zuan.tile<3x7xf32>) {
  // expected-error@+1 {{expected the leading dimensions of the lhs and rhs to be compatible}}
  %0 = zuan.outer <add> %lhs, %rhs : !zuan.tile<2x4xf32>, !zuan.tile<3x7xf32>
  return
}

// -----

func.func @gather_bad_index_count(%base: memref<?x?xf32>, %idx0: !zuan.tile<?xindex>) {
  // expected-error@+1 {{expected the number of indices to be the same as the rank of the base memref}}
  %0 = zuan.gather %base[%idx0] : memref<?x?xf32>, !zuan.tile<?xindex>
  return
}

// -----

func.func @gather_bad_index_shape(%base: memref<?x?xf32>, %idx0: !zuan.tile<?xindex>, %idx1: !zuan.tile<?x?xindex>) {
  // expected-error@+1 {{expected all index tile shapes to match}}
  %0 = zuan.gather %base[%idx0, %idx1] : memref<?x?xf32>, !zuan.tile<?xindex>, !zuan.tile<?x?xindex>
  return
}

// -----

func.func @scatter_bad_value_shape(%base: memref<?x?xf32>, %value: !zuan.tile<?x?xf32>, %idx0: !zuan.tile<?xindex>, %idx1: !zuan.tile<?xindex>) {
  // expected-error@+1 {{expected the result/value tile shape to match the index tile shape}}
  zuan.scatter %value, %base[%idx0, %idx1] : !zuan.tile<?x?xf32>, memref<?x?xf32>, !zuan.tile<?xindex>, !zuan.tile<?xindex>
  return
}
