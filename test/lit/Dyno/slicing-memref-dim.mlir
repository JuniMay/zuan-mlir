// RUN: dyno-opt %s \
// RUN:   -lower-dyno='target-rank=1' \
// RUN: | tee %t.lowered.mlir \
// RUN: | FileCheck %s \
// RUN: && mv -f %t.lowered.mlir \
// RUN:   $(dirname %t)/slicing-memref-dim.lowered.mlir

// CHECK-LABEL: func.func @scalar_expr_uses_sliced_dims
func.func @scalar_expr_uses_sliced_dims(%dst: memref<?x?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = memref.dim %dst, %c0 : memref<?x?xindex>
  %n = memref.dim %dst, %c1 : memref<?x?xindex>
  %delta = arith.subi %n, %m : index
  %tile = dyno.step %delta, 1, [%m, %n] : index
  dyno.store %tile, %dst : !dyno.tile<?x?xindex>, memref<?x?xindex>
  return
}

// CHECK: %[[COLS:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xindex>
// CHECK: scf.for
// CHECK: %[[SUBVIEW:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [1, %[[COLS]]] [1, 1]
// CHECK: %[[DELTA:.*]] = arith.subi %[[COLS]], %c1 : index
// CHECK: %[[STEP:.*]] = dyno.step %[[DELTA]], 0, [%[[COLS]]] : index
// CHECK: dyno.store %[[STEP]], %[[SUBVIEW]] : <?xindex>, memref<?xindex

// CHECK-LABEL: func.func @dropped_dim_first_use
func.func @dropped_dim_first_use(%dst: memref<?x?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = memref.dim %dst, %c0 : memref<?x?xindex>
  %n = memref.dim %dst, %c1 : memref<?x?xindex>
  %tile = dyno.step %m, 1, [%m, %n] : index
  dyno.store %tile, %dst : !dyno.tile<?x?xindex>, memref<?x?xindex>
  return
}

// CHECK: %[[COLS2:.*]] = memref.dim %{{.*}}, %c1 : memref<?x?xindex>
// CHECK: scf.for
// CHECK: %[[SUBVIEW2:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [1, %[[COLS2]]] [1, 1]
// CHECK: %[[STEP2:.*]] = dyno.step %c1, 0, [%[[COLS2]]] : index
// CHECK: dyno.store %[[STEP2]], %[[SUBVIEW2]] : <?xindex>, memref<?xindex
