// RUN: zuan-opt -lower-zuan %s | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  zuan.dynamic (%c : memref<?x?xf32>) {
  ^bb0(%c_tile: !zuan.tile<?x?xf32>):
    %a_tile = zuan.load %a : memref<?x?xf32>
    %b_tile = zuan.load %b : memref<?x?xf32>
    // CHECK: scf.for
    // CHECK: zuan.outer <mul>
    // CHECK: arith.addf
    %mm = zuan.matmul %a_tile, %b_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    %add = arith.addf %mm, %c_tile : !zuan.tile<?x?xf32>
    zuan.yield {
      zuan.store %add, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
    }
  }
  return
}

// TODO: check correctness of the lowering
// CHECK-LABEL: func.func @matmul2
func.func @matmul2(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>, %d: memref<?x?xf32>) {
  // A @ B @ C
  zuan.dynamic (%d : memref<?x?xf32>) {
  ^bb0(%d_tile: !zuan.tile<?x?xf32>):
    %a_tile = zuan.load %a : memref<?x?xf32>
    %b_tile = zuan.load %b : memref<?x?xf32>
    %c_tile = zuan.load %c : memref<?x?xf32>
    %mm1 = zuan.matmul %a_tile, %b_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    %mm2 = zuan.matmul %mm1, %c_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    %add = arith.addf %mm2, %d_tile : !zuan.tile<?x?xf32>
    zuan.yield {
      zuan.store %add, %d : !zuan.tile<?x?xf32>, memref<?x?xf32>
    }
  }
  return
}
