// RUN: zuan-opt -lower-zuan="target-rank=2" %s | FileCheck %s

// CHECK-LABEL: func.func @store
func.func @store(%from0: memref<?x?x?xf32>, %from1: memref<?x?x?x?xf32>,
                 %to0: memref<?x?x?xf32>, %to1: memref<?x?x?x?xf32>) {
  zuan.dynamic (%to0, %to1: memref<?x?x?xf32>, memref<?x?x?x?xf32>) {
  ^bb0(%to0tile: !zuan.tile<?x?x?xf32>, %to1tile: !zuan.tile<?x?x?x?xf32>):
    %from0tile = zuan.load %from0 : memref<?x?x?xf32>
    %from1tile = zuan.load %from1 : memref<?x?x?x?xf32>
    %add = arith.addf %from0tile, %to0tile : !zuan.tile<?x?x?xf32>
    zuan.yield {
      zuan.store %add, %to0 : !zuan.tile<?x?x?xf32>, memref<?x?x?xf32>
      zuan.store %from1tile, %to1 : !zuan.tile<?x?x?x?xf32>, memref<?x?x?x?xf32>
    }
  }
  return
}
