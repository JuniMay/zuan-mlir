// RUN: zuan-opt %s -lower-zuan='target-rank=2' -zuan-stripmining="vf=8 scalable=true" | FileCheck %s

// CHECK-LABEL: func.func @store
func.func @store(%from0: memref<?x?x?xf32>, %from1: memref<?x?x?x?xf32>,
                 %to0: memref<?x?x?xf32>, %to1: memref<?x?x?x?xf32>) {
  %from0_tile = zuan.load %from0 : memref<?x?x?xf32>
  %from1_tile = zuan.load %from1 : memref<?x?x?x?xf32>
  %to0_tile = zuan.load %to0 : memref<?x?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.while
  %sum = arith.addf %from0_tile, %to0_tile : !zuan.tile<?x?x?xf32>
  zuan.store %sum, %to0 : !zuan.tile<?x?x?xf32>, memref<?x?x?xf32>
  zuan.store %from1_tile, %to1 : !zuan.tile<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}

// CHECK-LABEL: func.func @store1d
func.func @store1d(%from: memref<?xf32>, %to: memref<?xf32>) {
  %from_tile = zuan.load %from : memref<?xf32>
  %to_tile = zuan.load %to : memref<?xf32>
  // CHECK: scf.while
  %sum = arith.addf %from_tile, %to_tile : !zuan.tile<?xf32>
  zuan.store %sum, %to : !zuan.tile<?xf32>, memref<?xf32>
  return
}
