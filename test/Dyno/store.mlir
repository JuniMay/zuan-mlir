// RUN: dyno-opt %s -lower-dyno='target-rank=2' -dyno-stripmining="vf=8 scalable=true" | FileCheck %s

// CHECK-LABEL: func.func @store
func.func @store(%from0: memref<?x?x?xf32>, %from1: memref<?x?x?x?xf32>,
                 %to0: memref<?x?x?xf32>, %to1: memref<?x?x?x?xf32>) {
  %from0_tile = dyno.load %from0 : memref<?x?x?xf32>
  %from1_tile = dyno.load %from1 : memref<?x?x?x?xf32>
  %to0_tile = dyno.load %to0 : memref<?x?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.while
  %sum = arith.addf %from0_tile, %to0_tile : !dyno.tile<?x?x?xf32>
  dyno.store %sum, %to0 : !dyno.tile<?x?x?xf32>, memref<?x?x?xf32>
  dyno.store %from1_tile, %to1 : !dyno.tile<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}

// CHECK-LABEL: func.func @store1d
func.func @store1d(%from: memref<?xf32>, %to: memref<?xf32>) {
  %from_tile = dyno.load %from : memref<?xf32>
  %to_tile = dyno.load %to : memref<?xf32>
  // CHECK: scf.while
  %sum = arith.addf %from_tile, %to_tile : !dyno.tile<?xf32>
  dyno.store %sum, %to : !dyno.tile<?xf32>, memref<?xf32>
  return
}
