// RUN: zuan-opt -lower-zuan -zuan-stripmining="vf=8 scalable=true" %s | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                  %c: memref<?x?xf32>) {
  %a_tile = zuan.load %a : memref<?x?xf32>
  %b_tile = zuan.load %b : memref<?x?xf32>
  %c_tile = zuan.load %c : memref<?x?xf32>
  // CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ACC:.+]] = %{{.+}})
  // CHECK: %[[A:.+]] = zuan.load
  // CHECK: %[[B:.+]] = zuan.load
  // CHECK: %[[OUTER:.+]] = zuan.outer <mul> %[[A]], %[[B]]
  // CHECK: %[[NEWACC:.+]] = arith.addf %[[ACC]], %[[OUTER]]
  %mm = zuan.matmul %a_tile, %b_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
  %sum = arith.addf %mm, %c_tile : !zuan.tile<?x?xf32>
  zuan.store %sum, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @matmul2
func.func @matmul2(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                   %c: memref<?x?xf32>, %d: memref<?x?xf32>) {
  %a_tile = zuan.load %a : memref<?x?xf32>
  %b_tile = zuan.load %b : memref<?x?xf32>
  %c_tile = zuan.load %c : memref<?x?xf32>
  %d_tile = zuan.load %d : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  %mm1 = zuan.matmul %a_tile, %b_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
  %mm2 = zuan.matmul %mm1, %c_tile : !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
  %sum = arith.addf %mm2, %d_tile : !zuan.tile<?x?xf32>
  zuan.store %sum, %d : !zuan.tile<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @matmul_masked
func.func @matmul_masked(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                         %c: memref<?x?xf32>, %m: memref<?x?xi1>) {
  %mask = zuan.load %m : memref<?x?xi1>
  %a_tile = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %loaded = zuan.load %a : memref<?x?xf32>
    zuan.mask_yield %loaded : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %b_tile = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %loaded = zuan.load %b : memref<?x?xf32>
    zuan.mask_yield %loaded : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %c_tile = zuan.load %c : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: zuan.mask %{{.*}} {
  // CHECK: zuan.mask_yield %{{.*}}
  %mm = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %matmul = zuan.matmul %a_tile, %b_tile :
        !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    zuan.mask_yield %matmul : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %sum = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %add = arith.addf %mm, %c_tile : !zuan.tile<?x?xf32>
    zuan.mask_yield %add : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  zuan.mask %mask : !zuan.tile<?x?xi1> {
    zuan.store %sum, %c : !zuan.tile<?x?xf32>, memref<?x?xf32>
  }
  return
}

// CHECK-LABEL: func.func @matmul2_masked
func.func @matmul2_masked(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                          %c: memref<?x?xf32>, %m: memref<?x?xi1>,
                          %d: memref<?x?xf32>) {
  %mask = zuan.load %m : memref<?x?xi1>
  %a_tile = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %loaded = zuan.load %a : memref<?x?xf32>
    zuan.mask_yield %loaded : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %b_tile = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %loaded = zuan.load %b : memref<?x?xf32>
    zuan.mask_yield %loaded : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %c_tile = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %loaded = zuan.load %c : memref<?x?xf32>
    zuan.mask_yield %loaded : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %d_tile = zuan.load %d : memref<?x?xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: zuan.mask %{{.*}} {
  // CHECK: zuan.mask_yield %{{.*}}
  %mm1 = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %matmul = zuan.matmul %a_tile, %b_tile :
        !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    zuan.mask_yield %matmul : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %mm2 = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %matmul = zuan.matmul %mm1, %c_tile :
        !zuan.tile<?x?xf32>, !zuan.tile<?x?xf32>
    zuan.mask_yield %matmul : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  %sum = zuan.mask %mask : !zuan.tile<?x?xi1> {
    %add = arith.addf %mm2, %d_tile : !zuan.tile<?x?xf32>
    zuan.mask_yield %add : !zuan.tile<?x?xf32>
  } : !zuan.tile<?x?xf32>
  zuan.mask %mask : !zuan.tile<?x?xi1> {
    zuan.store %sum, %d : !zuan.tile<?x?xf32>, memref<?x?xf32>
  }
  return
}
