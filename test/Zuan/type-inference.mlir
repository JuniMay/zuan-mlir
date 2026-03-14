// RUN: zuan-opt %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_static_leading
func.func @matmul_static_leading(%a: memref<2x4x3xf32>, %b: memref<2x3x7xf32>,
                                 %c: memref<2x4x7xf32>) {
  %a_tile = zuan.load %a : memref<2x4x3xf32>
  %b_tile = zuan.load %b : memref<2x3x7xf32>
  %c_tile = zuan.load %c : memref<2x4x7xf32>
  %mm = zuan.matmul %a_tile, %b_tile :
      !zuan.tile<2x4x3xf32>, !zuan.tile<2x3x7xf32>
  // CHECK: arith.addf %{{.*}}, %{{.*}} : !zuan.tile<2x4x7xf32>
  %sum = arith.addf %mm, %c_tile : !zuan.tile<2x4x7xf32>
  zuan.store %sum, %c : !zuan.tile<2x4x7xf32>, memref<2x4x7xf32>
  return
}

// CHECK-LABEL: func.func @extract_type
func.func @extract_type(%arg0: f32) -> f32 {
  %tile = zuan.splat %arg0 [] : f32
  // CHECK: %[[SCALAR:.*]] = zuan.extract %{{.*}} : !zuan.tile<f32>
  %scalar = zuan.extract %tile : !zuan.tile<f32>
  return %scalar : f32
}
