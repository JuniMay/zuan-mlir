// RUN: dyno-opt %s \
// RUN:   -dyno-stripmining='vf=8 scalable=true reduction-mode=auto \
// RUN:                     fp-policy=strict' \
// RUN: | tee %t.lowered.mlir \
// RUN: | FileCheck %s \
// RUN: && mv -f %t.lowered.mlir \
// RUN:   $(dirname %t)/reduction.lowered.mlir

func.func @factorized_i32(%src: memref<?x?xi32>) -> i32 {
  %tile = dyno.load %src : memref<?x?xi32>
  %reduced = dyno.reduction <add> %tile [0, 1] : !dyno.tile<?x?xi32>
  %scalar = dyno.extract %reduced : !dyno.tile<i32>
  return %scalar : i32
}

// CHECK-LABEL: func.func @factorized_i32
// CHECK: vp.getvl
// CHECK: dyno.splat %{{.*}} [%{{.*}}] : i32
// CHECK: scf.while
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<?xi32>)
// CHECK: arith.addi %{{.*}}, %{{.*}} : !dyno.tile<?xi32>
// CHECK: dyno.step %c0, 0, [%{{.*}}] : index
// CHECK: dyno.splat %{{.*}} [%{{.*}}] : index
// CHECK: arith.cmpi ult, %{{.*}}, %{{.*}} : !dyno.tile<?xindex>
// CHECK: dyno.mask %{{.*}} : <?xi1>, %{{.*}} : !dyno.tile<?xi32>
// CHECK: arith.addi %{{.*}}, %{{.*}} : !dyno.tile<?xi32>
// CHECK: dyno.reduction <add> %{{.*}} [0] {dyno.parallel_stripmine, dyno.stripmined}

func.func @factorized_i32_3d(%src: memref<?x?x?xi32>) -> i32 {
  %tile = dyno.load %src : memref<?x?x?xi32>
  %reduced = dyno.reduction <add> %tile [0, 1, 2] : !dyno.tile<?x?x?xi32>
  %scalar = dyno.extract %reduced : !dyno.tile<i32>
  return %scalar : i32
}

// CHECK-LABEL: func.func @factorized_i32_3d
// CHECK: vp.getvl
// CHECK: scf.while
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<?xi32>)
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<?xi32>)
// CHECK: dyno.load %{{.*}} : memref<?xi32, strided<[?], offset: ?>>
// CHECK: arith.addi %{{.*}}, %{{.*}} : !dyno.tile<?xi32>
// CHECK: dyno.reduction <add> %{{.*}} [0] {dyno.parallel_stripmine, dyno.stripmined}

func.func @factorized_i32_nonconsecutive(%src: memref<?x?x?xi32>) -> i32 {
  %tile = dyno.load %src : memref<?x?x?xi32>
  %partial = dyno.reduction <add> %tile [0, 2] : !dyno.tile<?x?x?xi32>
  %reduced = dyno.reduction <add> %partial [0] : !dyno.tile<?xi32>
  %scalar = dyno.extract %reduced : !dyno.tile<i32>
  return %scalar : i32
}

// CHECK-LABEL: func.func @factorized_i32_nonconsecutive
// CHECK: memref.dim %arg0, %c1 : memref<?x?x?xi32>
// CHECK: vp.getvl
// CHECK: scf.while
// CHECK: memref.subview %arg0[0, %{{.*}}, 0] [%{{.*}}, %{{.*}}, %{{.*}}] [1, 1, 1]
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<?xi32>)
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<?xi32>)
// CHECK: dyno.reduction <add> %{{.*}} [0] {dyno.parallel_stripmine, dyno.stripmined}

func.func @ordered_f32(%src: memref<?x?xf32>) -> f32 {
  %tile = dyno.load %src : memref<?x?xf32>
  %reduced = dyno.reduction <add> %tile [0, 1] : !dyno.tile<?x?xf32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @ordered_f32
// CHECK: dyno.splat %{{.*}} [] : f32
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<f32>)
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<f32>)
// CHECK: arith.addf %{{.*}}, %{{.*}} : !dyno.tile<f32>
// CHECK-NOT: dyno.parallel_stripmine
// CHECK-NOT: dyno.reduction <add>

func.func @ordered_f32_3d(%src: memref<?x?x?xf32>) -> f32 {
  %tile = dyno.load %src : memref<?x?x?xf32>
  %reduced = dyno.reduction <add> %tile [0, 1, 2] : !dyno.tile<?x?x?xf32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @ordered_f32_3d
// CHECK: dyno.splat %{{.*}} [] : f32
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<f32>)
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<f32>)
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %{{.*}}) -> (!dyno.tile<f32>)
// CHECK: dyno.load %{{.*}} : memref<f32, strided<[], offset: ?>>
// CHECK: arith.addf %{{.*}}, %{{.*}} : !dyno.tile<f32>
// CHECK-NOT: dyno.parallel_stripmine
// CHECK-NOT: dyno.reduction <add>

func.func @reduction0d(%arg0: f32) -> f32 {
  %tile = dyno.splat %arg0 [] : f32
  %reduced = dyno.reduction <add> %tile [] : !dyno.tile<f32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// CHECK-LABEL: func.func @reduction0d
// CHECK: dyno.reduction <add> %{{.*}} []
