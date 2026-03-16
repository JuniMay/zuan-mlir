// RUN: dyno-opt %s \
// RUN:   -lower-dyno='target-rank=2' \
// RUN: | tee %t.slice.mlir \
// RUN: | FileCheck %s --check-prefix=SLICE \
// RUN: && mv -f %t.slice.mlir \
// RUN:   $(dirname %t)/reduction-policy.slice.mlir
// RUN: dyno-opt %s \
// RUN:   -lower-dyno='target-rank=2' \
// RUN:   -dyno-stripmining='vf=8 scalable=true reduction-mode=parallel \
// RUN:                     fp-policy=strict' \
// RUN: | tee %t.strip.mlir \
// RUN: | FileCheck %s --check-prefix=STRIP \
// RUN: && mv -f %t.strip.mlir \
// RUN:   $(dirname %t)/reduction-policy.strip.mlir

func.func @slice_preserves_policy(%src: memref<?x?x?x?xf32>,
                                  %init: memref<?x?x?xf32>,
                                  %dst: memref<?x?x?xf32>) {
  %src_tile = dyno.load %src : memref<?x?x?x?xf32>
  %init_tile = dyno.load %init : memref<?x?x?xf32>
  %red = dyno.reduction <add> %src_tile [2], %init_tile {dyno.fp_policy = "relaxed"} :
      !dyno.tile<?x?x?x?xf32>, !dyno.tile<?x?x?xf32>
  dyno.store %red, %dst : !dyno.tile<?x?x?xf32>, memref<?x?x?xf32>
  return
}

// SLICE-LABEL: func.func @slice_preserves_policy
// SLICE: dyno.reduction <add> %{{.*}} [1], %{{.*}} {
// SLICE-SAME: dyno.fp_policy = "relaxed"

func.func @default_materialized_minimumf(%src: memref<?xf32>) -> f32 {
  %tile = dyno.load %src : memref<?xf32>
  %reduced = dyno.reduction <minimumf> %tile [0] : !dyno.tile<?xf32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// STRIP-LABEL: func.func @default_materialized_minimumf
// STRIP: dyno.reduction <minimumf> %{{.*}} [0] {
// STRIP-SAME: dyno.fp_policy = "strict"
// STRIP-SAME: dyno.parallel_stripmine
// STRIP-SAME: dyno.stripmined

func.func @override_relaxed_add_parallel(%src: memref<?xf32>) -> f32 {
  %tile = dyno.load %src : memref<?xf32>
  %reduced = dyno.reduction <add> %tile [0] {dyno.fp_policy = "relaxed"} : !dyno.tile<?xf32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// STRIP-LABEL: func.func @override_relaxed_add_parallel
// STRIP: dyno.reduction <add> %{{.*}} [0] {
// STRIP-SAME: dyno.fp_policy = "relaxed"
// STRIP-SAME: dyno.parallel_reassoc
// STRIP-SAME: dyno.parallel_stripmine
// STRIP-SAME: dyno.stripmined

func.func @factorized_maximumf_strict(%src: memref<?x?xf32>) -> f32 {
  %tile = dyno.load %src : memref<?x?xf32>
  %reduced = dyno.reduction <maximumf> %tile [0, 1] {dyno.fp_policy = "strict"} :
      !dyno.tile<?x?xf32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// STRIP-LABEL: func.func @factorized_maximumf_strict
// STRIP: dyno.reduction <maximumf> %{{.*}} [0] {
// STRIP-SAME: dyno.fp_policy = "strict"
// STRIP-SAME: dyno.parallel_stripmine
// STRIP-SAME: dyno.stripmined
