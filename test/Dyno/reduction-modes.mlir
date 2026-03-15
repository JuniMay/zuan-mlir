// RUN: dyno-opt %s \
// RUN:   -lower-dyno='target-rank=2' \
// RUN:   -dyno-stripmining='vf=16 scalable=true reduction-mode=parallel \
// RUN:                     fp-policy=relaxed' \
// RUN: | tee %t.parallel.mlir \
// RUN: | FileCheck %s --check-prefix=PAR \
// RUN: && mv -f %t.parallel.mlir \
// RUN:   $(dirname %t)/reduction-modes.parallel.mlir
// RUN: dyno-opt %s \
// RUN:   -lower-dyno='target-rank=2' \
// RUN:   -dyno-stripmining='vf=16 scalable=true reduction-mode=sequential \
// RUN:                     fp-policy=strict' \
// RUN: | tee %t.sequential.mlir \
// RUN: | FileCheck %s --check-prefix=SEQ \
// RUN: && mv -f %t.sequential.mlir \
// RUN:   $(dirname %t)/reduction-modes.sequential.mlir
// RUN: dyno-opt %s \
// RUN:   -lower-dyno='target-rank=2' \
// RUN:   -dyno-stripmining='vf=16 scalable=true reduction-mode=auto \
// RUN:                     fp-policy=strict' \
// RUN: | tee %t.auto.mlir \
// RUN: | FileCheck %s --check-prefix=AUTO \
// RUN: && mv -f %t.auto.mlir \
// RUN:   $(dirname %t)/reduction-modes.auto.mlir

func.func @reduce1d(%src: memref<?xf32>) -> f32 {
  %tile = dyno.load %src : memref<?xf32>
  %reduced = dyno.reduction <add> %tile [0] : !dyno.tile<?xf32>
  %scalar = dyno.extract %reduced : !dyno.tile<f32>
  return %scalar : f32
}

// PAR-LABEL: func.func @reduce1d
// PAR: vp.getvl
// PAR: dyno.splat
// PAR: scf.while
// PAR: dyno.step %c0, 0, [%{{.*}}] : index
// PAR: dyno.splat %{{.*}} [%{{.*}}] : index
// PAR: arith.cmpi ult, %{{.*}}, %{{.*}} : !dyno.tile<?xindex>
// PAR: dyno.mask %{{.*}} : <?xi1>, %{{.*}} : !dyno.tile<?xf32>
// PAR: arith.addf %{{.*}}, %{{.*}} : !dyno.tile<?xf32>
// PAR: dyno.reduction <add> %{{.*}} [0] {dyno.parallel_reassoc, dyno.parallel_stripmine, dyno.stripmined}
// PAR: dyno.extract

// SEQ-LABEL: func.func @reduce1d
// SEQ: dyno.splat
// SEQ: scf.while
// SEQ: dyno.reduction <add> %{{.*}} [0], %{{.*}} {dyno.sequential_stripmine, dyno.stripmined}
// SEQ-NOT: scf.for
// SEQ-NOT: dyno.parallel_stripmine

// AUTO-LABEL: func.func @reduce1d
// AUTO: dyno.splat
// AUTO: scf.while
// AUTO: dyno.reduction <add> %{{.*}} [0], %{{.*}} {dyno.sequential_stripmine, dyno.stripmined}
// AUTO-NOT: scf.for
// AUTO-NOT: dyno.parallel_stripmine
