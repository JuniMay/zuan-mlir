// RUN: dyno-opt -lower-dyno='target-rank=2' -dyno-stripmining='vf=16 scalable=true reduction-mode=parallel fp-policy=relaxed' %s | FileCheck %s --check-prefix=PAR
// RUN: dyno-opt -lower-dyno='target-rank=2' -dyno-stripmining='vf=16 scalable=true reduction-mode=sequential fp-policy=strict' %s | FileCheck %s --check-prefix=SEQ

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
// PAR: arith.addf %{{.*}}, %{{.*}} {dyno_passthru_operand = 0 : index} : !dyno.tile<?xf32>
// PAR: dyno.reduction <add> %{{.*}} [0] {dyno.stripmined}

// SEQ-LABEL: func.func @reduce1d
// SEQ: vp.getvl
// SEQ: dyno.splat
// SEQ: scf.while
// SEQ: arith.addf %{{.*}}, %{{.*}} {dyno_passthru_operand = 0 : index} : !dyno.tile<?xf32>
// SEQ: dyno.reduction <add> %{{.*}} [0] {dyno.stripmined}
