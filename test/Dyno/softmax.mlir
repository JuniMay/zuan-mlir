// RUN: dyno-opt -lower-dyno='target-rank=2' -dyno-stripmining='vf=8 scalable=true' %s | FileCheck %s

// CHECK-LABEL: func.func @softmax
func.func @softmax(%src: memref<?xf32>, %dst: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %src, %c0 : memref<?xf32>
  %vsrc = dyno.load %src : memref<?xf32>
  %max = dyno.reduction <maxnumf> %vsrc [0] : !dyno.tile<?xf32>
  %max_splat = dyno.splat %max [%dim] : !dyno.tile<f32>
  %sub = arith.subf %vsrc, %max_splat : !dyno.tile<?xf32>
  %exp = math.exp %sub : !dyno.tile<?xf32>
  %sum = dyno.reduction <add> %exp [0] : !dyno.tile<?xf32>
  %sum_splat = dyno.splat %sum [%dim] : !dyno.tile<f32>
  %vdst = arith.divf %exp, %sum_splat : !dyno.tile<?xf32>
  // CHECK: %[[MAX_LOOP:.*]]:3 = scf.while
  // CHECK: %[[MAX:.*]] = dyno.reduction <maxnumf> %[[MAX_LOOP]]#2 [0]
  // CHECK: %[[SUM_LOOP:.*]]:3 = scf.while
  // CHECK: %[[SUM:.*]] = dyno.reduction <add> %[[SUM_LOOP]]#2 [0]
  // CHECK: dyno.store %{{.*}}, %{{.*}}
  dyno.store %vdst, %dst : !dyno.tile<?xf32>, memref<?xf32>
  return
}
