// RUN: zuan-opt -lower-zuan='target-rank=2' -zuan-stripmining='vf=8 scalable=true' %s | FileCheck %s

// CHECK-LABEL: func.func @softmax
func.func @softmax(%src: memref<?xf32>, %dst: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %src, %c0 : memref<?xf32>
  %vsrc = zuan.load %src : memref<?xf32>
  %max = zuan.reduction <maxnumf> %vsrc [0] : !zuan.tile<?xf32>
  %max_splat = zuan.splat %max [%dim] : !zuan.tile<f32>
  %sub = arith.subf %vsrc, %max_splat : !zuan.tile<?xf32>
  %exp = math.exp %sub : !zuan.tile<?xf32>
  %sum = zuan.reduction <add> %exp [0] : !zuan.tile<?xf32>
  %sum_splat = zuan.splat %sum [%dim] : !zuan.tile<f32>
  %vdst = arith.divf %exp, %sum_splat : !zuan.tile<?xf32>
  // CHECK: %[[MAX_LOOP:.*]]:3 = scf.while
  // CHECK: %[[MAX:.*]] = zuan.reduction <maxnumf> %[[MAX_LOOP]]#2 [0]
  // CHECK: %[[SUM_LOOP:.*]]:3 = scf.while
  // CHECK: %[[SUM:.*]] = zuan.reduction <add> %[[SUM_LOOP]]#2 [0]
  // CHECK: zuan.store %{{.*}}, %{{.*}}
  zuan.store %vdst, %dst : !zuan.tile<?xf32>, memref<?xf32>
  return
}
