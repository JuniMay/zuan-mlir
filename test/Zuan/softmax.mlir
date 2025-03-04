// RUN: zuan-opt -convert-linalg-to-zuan \
// RUN:          -lower-zuan='target-rank=2' \
// RUN:          -zuan-stripmining='vf=8 scalable=true' %s \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @softmax
func.func @softmax(%src: memref<?xf32>, %dst: memref<?xf32>) {
  %0 = arith.constant 0 : index
  %dim = memref.dim %src, %0 : memref<?xf32>

  zuan.dynamic {
    %vsrc = zuan.load %src : memref<?xf32>
    %max = zuan.reduction <maxnumf> %vsrc [0] : !zuan.tile<?xf32>

    %max_splat = zuan.splat %max [%dim] : !zuan.tile<f32>
    %sub = arith.subf %vsrc, %max_splat : !zuan.tile<?xf32>
    %exp = math.exp %sub : !zuan.tile<?xf32>
    %sum = zuan.reduction <add> %exp [0] : !zuan.tile<?xf32>

    %sum_splat = zuan.splat %sum [%dim] : !zuan.tile<f32>
    %vdst = arith.divf %exp, %sum_splat : !zuan.tile<?xf32>
    zuan.yield {
      zuan.store %vdst, %dst : !zuan.tile<?xf32>, memref<?xf32>
    }
  }

  return
}
