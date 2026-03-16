func.func @kernel(%src: memref<?xf32>, %dst: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %dst_dim = memref.dim %dst, %c0 : memref<?xf32>
  %tile = dyno.load %src : memref<?xf32>
  %reduced = dyno.reduction <minimumf> %tile [0] : !dyno.tile<?xf32>
  %stored = dyno.splat %reduced [%dst_dim] : !dyno.tile<f32>
  dyno.store %stored, %dst : !dyno.tile<?xf32>, memref<?xf32>
  return
}
