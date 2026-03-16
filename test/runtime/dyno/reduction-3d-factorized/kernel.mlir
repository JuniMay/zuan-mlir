func.func @kernel(%src: memref<?x?x?xi32>, %dst: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %dst_dim = memref.dim %dst, %c0 : memref<?xi32>
  %tile = dyno.load %src : memref<?x?x?xi32>
  %reduced = dyno.reduction <add> %tile [0, 1, 2] : !dyno.tile<?x?x?xi32>
  %stored = dyno.splat %reduced [%dst_dim] : !dyno.tile<i32>
  dyno.store %stored, %dst : !dyno.tile<?xi32>, memref<?xi32>
  return
}
