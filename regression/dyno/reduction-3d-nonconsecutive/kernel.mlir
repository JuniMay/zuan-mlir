func.func @kernel(%src: memref<?x?x?xi32>, %partial_dst: memref<?xi32>,
                  %final_dst: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %final_dim = memref.dim %final_dst, %c0 : memref<?xi32>
  %tile = dyno.load %src : memref<?x?x?xi32>
  %partial = dyno.reduction <add> %tile [0, 2] : !dyno.tile<?x?x?xi32>
  dyno.store %partial, %partial_dst : !dyno.tile<?xi32>, memref<?xi32>
  %reduced = dyno.reduction <add> %partial [0] : !dyno.tile<?xi32>
  %stored = dyno.splat %reduced [%final_dim] : !dyno.tile<i32>
  dyno.store %stored, %final_dst : !dyno.tile<?xi32>, memref<?xi32>
  return
}
