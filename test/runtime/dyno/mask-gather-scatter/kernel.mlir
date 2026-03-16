func.func @kernel(%src: memref<?x?xf32>, %idx0: memref<?xindex>,
                  %idx1: memref<?xindex>, %mask: memref<?xi1>,
                  %dst: memref<?xf32>) {
  %idx0_tile = dyno.load %idx0 : memref<?xindex>
  %idx1_tile = dyno.load %idx1 : memref<?xindex>
  %mask_tile = dyno.load %mask : memref<?xi1>
  %gathered = dyno.gather %src[%idx0_tile, %idx1_tile] :
      memref<?x?xf32>, !dyno.tile<?xindex>, !dyno.tile<?xindex>
  %masked = dyno.mask %mask_tile : !dyno.tile<?xi1> {
    dyno.mask_yield %gathered : !dyno.tile<?xf32>
  } : !dyno.tile<?xf32>
  dyno.scatter %masked, %dst[%idx0_tile] :
      !dyno.tile<?xf32>, memref<?xf32>, !dyno.tile<?xindex>
  return
}
