func.func @kernel(%in : memref<?x?xf32>, %filter : memref<?x?xf32>, %out : memref<?x?xf32>) {
  linalg.conv_2d ins(%in, %filter : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out: memref<?x?xf32>)
  return
}
