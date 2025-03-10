
// `zp` for zero point
func.func @kernel(%a: memref<?x?xi8>, %b: memref<?x?xi8>, %a_zp: i32, %b_zp: i32, %c: memref<?x?xi64>) {
  linalg.quantized_matmul ins(%a, %b, %a_zp, %b_zp : memref<?x?xi8>, memref<?x?xi8>, i32, i32) outs(%c : memref<?x?xi64>)
  return
}
