
// Dot product of two vectors of fp16 values into a fp32 accumulator.
func.func @kernel(%a: memref<?xf16>, %b: memref<?xf16>, %c: memref<f32>) {
  linalg.generic {
    indexing_maps = [
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>,
      affine_map<(i) -> ()>
    ],
    iterator_types = ["reduction"]
  } ins(%a, %b : memref<?xf16>, memref<?xf16>) outs(%c: memref<f32>) {
    ^bb0(%a_in: f16, %b_in: f16, %c_out: f32):
      %a_fp32 = arith.extf %a_in : f16 to f32
      %b_fp32 = arith.extf %b_in : f16 to f32
      %prod = arith.mulf %a_fp32, %b_fp32 : f32
      %add = arith.addf %c_out, %prod : f32
      linalg.yield %add : f32
    }
  return
}
