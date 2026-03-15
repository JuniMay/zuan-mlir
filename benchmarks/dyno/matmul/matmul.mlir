func.func @matmul(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %m = memref.dim %a, %c0 : memref<?x?xf32>
  %n = memref.dim %b, %c1 : memref<?x?xf32>
  %k = memref.dim %a, %c1 : memref<?x?xf32>

  %a_rows = memref.dim %a, %c0 : memref<?x?xf32>
  %a_cols = memref.dim %a, %c1 : memref<?x?xf32>
  %a_expand = memref.expand_shape %a [[0, 1], [2]] output_shape [1, %a_rows, %a_cols]
      : memref<?x?xf32> into memref<1x?x?xf32>
  %a_transpose = memref.transpose %a_expand (d0, d1, d2) -> (d1, d0, d2)
      : memref<1x?x?xf32> to memref<?x1x?xf32, strided<[?, ?, 1]>>
  %a_common = memref.subview %a_transpose[0, 0, 0] [%m, %n, %k] [1, 0, 1]
      : memref<?x1x?xf32, strided<[?, ?, 1]>> to memref<?x?x?xf32, strided<[?, 0, 1]>>

  %b_rows = memref.dim %b, %c0 : memref<?x?xf32>
  %b_cols = memref.dim %b, %c1 : memref<?x?xf32>
  %b_expand = memref.expand_shape %b [[0, 1], [2]] output_shape [1, %b_rows, %b_cols]
      : memref<?x?xf32> into memref<1x?x?xf32>
  %b_transpose = memref.transpose %b_expand (d0, d1, d2) -> (d0, d2, d1)
      : memref<1x?x?xf32> to memref<1x?x?xf32, strided<[?, 1, ?]>>
  %b_common = memref.subview %b_transpose[0, 0, 0] [%m, %n, %k] [0, 1, 1]
      : memref<1x?x?xf32, strided<[?, 1, ?]>> to memref<?x?x?xf32, strided<[0, 1, ?]>>

  %c_view = memref.transpose %c (d0, d1) -> (d0, d1)
      : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>

  %a_tile = dyno.load %a_common : memref<?x?x?xf32, strided<[?, 0, 1]>>
  %b_tile = dyno.load %b_common : memref<?x?x?xf32, strided<[0, 1, ?]>>
  %c_tile = dyno.load %c_view : memref<?x?xf32, strided<[?, 1]>>
  %mul = arith.mulf %a_tile, %b_tile : !dyno.tile<?x?x?xf32>
  %red = dyno.reduction <add> %mul [2], %c_tile :
      !dyno.tile<?x?x?xf32>, !dyno.tile<?x?xf32>
  dyno.store %red, %c_view : !dyno.tile<?x?xf32>, memref<?x?xf32, strided<[?, 1]>>
  return
}
