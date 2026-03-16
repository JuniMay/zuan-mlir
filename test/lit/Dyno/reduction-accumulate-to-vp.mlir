// RUN: dyno-opt %s \
// RUN:   -convert-dyno-to-vp='vf=8 scalable=true' \
// RUN: | tee %t.vp.mlir \
// RUN: | FileCheck %s \
// RUN: && mv -f %t.vp.mlir \
// RUN:   $(dirname %t)/reduction-accumulate-to-vp.lowered.mlir

func.func @reduction_accumulate_passthru(%acc: memref<8xf32>,
                                         %chunk: memref<?xf32>,
                                         %dst: memref<8xf32>) {
  %acc_tile = dyno.load %acc : memref<8xf32>
  %chunk_tile = dyno.load %chunk : memref<?xf32>
  %res = dyno.reduction_accumulate <add> %acc_tile, %chunk_tile :
      !dyno.tile<8xf32>, !dyno.tile<?xf32>
  dyno.store %res, %dst : !dyno.tile<8xf32>, memref<8xf32>
  return
}

// CHECK-LABEL: func.func @reduction_accumulate_passthru
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[ACC_VEC:.*]] = vp.predicate %[[C8]] : index, mask = none, passthru = none, maskedoff = none {
// CHECK: vp.load
// CHECK: %[[CHUNK_DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
// CHECK: %[[CHUNK_VEC:.*]] = vp.predicate %[[CHUNK_DIM]] : index, mask = none, passthru = none, maskedoff = none {
// CHECK: vp.load
// CHECK-NOT: vp.step
// CHECK-NOT: vp.intr.icmp
// CHECK: %[[CHUNK_DIM2:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
// CHECK: %[[UPDATED:.*]] = vp.predicate %[[CHUNK_DIM2]] : index, mask = none, passthru = %[[ACC_VEC]] : vector<[8]xf32>, maskedoff = none {
// CHECK: arith.addf %[[ACC_VEC]], %[[CHUNK_VEC]] : vector<[8]xf32>

func.func @masked_reduction_accumulate_passthru(%acc: memref<8xf32>,
                                                %chunk: memref<?xf32>,
                                                %mask: memref<8xi1>,
                                                %maskedoff: memref<8xf32>,
                                                %dst: memref<8xf32>) {
  %acc_tile = dyno.load %acc : memref<8xf32>
  %chunk_tile = dyno.load %chunk : memref<?xf32>
  %mask_tile = dyno.load %mask : memref<8xi1>
  %maskedoff_tile = dyno.load %maskedoff : memref<8xf32>
  %res = dyno.mask %mask_tile : !dyno.tile<8xi1>, %maskedoff_tile : !dyno.tile<8xf32> {
    %updated = dyno.reduction_accumulate <add> %acc_tile, %chunk_tile :
        !dyno.tile<8xf32>, !dyno.tile<?xf32>
    dyno.mask_yield %updated : !dyno.tile<8xf32>
  } : !dyno.tile<8xf32>
  dyno.store %res, %dst : !dyno.tile<8xf32>, memref<8xf32>
  return
}

// CHECK-LABEL: func.func @masked_reduction_accumulate_passthru
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[MASK_VEC:.*]] = vp.predicate %[[C8]] : index, mask = none, passthru = none, maskedoff = none {
// CHECK: vp.load
// CHECK: %[[MASKEDOFF_VEC:.*]] = vp.predicate %[[C8]] : index, mask = none, passthru = none, maskedoff = none {
// CHECK: vp.load
// CHECK: %[[ACC_VEC:.*]] = vp.predicate %[[C8]] : index, mask = %[[MASK_VEC]] : vector<[8]xi1>, passthru = none, maskedoff = %[[MASKEDOFF_VEC]] : vector<[8]xf32> {
// CHECK: vp.load
// CHECK: %[[CHUNK_DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
// CHECK: %[[CHUNK_VEC:.*]] = vp.predicate %[[CHUNK_DIM]] : index, mask = %[[MASK_VEC]] : vector<[8]xi1>, passthru = none, maskedoff = %[[MASKEDOFF_VEC]] : vector<[8]xf32> {
// CHECK: vp.load
// CHECK: %[[CHUNK_DIM2:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
// CHECK: %[[UPDATED:.*]] = vp.predicate %[[CHUNK_DIM2]] : index, mask = %[[MASK_VEC]] : vector<[8]xi1>, passthru = %[[ACC_VEC]] : vector<[8]xf32>, maskedoff = %[[MASKEDOFF_VEC]] : vector<[8]xf32> {
// CHECK: arith.addf %[[ACC_VEC]], %[[CHUNK_VEC]] : vector<[8]xf32>
