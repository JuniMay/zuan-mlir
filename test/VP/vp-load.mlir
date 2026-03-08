//RUN: zuan-opt -convert-vp-to-llvm=enable-rvv=true %s | FileCheck %s

// CHECK-LABEL: func.func @vp_load
func.func @vp_load(%mem: memref<?x?xf32>, %i: index, %j: index, %evl: index) -> vector<[4]xf32> {
  // CHECK: "llvm.intr.vp.load"
  %res = vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    %res = vp.load %mem[%i, %j] : memref<?x?xf32> , vector<[4]xf32>
    vector.yield %res : vector<[4]xf32>
  } : vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// CHECK-LABEL: func.func @vp_load_strided
func.func @vp_load_strided(%mem: memref<?x?xf32, strided<[?, 3]>>, %i: index, %j: index, %evl: index) -> vector<[4]xf32> {
  // CHECK: "llvm.intr.experimental.vp.strided.load"
  %res = vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    %res = vp.load %mem[%i, %j] : memref<?x?xf32, strided<[?, 3]>>, vector<[4]xf32>
    vector.yield %res : vector<[4]xf32>
  } : vector<[4]xf32>
  return %res : vector<[4]xf32>
}

// CHECK-LABEL: func.func @vp_load_masked_i1
func.func @vp_load_masked_i1(%mem: memref<?xi1>, %i: index, %evl: index, %mask: vector<[8]xi1>) -> vector<[8]xi1> {
  // CHECK: "llvm.intr.vp.load"
  // CHECK-NOT: vp.rvv_intr.vlm
  %res = vp.predicate %evl : index, mask = %mask : vector<[8]xi1>, passthru = none, maskedoff = none {
    %vec = vp.load %mem[%i] : memref<?xi1>, vector<[8]xi1>
    vector.yield %vec : vector<[8]xi1>
  } : vector<[8]xi1>
  return %res : vector<[8]xi1>
}
