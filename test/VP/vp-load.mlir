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
