//RUN: zuan-opt -convert-vp-to-llvm=enable-rvv=true %s | FileCheck %s

// CHECK-LABEL: func.func @vp_store
func.func @vp_store(%vec: vector<[4]xf32>, %mem: memref<?x?xf32>, %i: index, %j: index, %evl: index) {
  // CHECK: "llvm.intr.vp.store"
  vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    vp.store %vec, %mem[%i, %j] : vector<[4]xf32>, memref<?x?xf32>
  }
  return
}

// CHECK-LABEL: func.func @vp_store_strided
func.func @vp_store_strided(%vec: vector<[4]xf32>, %mem: memref<?x?xf32, strided<[?, 3]>>, %i: index, %j: index, %evl: index) {
  // CHECK: "llvm.intr.experimental.vp.strided.store"
  vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    vp.store %vec, %mem[%i, %j] : vector<[4]xf32>, memref<?x?xf32, strided<[?, 3]>>
  }
  return
}
