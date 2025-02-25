//RUN: zuan-opt -convert-vp-to-llvm %s | FileCheck %s

func.func @vp_gather(%mem: memref<?x?xf32>, %idx0: vector<[4]xindex>, %idx1: vector<[4]xindex>, %evl: index) -> vector<[4]xf32> {
  // CHECK: llvm.call_intrinsic "llvm.vp.gather"
  %res = vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    %res = vp.gather %mem[%idx0, %idx1 : vector<[4]xindex>, vector<[4]xindex>] : memref<?x?xf32>, vector<[4]xf32>
    vector.yield %res : vector<[4]xf32>
  } : vector<[4]xf32>
  return %res : vector<[4]xf32>
}

func.func @vp_scatter(%mem: memref<?x?xf32>, %idx0: vector<[4]xindex>, %idx1: vector<[4]xindex>, %vec: vector<[4]xf32>, %evl: index) {
  // CHECK: llvm.call_intrinsic "llvm.vp.scatter"
  vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    vp.scatter %vec, %mem[%idx0, %idx1 : vector<[4]xindex>, vector<[4]xindex>] : vector<[4]xf32>, memref<?x?xf32>
  }
  return
}
