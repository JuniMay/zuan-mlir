//RUN: zuan-opt -convert-vp-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @vp_splatf32
func.func @vp_splatf32(%a: f32, %evl: index) -> vector<[4]xf32> {
  // CHECK: %[[SPLAT:.+]] = llvm.call_intrinsic "llvm.experimental.vp.splat"
  %c = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = vector.splat %a : vector<[4]xf32>
    vector.yield %0 : vector<[4]xf32>
  } : vector<[4]xf32>
  // CHECK-NEXT: return %[[SPLAT]]
  return %c : vector<[4]xf32>
}
