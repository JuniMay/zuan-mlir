//RUN: dyno-opt -convert-vp-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @vp_splatf32
func.func @vp_splatf32(%a: f32, %evl: index) -> vector<[4]xf32> {
  // CHECK: %[[SPLAT:.+]] = llvm.shufflevector
  %c = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = vector.broadcast %a : f32 to vector<[4]xf32>
    vector.yield %0 : vector<[4]xf32>
  } : vector<[4]xf32>
  // CHECK-NEXT: return %[[SPLAT]]
  return %c : vector<[4]xf32>
}
