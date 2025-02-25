//RUN: zuan-opt -convert-vp-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @vp_reductionf32
func.func @vp_reductionf32(%a: vector<[4]xf32>, %evl: index) -> f32 {
  // CHECK: %[[ADD:.+]] = llvm.call_intrinsic "llvm.vp.reduce.fadd"
  %c = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = vector.reduction <add>, %a : vector<[4]xf32> into f32
    vector.yield %0 : f32
  } : f32
  // CHECK-NEXT: return %[[ADD]]
  return %c : f32
}

