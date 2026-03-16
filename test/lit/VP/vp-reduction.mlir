// RUN: dyno-opt -convert-vp-to-llvm %s \
// RUN: | tee %t.llvm.mlir \
// RUN: | FileCheck %s \
// RUN: && mv -f %t.llvm.mlir \
// RUN:   $(dirname %t)/vp-reduction.lowered.mlir

// CHECK-LABEL: func.func @vp_reductionf32
func.func @vp_reductionf32(%a: vector<[4]xf32>, %init: f32, %evl: index) -> f32 {
  // CHECK: %[[ADD:.+]] = llvm.call_intrinsic "llvm.vp.reduce.fadd"
  // CHECK-SAME: fastmathFlags = #llvm.fastmath<reassoc>
  %c = vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    %0 = vector.reduction <add>, %a, %init fastmath<reassoc> : vector<[4]xf32> into f32
    vector.yield %0 : f32
  } : f32
  // CHECK-NEXT: return %[[ADD]]
  return %c : f32
}

// CHECK-LABEL: func.func @vp_reductionf32_noreassoc
func.func @vp_reductionf32_noreassoc(%a: vector<[4]xf32>, %init: f32,
                                     %evl: index) -> f32 {
  // CHECK: %[[ADD:.+]] = llvm.call_intrinsic "llvm.vp.reduce.fadd"
  // CHECK-NOT: fastmathFlags = #llvm.fastmath<reassoc>
  %c = vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    %0 = vector.reduction <add>, %a, %init : vector<[4]xf32> into f32
    vector.yield %0 : f32
  } : f32
  // CHECK-NEXT: return %[[ADD]]
  return %c : f32
}

// CHECK-LABEL: func.func @vp_reductionsmax
func.func @vp_reductionsmax(%a: vector<[4]xi32>, %init: i32, %evl: index) -> i32 {
  // CHECK: %[[MAX:.+]] = llvm.call_intrinsic "llvm.vp.reduce.smax"
  %c = vp.predicate %evl : index, mask = none, passthru = none, maskedoff = none {
    %0 = vector.reduction <maxsi>, %a, %init : vector<[4]xi32> into i32
    vector.yield %0 : i32
  } : i32
  // CHECK-NEXT: return %[[MAX]]
  return %c : i32
}
