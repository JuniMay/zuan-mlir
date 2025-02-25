//RUN: zuan-opt -convert-vp-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @vp_addf16
func.func @vp_addf16(%a: vector<[4]xf16>, %b: vector<[4]xf16>, %evl: index) -> vector<[4]xf16> {
  // CHECK: %[[ADD:.+]] = "llvm.intr.vp.fadd"
  %c = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = arith.addf %a, %b : vector<[4]xf16>
    vector.yield %0 : vector<[4]xf16>
  } : vector<[4]xf16>
  // CHECK-NEXT: return %[[ADD]]
  return %c : vector<[4]xf16>
}

// CHECK-LABEL: func.func @vp_addf32
func.func @vp_addf32(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %evl: index) -> vector<[4]xf32> {
  // CHECK: %[[ADD:.+]] = "llvm.intr.vp.fadd"
  %c = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = arith.addf %a, %b : vector<[4]xf32>
    vector.yield %0 : vector<[4]xf32>
  } : vector<[4]xf32>
  // CHECK-NEXT: return %[[ADD]]
  return %c : vector<[4]xf32>
}

// CHECK-LABEL: func.func @vp_mulf32
func.func @vp_mulf32(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %evl: index) -> vector<[4]xf32> {
  // CHECK: %[[MUL:.+]] = "llvm.intr.vp.fmul"
  %c = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = arith.mulf %a, %b : vector<[4]xf32>
    vector.yield %0 : vector<[4]xf32>
  } : vector<[4]xf32>
  // CHECK-NEXT: return %[[MUL]]
  return %c : vector<[4]xf32>
}

// CHECK-LABEL: func.func @vp_maximumf32
func.func @vp_maximumf32(%a: vector<[4]xf32>, %b: vector<[4]xf32>, %evl: index) -> vector<[4]xf32> {
  // CHECK: %[[MAX:.+]] = llvm.call_intrinsic "llvm.vp.maximum"
  %c = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = arith.maximumf %a, %b : vector<[4]xf32>
    vector.yield %0 : vector<[4]xf32>
  } : vector<[4]xf32>
  // CHECK-NEXT: return %[[MAX]]
  return %c : vector<[4]xf32>
}
