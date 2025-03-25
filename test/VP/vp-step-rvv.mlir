//RUN: zuan-opt -convert-vp-to-llvm=enable-rvv=true %s | FileCheck %s

// CHECK-LABEL: func.func @vstep_unmasked
func.func @vstep_unmasked(%vl: index) -> vector<[8]xindex> {
  // CHECK: llvm.intr.stepvector
  %res = vp.predicate %vl : index, mask = none, passthru = none, maskedoff = none {
    %res = vector.step : vector<[8]xindex>
    vector.yield %res : vector<[8]xindex>
  } : vector<[8]xindex>
  return %res : vector<[8]xindex>
}

// CHECK-LABEL: func.func @vstep_masked
func.func @vstep_masked(%vl: index, %mask: vector<[8]xi1>) -> vector<[8]xindex> {
  // CHECK: %[[POLICY:.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: "vp.intr.rvv.vid_mask"(%{{.*}}, %{{.*}}, %{{.*}}, %[[POLICY]])
  %res = vp.predicate %vl : index, mask = %mask : vector<[8]xi1>, passthru = none, maskedoff = none {
    %res = vector.step : vector<[8]xindex>
    vector.yield %res : vector<[8]xindex>
  } : vector<[8]xindex>
  return %res : vector<[8]xindex>
}

// CHECK-LABEL: func.func @vstep_masked_tuma
func.func @vstep_masked_tuma(%vl: index, %mask: vector<[8]xi1>, %passthru: vector<[8]xindex>) -> vector<[8]xindex> {
  // CHECK: %[[POLICY:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: "vp.intr.rvv.vid_mask"(%{{.*}}, %{{.*}}, %{{.*}}, %[[POLICY]])
  %res = vp.predicate %vl : index, mask = %mask : vector<[8]xi1>, passthru = %passthru : vector<[8]xindex>, maskedoff = none {
    %res = vector.step : vector<[8]xindex>
    vector.yield %res : vector<[8]xindex>
  } : vector<[8]xindex>
  return %res : vector<[8]xindex>
}

// CHECK-LABEL: func.func @vstep_masked_tamu
func.func @vstep_masked_tamu(%vl: index, %mask: vector<[8]xi1>, %maskedoff: vector<[8]xindex>) -> vector<[8]xindex> {
  // CHECK: %[[POLICY:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: "vp.intr.rvv.vid_mask"(%{{.*}}, %{{.*}}, %{{.*}}, %[[POLICY]])
  %res = vp.predicate %vl : index, mask = %mask : vector<[8]xi1>, passthru = none, maskedoff = %maskedoff : vector<[8]xindex> {
    %res = vector.step : vector<[8]xindex>
    vector.yield %res : vector<[8]xindex>
  } : vector<[8]xindex>
  return %res : vector<[8]xindex>
}

// CHECK-LABEL: func.func @vstep_masked_tumu
func.func @vstep_masked_tumu(%vl: index, %mask: vector<[8]xi1>, %passthru: vector<[8]xindex>, %maskedoff: vector<[8]xindex>) -> vector<[8]xindex> {
  // CHECK: %[[POLICY:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: "vp.intr.rvv.vid_mask"(%{{.*}}, %{{.*}}, %{{.*}}, %[[POLICY]])
  %res = vp.predicate %vl : index, mask = %mask : vector<[8]xi1>, passthru = %passthru : vector<[8]xindex>, maskedoff = %maskedoff : vector<[8]xindex> {
    %res = vector.step : vector<[8]xindex>
    vector.yield %res : vector<[8]xindex>
  } : vector<[8]xindex>
  return %res : vector<[8]xindex>
}

