//RUN: zuan-opt -convert-vp-to-llvm \
//RUN:          -convert-func-to-llvm \
//RUN:          -reconcile-unrealized-casts %s \
//RUN: | zuan-translate --zuan-to-llvmir | FileCheck %s

func.func @vp_fcmp(%lhs: vector<[4]xf32>, %rhs: vector<[4]xf32>, %evl: index) -> vector<[4]xi1> {
  // CHECK: @llvm.vp.fcmp.nxv4f32
  // CHECK: metadata !"olt"
  %res = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = arith.cmpf olt, %lhs, %rhs : vector<[4]xf32>
    vector.yield %0 : vector<[4]xi1>
  } : vector<[4]xi1>
  return %res : vector<[4]xi1>
}

func.func @vp_icmp(%lhs: vector<[4]xi32>, %rhs: vector<[4]xi32>, %evl: index) -> vector<[4]xi1> {
  // CHECK: @llvm.vp.icmp.nxv4i32
  // CHECK: metadata !"eq"
  %res = vp.predicate %evl : index , mask = none, passthru = none, maskedoff = none {
    %0 = arith.cmpi eq, %lhs, %rhs : vector<[4]xi32>
    vector.yield %0 : vector<[4]xi1>
  } : vector<[4]xi1>
  return %res : vector<[4]xi1>
}
