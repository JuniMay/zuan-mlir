//RUN: zuan-opt -convert-vp-to-llvm=enable-rvv=true %s | FileCheck %s

// CHECK-LABEL: func.func @getvl4xf32
func.func @getvl4xf32(%avl: index) -> index {
  // CHECK: %[[LMUL:.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[SEW:.+]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-NEXT: %[[VL:.+]] = "vp.intr.rvv.vsetvli"(%{{.*}}, %[[SEW]], %[[LMUL]]) : (i64, i64, i64) -> i64
  %evl = vp.getvl %avl, 4 x f32, true : index
  return %evl : index
}

// CHECK: func.func @getvl4xf32_i32(%[[AVL:.+]]: i32) -> i32 {
func.func @getvl4xf32_i32(%avl: i32) -> i32 {
  // CHECK: %[[ZEXT:.+]] = llvm.zext %[[AVL]] : i32 to i64
  // CHECK: %[[LMUL:.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[SEW:.+]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-NEXT: %[[VL:.+]] = "vp.intr.rvv.vsetvli"(%[[ZEXT]], %[[SEW]], %[[LMUL]]) : (i64, i64, i64) -> i64
  %evl = vp.getvl %avl, 4 x f32, true : i32
  // CHECK-NEXT: %[[TRUNC:.+]] = llvm.trunc %[[VL]] : i64 to i32
  // CHECK-NEXT: return %[[TRUNC]]
  return %evl : i32
}
