//RUN: zuan-opt -convert-vp-to-llvm=enable-rvv=true %s | FileCheck %s

// CHECK-LABEL: func.func @getvl4xf32
func.func @getvl4xf32(%avl: index) -> index {
  // CHECK: %[[AVL:.+]] = llvm.trunc %{{.*}} : i64 to i32
  // CHECK: %[[LMUL:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[SEW:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: %[[VL:.+]] = "vp.intr.rvv.vsetvli"(%[[AVL]], %[[SEW]], %[[LMUL]]) : (i32, i32, i32) -> i32
  %evl = vp.getvl %avl, 4 x f32, true : index
  // CHECK-NEXT: %[[ZEXT:.+]] = llvm.zext %[[VL]] : i32 to i64
  return %evl : index
}

// CHECK: func.func @getvl4xf32_i32(%[[AVL:.+]]: i32) -> i32 {
func.func @getvl4xf32_i32(%avl: i32) -> i32 {
  // CHECK: %[[LMUL:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[SEW:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: %[[VL:.+]] = "vp.intr.rvv.vsetvli"(%[[AVL]], %[[SEW]], %[[LMUL]]) : (i32, i32, i32) -> i32
  %evl = vp.getvl %avl, 4 x f32, true : i32
  // CHECK-NEXT: return %[[VL]]
  return %evl : i32
}
