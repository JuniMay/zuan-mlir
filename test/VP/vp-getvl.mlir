//RUN: zuan-opt -convert-vp-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @getvl4xf32
func.func @getvl4xf32(%avl: index) -> index {
  // CHECK: %[[AVL:.+]] = llvm.trunc %{{.*}} : i64 to i32
  // CHECK: %[[VF:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1
  // CHECK-NEXT: %[[VL:.+]] = llvm.call_intrinsic "llvm.experimental.get.vector.length"(%[[AVL]], %[[VF]], %[[TRUE]]) : (i32, i32, i1) -> i32
  %evl = vp.getvl %avl, 4 x f32, true : index
  // CHECK-NEXT: %[[ZEXT:.+]] = llvm.zext %[[VL]] : i32 to i64
  return %evl : index
}

// CHECK: func.func @getvl4xf32_i32(%[[AVL:.+]]: i32) -> i32 {
func.func @getvl4xf32_i32(%avl: i32) -> i32 {
  // CHECK: %[[VF:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1
  // CHECK-NEXT: %[[VL:.+]] = llvm.call_intrinsic "llvm.experimental.get.vector.length"(%[[AVL]], %[[VF]], %[[TRUE]]) : (i32, i32, i1) -> i32
  %evl = vp.getvl %avl, 4 x f32, true : i32
  // CHECK-NEXT: return %[[VL]]
  return %evl : i32
}

