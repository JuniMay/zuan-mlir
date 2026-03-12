import argparse
from wrappergen import generate_kernel_wrapper

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def kernel(
    q_ptr,
    k_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    sm_scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = q_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = k_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_iter * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    output = accumulator * sm_scale

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, output, mask=c_mask)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output", "-o", default="attention.ttir")
    argparser.add_argument("--BLOCK_N", type=int, default=128)
    args = argparser.parse_args()

    signature = {
        "q_ptr": "*fp32",
        "k_ptr": "*fp32",
        "output_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "sm_scale": "fp32",
        "BLOCK_SIZE_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_SIZE_K": "constexpr",
        "GROUP_SIZE_M": "constexpr",
    }

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature=signature,
        constexprs={
            "BLOCK_SIZE_M": 64,
            "BLOCK_N": args.BLOCK_N,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 4,
        },
    )
    compiled = triton.compile(src)

    with open(args.output, "w") as f:
        f.write(compiled.asm["ttir"])

    with open(args.output.replace(".ttir", ".h"), "w") as f:
        f.write(generate_kernel_wrapper("kernel", signature))


if __name__ == "__main__":
    main()
