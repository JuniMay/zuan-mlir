import argparse
from wrappergen import generate_kernel_wrapper

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def kernel(alpha, x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = alpha * x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--BLOCK_SIZE", type=int, default=128)
    argparser.add_argument("--output", "-o", default="saxpy.ttir")
    args = argparser.parse_args()

    signature = {
        "alpha": "fp32",
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "output_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    }

    src = triton.compiler.ASTSource(
        fn=kernel,
        signature=signature,
        constexprs={"BLOCK_SIZE": args.BLOCK_SIZE},
    )
    compiled = triton.compile(src)

    with open(args.output, "w") as f:
        f.write(compiled.asm["ttir"])

    with open(args.output.replace(".ttir", ".h"), "w") as f:
        f.write(generate_kernel_wrapper("kernel", signature))


if __name__ == "__main__":
    main()
