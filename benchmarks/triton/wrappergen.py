def ty_to_cpp(ty):
    if ty[0] == "*":
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def generate_kernel_wrapper(kernel_name: str, signature: str | dict) -> str:
    """Generate a C++ wrapper function for a Triton kernel."""

    if isinstance(signature, dict):
        signature = signature.values()
    else:
        assert isinstance(signature, str)
        signature = signature.split(",")

    triton_cpu_arg_decls = ", ".join(
        f"{ty_to_cpp(ty)} arg_{i}"
        for i, ty in enumerate(signature)
        if ty != "constexpr"
    )
    triton_cpu_args = ", ".join(
        f"arg_{i}" for i, ty in enumerate(signature) if ty != "constexpr"
    )
    triton_cpu_arg_types = ", ".join(
        f"{ty_to_cpp(ty)}" for ty in signature if ty != "constexpr"
    )
    triton_shared_arg_decls = ", ".join(
        (
            f"{ty_to_cpp(ty)} arg_{i}"
            if ty[0] != "*"
            else f"int64_t rank_{i}, {ty_to_cpp(ty)} arg_{i}"
        )
        for i, ty in enumerate(signature)
        if ty != "constexpr"
    )
    triton_shared_memrefs = "\n".join(
        f"StridedMemRefType<char, 0> memref_arg_{i} = {{static_cast<char *>(arg_{i}), static_cast<char *>(arg_{i}), 0}};"
        for i, ty in enumerate(signature)
        if ty != "constexpr" and ty[0] == "*"
    )
    triton_shared_args = ", ".join(
        f"arg_{i}" if ty[0] != "*" else f"0, &memref_arg_{i}"
        for i, ty in enumerate(signature)
        if ty != "constexpr"
    )

    return f"""
#include <cstdint>
#include <memory>
#include <omp.h>
#include "mlir/ExecutionEngine/CRunnerUtils.h"

extern "C" {{
  void {kernel_name}_triton_cpu({triton_cpu_arg_decls}, uint32_t x, uint32_t y, uint32_t z, uint32_t gridX, uint32_t gridY, uint32_t gridZ);
  void {kernel_name}_zuan({triton_shared_arg_decls}, uint32_t x, uint32_t y, uint32_t z, uint32_t gridX, uint32_t gridY, uint32_t gridZ);
}}

void {kernel_name}_zuan_wrapper({triton_cpu_arg_decls}, uint32_t x, uint32_t y, uint32_t z, uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  {triton_shared_memrefs}
  // The grids order are different between Triton-CPU and Triton-Shared
  {kernel_name}_zuan({triton_shared_args}, gridX, gridY, gridZ, x, y, z);
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

using kernel_ptr_t = void(*)({triton_cpu_arg_types}, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

void launch_kernel(uint32_t gridX, uint32_t gridY, uint32_t gridZ, kernel_ptr_t kernel, {triton_cpu_arg_decls}) {{
  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  size_t N = gridX * gridY * gridZ;

  auto omp_max_threads = omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(omp_max_threads)
  for (size_t i = 0; i < N; ++i) {{
    const auto [x, y, z] = all_grids[i];
    kernel({triton_cpu_args}, x, y, z, gridX, gridY, gridZ);
  }}
}}
"""
