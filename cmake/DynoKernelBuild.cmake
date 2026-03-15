include_guard(GLOBAL)

include(DynoHostTools)

#-------------------------------------------------------------------------------
# Shared kernel build configuration
#-------------------------------------------------------------------------------

# Guard helper for modules that expect `dyno_configure_kernel_build()` to have
# run already.
#
# This checks the global property written by the public configuration entry
# point and fails early with a clear message if a lowering helper is used
# without the shared toolchain state being initialized first.
function(_dyno_require_kernel_build_configured)
  get_property(_dyno_kernel_build_configured
               GLOBAL PROPERTY DYNO_KERNEL_BUILD_CONFIGURED)
  if (NOT _dyno_kernel_build_configured)
    message(FATAL_ERROR
      "Call dyno_configure_kernel_build() before using Dyno kernel build "
      "helpers.")
  endif()
endfunction()

# Initialize the shared kernel-build state used by benchmarks and runtime
# regressions.
#
# Work performed:
# - resolves host lowering tools via `dyno_configure_host_tools()`
# - canonicalizes the target compile flags from `CMAKE_C_FLAGS`
# - stores the result in global properties for later helper functions
# - re-exports the resolved tool paths to the caller's scope for convenience
#
# This function is intentionally side-effectful and should be called once near
# the top of any directory tree that builds lowered kernel artifacts.
function(dyno_configure_kernel_build)
  dyno_configure_host_tools()

  separate_arguments(_dyno_kernel_clang_flags UNIX_COMMAND "${CMAKE_C_FLAGS}")

  set_property(GLOBAL PROPERTY DYNO_KERNEL_BUILD_CONFIGURED TRUE)
  set_property(GLOBAL PROPERTY DYNO_KERNEL_CLANG_FLAGS
               "${_dyno_kernel_clang_flags}")

  foreach(_tool_var IN ITEMS
      DYNO_HOST_MLIR_OPT
      DYNO_HOST_MLIR_TRANSLATE
      DYNO_HOST_CLANG
      DYNO_HOST_DYNO_OPT
      DYNO_HOST_DYNO_TRANSLATE
      DYNO_HOST_DYNO_OPT_CMD
      DYNO_HOST_DYNO_TRANSLATE_CMD)
    set(${_tool_var} "${${_tool_var}}" PARENT_SCOPE)
  endforeach()
endfunction()

# Read back the canonical target compile flags prepared by
# `dyno_configure_kernel_build()`.
#
# Arguments:
# - `out_var`: caller-scope variable that receives the split compiler flag list.
#
# The returned list is used by `dyno_compile_llvm_ir_to_object()` so individual
# benchmark and regression definitions only need to specify local optimization
# flags such as `-O3` or `-g`.
function(dyno_get_kernel_compile_flags out_var)
  _dyno_require_kernel_build_configured()
  get_property(_dyno_kernel_clang_flags
               GLOBAL PROPERTY DYNO_KERNEL_CLANG_FLAGS)
  set(${out_var} ${_dyno_kernel_clang_flags} PARENT_SCOPE)
endfunction()
