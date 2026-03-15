include_guard(GLOBAL)

#-------------------------------------------------------------------------------
# Cache configuration
#-------------------------------------------------------------------------------

set(DYNO_HOST_TOOLS_DIR "" CACHE PATH
    "Path to host-built Dyno tools used while cross-compiling target binaries")
set(DYNO_HOST_LLVM_TOOLS_DIR "" CACHE PATH
    "Path to host LLVM tools used by Dyno benchmark and regression lowering")
set(DYNO_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}" CACHE PATH
    "QEMU_LD_PREFIX used when target-built helper tools run through the emulator")
set(DYNO_QEMU_CPU "$ENV{QEMU_CPU}" CACHE STRING
    "QEMU_CPU used when target-built helper tools run through the emulator")

#-------------------------------------------------------------------------------
# Target execution wrappers
#-------------------------------------------------------------------------------

# Wrap a target executable so tests and helper commands can run in both native
# and cross builds.
#
# Arguments:
# - `out_var`: caller-scope variable that receives the final command list.
# - `executable`: target file path or generator expression to execute.
#
# Behavior:
# - In native builds, the result is just the executable path.
# - In cross builds with `CMAKE_CROSSCOMPILING_EMULATOR`, the result is an
#   emulator command, optionally prefixed with `QEMU_LD_PREFIX` and `QEMU_CPU`.
function(dyno_wrap_target_command out_var executable)
  if (CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
    set(_dyno_env)
    if (NOT DYNO_QEMU_LD_PREFIX STREQUAL "")
      list(APPEND _dyno_env "QEMU_LD_PREFIX=${DYNO_QEMU_LD_PREFIX}")
    endif()
    if (NOT DYNO_QEMU_CPU STREQUAL "")
      list(APPEND _dyno_env "QEMU_CPU=${DYNO_QEMU_CPU}")
    endif()
    if (_dyno_env)
      set(_dyno_cmd
        ${CMAKE_COMMAND} -E env
        ${_dyno_env}
        ${CMAKE_CROSSCOMPILING_EMULATOR} "${executable}")
    else()
      set(_dyno_cmd ${CMAKE_CROSSCOMPILING_EMULATOR} "${executable}")
    endif()
  else()
    set(_dyno_cmd "${executable}")
  endif()
  set(${out_var} ${_dyno_cmd} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# Host tool resolution
#-------------------------------------------------------------------------------

# Resolve the host-side tools used by the kernel build helpers.
#
# Exports to the caller:
# - `DYNO_HOST_MLIR_OPT`
# - `DYNO_HOST_MLIR_TRANSLATE`
# - `DYNO_HOST_CLANG`
# - `DYNO_HOST_DYNO_OPT`
# - `DYNO_HOST_DYNO_TRANSLATE`
# - `DYNO_HOST_DYNO_OPT_CMD`
# - `DYNO_HOST_DYNO_TRANSLATE_CMD`
#
# Resolution policy:
# - Cross builds are explicit-only. The host LLVM and Dyno tool directories must
#   be provided through `DYNO_HOST_LLVM_TOOLS_DIR` and `DYNO_HOST_TOOLS_DIR`.
# - Native builds prefer explicit directories, then in-tree tool targets, then
#   derived LLVM bin directories, and finally fall back to `PATH`.
#
# In cross builds, the `*_CMD` outputs may be wrapped through QEMU so later
# custom commands can still execute the tools from the build tree.
function(dyno_configure_host_tools)
  set(_dyno_host_clang "${CMAKE_C_COMPILER}")
  if (NOT _dyno_host_clang OR NOT EXISTS "${_dyno_host_clang}")
    message(FATAL_ERROR
      "Unable to locate the configured host compiler `${CMAKE_C_COMPILER}`.")
  endif()

  if (CMAKE_CROSSCOMPILING)
    if (NOT DYNO_HOST_LLVM_TOOLS_DIR)
      message(FATAL_ERROR
        "Cross builds require DYNO_HOST_LLVM_TOOLS_DIR so host-side `mlir-opt` "
        "and `mlir-translate` are explicit.")
    endif()
    if (NOT DYNO_HOST_TOOLS_DIR)
      message(FATAL_ERROR
        "Cross builds require DYNO_HOST_TOOLS_DIR so host-side `dyno-opt` and "
        "`dyno-translate` are explicit.")
    endif()

    set(_dyno_host_mlir_opt "${DYNO_HOST_LLVM_TOOLS_DIR}/mlir-opt")
    set(_dyno_host_mlir_translate "${DYNO_HOST_LLVM_TOOLS_DIR}/mlir-translate")
    if (NOT EXISTS "${_dyno_host_mlir_opt}")
      message(FATAL_ERROR
        "DYNO_HOST_LLVM_TOOLS_DIR is set to `${DYNO_HOST_LLVM_TOOLS_DIR}`, "
        "but `mlir-opt` was not found there.")
    endif()
    if (NOT EXISTS "${_dyno_host_mlir_translate}")
      message(FATAL_ERROR
        "DYNO_HOST_LLVM_TOOLS_DIR is set to `${DYNO_HOST_LLVM_TOOLS_DIR}`, "
        "but `mlir-translate` was not found there.")
    endif()

    foreach(_tool IN ITEMS dyno-opt dyno-translate)
      if (NOT EXISTS "${DYNO_HOST_TOOLS_DIR}/${_tool}")
        message(FATAL_ERROR
          "DYNO_HOST_TOOLS_DIR is set to `${DYNO_HOST_TOOLS_DIR}`, "
          "but `${_tool}` was not found there.")
      endif()
    endforeach()
    set(_dyno_host_dyno_opt "${DYNO_HOST_TOOLS_DIR}/dyno-opt")
    set(_dyno_host_dyno_translate "${DYNO_HOST_TOOLS_DIR}/dyno-translate")
    set(_dyno_host_dyno_opt_cmd "${_dyno_host_dyno_opt}")
    set(_dyno_host_dyno_translate_cmd "${_dyno_host_dyno_translate}")
  else()
    if (DYNO_HOST_LLVM_TOOLS_DIR)
      set(_dyno_host_mlir_opt "${DYNO_HOST_LLVM_TOOLS_DIR}/mlir-opt")
      set(_dyno_host_mlir_translate "${DYNO_HOST_LLVM_TOOLS_DIR}/mlir-translate")
    elseif (MLIR_TOOLS_DIR)
      set(_dyno_host_mlir_opt "${MLIR_TOOLS_DIR}/mlir-opt")
      set(_dyno_host_mlir_translate "${MLIR_TOOLS_DIR}/mlir-translate")
    elseif (LLVM_TOOLS_BINARY_DIR)
      set(_dyno_host_mlir_opt "${LLVM_TOOLS_BINARY_DIR}/mlir-opt")
      set(_dyno_host_mlir_translate "${LLVM_TOOLS_BINARY_DIR}/mlir-translate")
    elseif (DEFINED LLVM_BINARY_DIR AND EXISTS "${LLVM_BINARY_DIR}/bin/mlir-opt")
      set(_dyno_host_mlir_opt "${LLVM_BINARY_DIR}/bin/mlir-opt")
      set(_dyno_host_mlir_translate "${LLVM_BINARY_DIR}/bin/mlir-translate")
    else()
      find_program(_dyno_host_mlir_opt NAMES mlir-opt)
      find_program(_dyno_host_mlir_translate NAMES mlir-translate)
    endif()

    if (NOT _dyno_host_mlir_opt OR NOT EXISTS "${_dyno_host_mlir_opt}")
      message(FATAL_ERROR
        "Unable to locate required host tool `mlir-opt`. "
        "Set DYNO_HOST_LLVM_TOOLS_DIR or adjust PATH.")
    endif()
    if (NOT _dyno_host_mlir_translate OR NOT EXISTS "${_dyno_host_mlir_translate}")
      message(FATAL_ERROR
        "Unable to locate required host tool `mlir-translate`. "
        "Set DYNO_HOST_LLVM_TOOLS_DIR or adjust PATH.")
    endif()

    if (DYNO_HOST_TOOLS_DIR)
      foreach(_tool IN ITEMS dyno-opt dyno-translate)
        if (NOT EXISTS "${DYNO_HOST_TOOLS_DIR}/${_tool}")
          message(FATAL_ERROR
            "DYNO_HOST_TOOLS_DIR is set to `${DYNO_HOST_TOOLS_DIR}`, "
            "but `${_tool}` was not found there.")
        endif()
      endforeach()
      set(_dyno_host_dyno_opt "${DYNO_HOST_TOOLS_DIR}/dyno-opt")
      set(_dyno_host_dyno_translate "${DYNO_HOST_TOOLS_DIR}/dyno-translate")
      set(_dyno_host_dyno_opt_cmd "${_dyno_host_dyno_opt}")
      set(_dyno_host_dyno_translate_cmd "${_dyno_host_dyno_translate}")
    elseif (TARGET dyno-opt AND TARGET dyno-translate)
      set(_dyno_host_dyno_opt "$<TARGET_FILE:dyno-opt>")
      set(_dyno_host_dyno_translate "$<TARGET_FILE:dyno-translate>")
      dyno_wrap_target_command(_dyno_host_dyno_opt_cmd
                               "${_dyno_host_dyno_opt}")
      dyno_wrap_target_command(_dyno_host_dyno_translate_cmd
                               "${_dyno_host_dyno_translate}")
    else()
      find_program(_dyno_host_dyno_opt NAMES dyno-opt)
      find_program(_dyno_host_dyno_translate NAMES dyno-translate)
      if (NOT _dyno_host_dyno_opt)
        message(FATAL_ERROR
          "Unable to locate required host tool `dyno-opt`. "
          "Set DYNO_HOST_TOOLS_DIR or adjust PATH.")
      endif()
      if (NOT _dyno_host_dyno_translate)
        message(FATAL_ERROR
          "Unable to locate required host tool `dyno-translate`. "
          "Set DYNO_HOST_TOOLS_DIR or adjust PATH.")
      endif()
      set(_dyno_host_dyno_opt_cmd "${_dyno_host_dyno_opt}")
      set(_dyno_host_dyno_translate_cmd "${_dyno_host_dyno_translate}")
    endif()
  endif()

  if (NOT _dyno_host_dyno_opt)
      message(FATAL_ERROR
        "dyno_configure_host_tools produced an empty dyno-opt path.")
  endif()
  if (NOT _dyno_host_dyno_translate)
    message(FATAL_ERROR
      "dyno_configure_host_tools produced an empty dyno-translate path.")
  endif()

  set(DYNO_HOST_MLIR_OPT "${_dyno_host_mlir_opt}" PARENT_SCOPE)
  set(DYNO_HOST_MLIR_TRANSLATE "${_dyno_host_mlir_translate}" PARENT_SCOPE)
  set(DYNO_HOST_CLANG "${_dyno_host_clang}" PARENT_SCOPE)

  set(DYNO_HOST_DYNO_OPT "${_dyno_host_dyno_opt}" PARENT_SCOPE)
  set(DYNO_HOST_DYNO_TRANSLATE "${_dyno_host_dyno_translate}" PARENT_SCOPE)
  set(DYNO_HOST_DYNO_OPT_CMD ${_dyno_host_dyno_opt_cmd} PARENT_SCOPE)
  set(DYNO_HOST_DYNO_TRANSLATE_CMD ${_dyno_host_dyno_translate_cmd}
      PARENT_SCOPE)
endfunction()
