include_guard(GLOBAL)

include(CMakeParseArguments)

set(DYNO_HOST_TOOLS_DIR "" CACHE PATH
    "Path to host-built Dyno tools used while cross-compiling target binaries")
set(DYNO_HOST_LLVM_TOOLS_DIR "" CACHE PATH
    "Path to host LLVM tools used by Dyno benchmark and regression lowering")
set(DYNO_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}" CACHE PATH
    "QEMU_LD_PREFIX used when target-built helper tools run through the emulator")

function(_dyno_require_program out_var program)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs HINTS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  unset(_dyno_program CACHE)
  if (ARG_HINTS)
    find_program(_dyno_program NAMES "${program}" HINTS ${ARG_HINTS}
                 NO_DEFAULT_PATH)
  endif()
  if (NOT _dyno_program)
    find_program(_dyno_program NAMES "${program}" HINTS ${ARG_HINTS})
  endif()
  if (NOT _dyno_program)
    message(FATAL_ERROR
      "Unable to locate required host tool `${program}`. "
      "Set DYNO_HOST_LLVM_TOOLS_DIR, DYNO_HOST_TOOLS_DIR, or adjust PATH.")
  endif()
  set(${out_var} "${_dyno_program}" PARENT_SCOPE)
endfunction()

function(dyno_wrap_target_command out_var executable)
  if (CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
    if (NOT DYNO_QEMU_LD_PREFIX STREQUAL "")
      set(_dyno_cmd
        ${CMAKE_COMMAND} -E env "QEMU_LD_PREFIX=${DYNO_QEMU_LD_PREFIX}"
        ${CMAKE_CROSSCOMPILING_EMULATOR} "${executable}")
    else()
      set(_dyno_cmd ${CMAKE_CROSSCOMPILING_EMULATOR} "${executable}")
    endif()
  else()
    set(_dyno_cmd "${executable}")
  endif()
  set(${out_var} ${_dyno_cmd} PARENT_SCOPE)
endfunction()

function(dyno_configure_host_tools)
  set(_dyno_llvm_hints)
  foreach(_hint IN ITEMS
      "${DYNO_HOST_LLVM_TOOLS_DIR}"
      "${MLIR_TOOLS_DIR}"
      "${LLVM_TOOLS_BINARY_DIR}"
      "${LLVM_BINARY_DIR}/bin")
    if (_hint)
      list(APPEND _dyno_llvm_hints "${_hint}")
    endif()
  endforeach()

  _dyno_require_program(_dyno_host_mlir_opt mlir-opt HINTS ${_dyno_llvm_hints})
  _dyno_require_program(_dyno_host_mlir_translate mlir-translate
                        HINTS ${_dyno_llvm_hints})
  _dyno_require_program(_dyno_host_clang clang HINTS ${_dyno_llvm_hints})

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
    _dyno_require_program(_dyno_host_dyno_opt dyno-opt HINTS ${DYNO_HOST_TOOLS_DIR})
    _dyno_require_program(_dyno_host_dyno_translate dyno-translate
                          HINTS ${DYNO_HOST_TOOLS_DIR})
    set(_dyno_host_dyno_opt_cmd "${_dyno_host_dyno_opt}")
    set(_dyno_host_dyno_translate_cmd "${_dyno_host_dyno_translate}")
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
