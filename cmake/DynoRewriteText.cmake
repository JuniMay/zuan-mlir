# Small script-mode helper used by `dyno_rewrite_mlir_kernel()`.
#
# Required cache/script variables:
# - `INPUT`: source text file
# - `OUTPUT`: rewritten output file
# - `OLD`: exact substring to replace
# - `NEW`: replacement substring
#
# The rewrite is intentionally simple and textual because the kernel entry
# points are generated from templates with a stable `kernel(` spelling.
foreach(_required IN ITEMS INPUT OUTPUT OLD NEW)
  if (NOT DEFINED ${_required})
    message(FATAL_ERROR "DynoRewriteText.cmake requires `${_required}`")
  endif()
endforeach()

file(READ "${INPUT}" _dyno_contents)
string(REPLACE "${OLD}" "${NEW}" _dyno_contents "${_dyno_contents}")
file(WRITE "${OUTPUT}" "${_dyno_contents}")
