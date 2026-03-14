foreach(_required IN ITEMS INPUT OUTPUT OLD NEW)
  if (NOT DEFINED ${_required})
    message(FATAL_ERROR "DynoRewriteText.cmake requires `${_required}`")
  endif()
endforeach()

file(READ "${INPUT}" _dyno_contents)
string(REPLACE "${OLD}" "${NEW}" _dyno_contents "${_dyno_contents}")
file(WRITE "${OUTPUT}" "${_dyno_contents}")
