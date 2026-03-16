#-------------------------------------------------------------------------------
# lit configuration
#-------------------------------------------------------------------------------

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(DYNO_TEST_DEPENDS
  FileCheck count not
  dyno-opt
  dyno-translate
)

#-------------------------------------------------------------------------------
# lit targets
#-------------------------------------------------------------------------------

add_lit_testsuite(check-dyno "Running the dyno compiler regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNO_TEST_DEPENDS}
)

set_target_properties(check-dyno PROPERTIES FOLDER "Tests")

# Only materialize the lit sub-suite targets that correspond to real compiler
# test trees under `test/lit`. The sibling `test/runtime` tree is reserved for
# CTest-based runtime regressions and must not produce a misleading lit target.
add_lit_testsuites(
  DYNO
  ${CMAKE_CURRENT_SOURCE_DIR}/lit/Dyno
  ${CMAKE_CURRENT_SOURCE_DIR}/lit/VP
  DEPENDS ${DYNO_TEST_DEPENDS}
)

if (ENABLE_RUNTIME_REGRESSIONS)
  add_subdirectory(runtime)
endif()
