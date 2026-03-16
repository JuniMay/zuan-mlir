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
# test trees. The empty `test/runtime` directory is reserved for CTest-based
# runtime regressions configured under `regression/`, so it must not produce a
# misleading `check-dyno-runtime` lit target.
add_lit_testsuites(
  DYNO
  ${CMAKE_CURRENT_SOURCE_DIR}/Dyno
  ${CMAKE_CURRENT_SOURCE_DIR}/VP
  DEPENDS ${DYNO_TEST_DEPENDS}
)
