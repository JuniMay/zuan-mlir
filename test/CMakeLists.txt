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

add_lit_testsuite(check-dyno "Running the dyno compiler regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DYNO_TEST_DEPENDS}
)

set_target_properties(check-dyno PROPERTIES FOLDER "Tests")

add_lit_testsuites(DYNO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${DYNO_TEST_DEPENDS})
