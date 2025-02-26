configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ZUAN_TEST_DEPENDS
  FileCheck count not
  zuan-opt
  zuan-translate
)

add_lit_testsuite(check-zuan "Running the zuan compiler regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ZUAN_TEST_DEPENDS}
)

set_target_properties(check-zuan PROPERTIES FOLDER "Tests")

add_lit_testsuites(ZUAN ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ZUAN_TEST_DEPENDS})
