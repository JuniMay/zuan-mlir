#-------------------------------------------------------------------------------
# Shared kernel build configuration
#-------------------------------------------------------------------------------

include(DynoKernelBuild)
dyno_configure_kernel_build()

include(DynoRegressionKernels)

#-------------------------------------------------------------------------------
# Runtime regression suites
#-------------------------------------------------------------------------------

add_subdirectory(linalg)
add_subdirectory(dyno)

#-------------------------------------------------------------------------------
# Runtime regression convenience target
#-------------------------------------------------------------------------------

get_property(_dyno_runtime_regression_targets GLOBAL
             PROPERTY DYNO_RUNTIME_REGRESSION_TARGETS)

add_custom_target(
  check-dyno-runtime-regressions
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L runtime-regression
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  USES_TERMINAL
)

if (_dyno_runtime_regression_targets)
  add_dependencies(check-dyno-runtime-regressions
                   ${_dyno_runtime_regression_targets})
endif()
