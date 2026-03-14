if (NOT BUILD_TESTING)
  return()
endif()

if (ENABLE_DYNO_RUNTIME_REGRESSIONS)
  # TODO: Add regression test.
  # add_custom_target(check-dyno-runtime)
endif()
