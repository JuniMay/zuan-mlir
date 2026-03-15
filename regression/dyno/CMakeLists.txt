add_subdirectory(mask-gather-scatter)

# TODO: Fix the reduction regression test.
# This executable regression depends on the current temporary reduction
# lowering path through value-producing reductions and broadcasted reuse.
if (DYNO_ENABLE_EXPERIMENTAL_REDUCTION_EXECUTABLES)
  add_subdirectory(softmax)
endif()
