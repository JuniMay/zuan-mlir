add_subdirectory(relu)

# TODO: Fix the reduction regression test.
# These executable regressions still rely on the current temporary reduction
# lowering path. Keep them opt-in until future work finishes the end-to-end
# reduction semantics they exercise.
if (DYNO_ENABLE_EXPERIMENTAL_REDUCTION_EXECUTABLES)
  add_subdirectory(matmul)
  add_subdirectory(matmul-transpose-a)
  add_subdirectory(reduce-sequential)
  add_subdirectory(reduce-parallel)
  add_subdirectory(reduce-2d-ordered)
  add_subdirectory(reduce-2d-factorized)
endif()
