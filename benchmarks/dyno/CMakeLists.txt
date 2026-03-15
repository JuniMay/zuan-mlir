# Keep direct Dyno executable benchmarks opt-in while reduction lowering still
# relies on the current temporary compatibility path.
if (DYNO_ENABLE_EXPERIMENTAL_REDUCTION_EXECUTABLES)
  add_subdirectory(matmul)
endif()
