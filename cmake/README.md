# CMake Helpers

This directory contains project-local CMake modules used by Dyno's build.

- `DynoHostTools.cmake`: resolves host-side `mlir-opt`, `mlir-translate`,
  `clang`, `dyno-opt`, and `dyno-translate` for native and cross builds.
- `DynoKernelBuild.cmake`: shared kernel-build configuration entry point used
  by benchmark and runtime-regression directories.
- `DynoKernelLowering.cmake`: shared lowering and object-generation helpers for
  benchmark and runtime-regression kernels.
- `DynoBenchmarkKernels.cmake`: benchmark-specific kernel target helpers built
  on top of the shared lowering layer.
- `DynoRegressionKernels.cmake`: runtime-regression helper layer built on top
  of the shared lowering layer.
- `DynoGoogleBenchmark.cmake`: configures the vendored Google Benchmark
  dependency used by `benchmarks/`.
- `DynoTriton.cmake`: Triton benchmark configuration and validation helpers.
- `DynoRewriteText.cmake`: small helper used to rewrite kernel entry names in
  generated MLIR inputs.

See `docs/kernel-building.md` for the end-to-end kernel build flow.
