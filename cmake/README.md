# CMake Helpers

This directory contains project-local CMake modules used by Dyno's build.

- `DynoHostTools.cmake`: resolves host-side `mlir-opt`, `mlir-translate`,
  `clang`, `dyno-opt`, and `dyno-translate` for native and cross builds.
- `DynoBenchmarkKernels.cmake`: shared lowering/object-generation helpers for
  benchmark and future runtime-regression kernels.
- `DynoTriton.cmake`: Triton benchmark configuration and validation helpers.
- `DynoRewriteText.cmake`: small helper used to rewrite kernel entry names in
  generated MLIR inputs.
