# Dyno MLIR Compiler

## General Requirements

Unless otherwise specified, each implementation patch should preserve the
formalized semantics and structure defined in `docs/formalism.md`. The
vocabulary of variable names, function names, IR constructions, test cases,
comments, and other documentation should also adhere to the formalism.

## Regression Testing

Regression testing includes:
- `lit`-based IR transformation regressions, and
- runtime artifact execution and result checking. 

Unless otherwise specified, runtime artifacts are ment to run on RISC-V
platforms and should be executed using QEMU on the host machine that are not of
RISC-V architecture.

Unless otherwise specified, or reasonably justified as unnecessary or
impractical, each implementation patch should do both of the following:
- update or add `lit` tests in `test/`;
- update or add runtime regressions in `regression/` when the patch can change runtime semantics.

Do not defer all testing to the end.

The regression failures that do not belong to the scope of the specified task
should be reported but not necessarily fixed in the same patch series.

## Historical context

This project was originally developed under the name Zuan. If any old names are
encountered in the codebase, they should be considered as candidates for renaming
to the new name, Dyno. All these renaming needs to be notified after the task is
completed.
