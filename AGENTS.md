# Dyno MLIR Compiler

## General Requirements

Unless otherwise specified, each implementation patch should preserve the
formalized semantics and structure defined in `docs/formalism.md`. The
vocabulary of variable names, function names, IR constructions, test cases,
comments, and other documentation should also adhere to the formalism.

Add comments to the codebase when the code is not self-explanatory or complex to
understand. A comment is mandatory on the function declaration, struct/class
fields and enum kinds. The comments should be added along with the code
instead of being deferred to the end.

Follow the principle of Occam's Razor when implementing the features. Avoid
unncessary complexity and over-engineering. If existing code are obsolete or
redundant, they should be notified after the on-going task is completed.

Unless otherwise specified or properly justified, no ad-hoc or case-specific
code should be added to the codebase. To report a case that requires an ad-hoc
solution, stop the current task and report immediately before any ad-hoc changes
are made.

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
- update or add runtime regressions in `regression/` when the patch can change
  runtime semantics.

Do not defer all testing to the end.

The regression failures that do not belong to the scope of the specified task
should be reported at the end of the task, and should not be fixed until 
explicit requests are made.
