# A Formalism Attempt for Dyno

## Design objective

Dyno is a structured IR for **dynamic-size register tiles**.

A `dyno.tile` is not a buffer. It is a conceptual register-like value whose shape may be dynamic at runtime. Dyno is intended to preserve the local memory access pattern and computation structure of Linalg-like kernels while remaining small enough to reason about formally and lower to vector, VP, and GPU-style backends.

The core Dyno basis does **not** require first-class contraction-style operations. In particular, Dyno core does not require:
- `dyno.contract`,
- `dyno.matmul`,
- `dyno.outer`.

Instead, the semantic basis is:
- memref/view transformation when an indexing map is view-representable,
- `dyno.load`, `dyno.gather`,
- shape-preserving pointwise elementwise operations,
- one `dyno.reduction` operation,
- `dyno.store`, `dyno.scatter`,
- structured control such as `scf.for` and `scf.while`.

This is sufficient because a structured kernel decomposes into:
1. access alignment into a common iteration domain,
2. pointwise local computation in that domain,
3. reduction over the reduced iterators,
4. store or scatter to the destination pattern.

## Semantic objects

### Scalar domains

Let $T$ range over scalar element domains such as integers, indices, booleans, and floating-point types.

### Shapes and domains

A rank-$r$ shape is a tuple

$$
S = (s_0, s_1, \dots, s_{r-1})
$$

with each $s_i > 0$, possibly dynamic at runtime.

The domain of $S$ is

$$
D(S) = \prod_{i=0}^{r-1} [0, s_i).
$$

Rank-0 shapes are allowed and have singleton domain.

### Tile values

A tile value `v : !dyno.tile<S x T>` denotes a total map

$$
\llbracket v \rrbracket : D(S) \to T.
$$

### Masks

A mask tile over shape $S$ denotes a boolean map

$$
\llbracket m \rrbracket : D(S) \to \{\mathrm{false}, \mathrm{true}\}.
$$

### Memrefs and access maps

A memref is treated denotationally as a total map from its legal index set to elements.

For an operand with structured iteration space $I$ and indexing map

$$
\phi : I \to J
$$

into a memref index space $J$, the semantic access equation is

$$
\mathrm{operand}(i) = A[\phi(i)]
\quad \text{for all } i \in I.
$$

Dyno may realize this either by:
1. a view transformation plus `dyno.load` / `dyno.store`, or
2. explicit index tiles plus `dyno.gather` / `dyno.scatter`.

The implementation choice does not change the semantics.

## Core operation classes

### Access-like operations

These materialize or consume tiles through memory mappings.
- `dyno.load`
- `dyno.gather`
- `dyno.store`
- `dyno.scatter`

### Pointwise shape-preserving operations

These preserve the tile domain and apply scalar operators pointwise.
- a restricted subset of `arith` and `math`,
- `dyno.cast`,
- `dyno.select`,
- value-producing `dyno.mask`.

### Domain constructors and transformers

These create or transform domains without contraction semantics.
- `dyno.splat`, restricted to prefix broadcast,
- `dyno.step`.

### Reduction-like operation

- `dyno.reduction`

### Structured control

- `scf.for`
- `scf.while`
- memref view/subview/transpose-like operations needed to preserve access patterns.

## Denotational semantics of core operations

### `dyno.load`

For a memref $A$ with shape $S$,

$$
\llbracket \mathrm{load}(A) \rrbracket(p) = A[p]
\quad \text{for all } p \in D(S).
$$

### `dyno.gather`

If index tiles $i_0, \dots, i_{k-1}$ share common domain $D$, then

$$
\llbracket \mathrm{gather}(A, i_0,\dots,i_{k-1}) \rrbracket(p) =
A[i_0(p),\dots,i_{k-1}(p)].
$$

### `dyno.store`

If $v$ has domain $D(S)$, then `store(v, A)` writes

$$
A[p] := \llbracket v \rrbracket(p)
\quad \text{for all } p \in D(S).
$$

### `dyno.scatter`

If $v$ and the index tiles share common domain $D$, then $\mathrm{scatter}(v, A, i_0, \dots, i_{k-1})$ writes

$$
A[i_0(p),\dots,i_{k-1}(p)] := \llbracket v \rrbracket(p)
\quad \text{for all } p \in D.
$$

### `dyno.splat`

`dyno.splat` is **prefix broadcast only**.

If $x$ is scalar and the new prefix shape is $P$, then

$$
\llbracket \mathrm{splat}(x, P) \rrbracket(p) = x.
$$

If $x$ is a tile over shape $S_x$ and the prefix shape is $P$, then the result domain is

$$
D(P) \times D(S_x)
$$

and

$$
\llbracket \mathrm{splat}(x, P) \rrbracket(p,q) = \llbracket x \rrbracket(q).
$$

No non-prefix broadcast is part of Dyno core semantics.

### `dyno.step`

For shape $S$, start value $a$, and distinguished dimension $k$,

$$
\llbracket \mathrm{step}(a, k, S) \rrbracket(p) = a + p_k.
$$

### Pointwise elementwise operations

Let $f : T_1 \times \dots \times T_n \to U$ be a scalar operator. If operands $v_1, \dots, v_n$ share common domain $D$, then

$$
\llbracket f(v_1,\dots,v_n) \rrbracket(p) = 
f(\llbracket v_1 \rrbracket(p),\dots,\llbracket v_n \rrbracket(p)).
$$

This covers the supported subset of `arith`, `math`, `dyno.cast`, and `dyno.select`.

### `dyno.mask`

Dyno uses one precise mask model.

Let $m$ be a boolean tile with domain $D$. For a value-producing masked region that yields $x$ and optional `maskedoff` value $y$ with domain $D$,

$$
\llbracket \mathrm{mask}(m, x, y) \rrbracket(p) =
\begin{cases}
\llbracket x \rrbracket(p) & \text{if } m(p) = \mathrm{true},\\
\llbracket y \rrbracket(p) & \text{if } m(p) = \mathrm{false} \text{ and } y \text{ exists.}
\end{cases}
$$

If `maskedoff` is absent, false lanes are semantically unspecified at the Dyno level and must be made explicit before any lowering step that requires definite values.

For effect-only masked regions, false lanes mean “do not perform the pointwise write”.

### `dyno.reduction`

Let the source tile have rank $r$, shape

$$
S = (s_0,\dots,s_{r-1}),
$$

and reduced-dimension set

$$
R = \{r_0 < \dots < r_{k-1}\}.
$$

Let the preserved-dimension set be

$$
P = \{0,\dots,r-1\} \setminus R.
$$

Let $D_P$ be the projected domain over preserved dimensions and $D_R$ the projected domain over reduced dimensions. Let

$$
\iota_{P,R}(q,t)
$$

reconstruct the full source index from preserved coordinate $q \in D_P$ and reduced coordinate $t \in D_R$.

For combiner $\oplus$ and optional explicit init $a_0$, define the canonical reduced sequence

$$
\mathrm{Seq}_{R}(q) =
\Bigl[
\llbracket v \rrbracket(\iota_{P,R}(q,t))
\Bigr]_{t \in D_R}^{\mathrm{lex}}
$$

where the enumeration is the canonical lexicographic order over $D_R$ in ascending source-dimension order.

Then

$$
\llbracket \mathrm{reduction}^{\oplus}_{R}(v, a_0) \rrbracket(q) =
\mathrm{foldl}_{\oplus}(a_0(q), \mathrm{Seq}_{R}(q)).
$$

If no explicit init is present, the combiner must admit an implicit identity $e_\oplus$, and $a_0(q)$ is taken to be that identity.

This canonical ordered fold is the source semantics against which all reduction lowerings are judged.

## Numerical policy and admissibility

Dyno reduction-preserving rewrites are parameterized by an explicit numerical policy $\Pi$.

Use notation

$$
x \approx_{\Pi} y
$$

for semantic equivalence under the active policy.

### Strict policy

Under strict policy, $\approx_{\Pi}$ means exact preservation of the canonical source semantics.

### Relaxed policy (fast math)

Under relaxed policy, certain floating-point transformations may be treated as valid even when they change parenthesization or lane grouping, provided the compiler has been told that top precision is not the primary requirement.

This relaxation must be explicit. It must not be inferred silently from the element type.

### Factorization-admissibility

A combiner $\oplus$ is **factorization-admissible under policy $\Pi$** if every parenthesization of the same ordered finite sequence has the same meaning under $\approx_{\Pi}$.

This is the property needed to replace one ordered fold by a different tree structure **without permuting the sequence order**.

Examples:
- integer addition is factorization-admissible under strict policy,
- integer multiplication is factorization-admissible under strict policy,
- integer min/max are factorization-admissible under strict policy,
- floating-point addition is generally **not** factorization-admissible under strict policy,
- floating-point addition may be treated as factorization-admissible under an explicit relaxed policy.

### Parallel-stripmine-admissibility

A combiner $\oplus$ is **parallel-stripmine-admissible under policy $\Pi$** if the regrouping induced by lane-wise chunk accumulation and final lane reduction is valid under $\approx_{\Pi}$.

This is strictly stronger than factorization-admissibility because lane-wise chunking changes both:
- parenthesization, and
- the grouping/permutation of terms.

In practice, this is the property needed for the “parallel per-lane accumulation + final one-register reduction” lowering.

Examples:
- integer addition is parallel-stripmine-admissible under strict policy,
- integer multiplication is parallel-stripmine-admissible under strict policy,
- min/max are parallel-stripmine-admissible under strict policy,
- floating-point addition is generally **not** parallel-stripmine-admissible under strict policy,
- floating-point addition may be treated as parallel-stripmine-admissible under an explicit relaxed policy.

The implementation must maintain a precise legality table for supported reduction kinds and element types.

## Two distinct transformation families

Dyno requires two transformation families, with different laws and different proof obligations.

### Result-domain slicing

This slices the visible result domain of a pointwise or access-like operation.

### Reduction strip-mining

This partitions a reduced dimension into chunks and combines chunk results.

The old “one generic unroll interface for everything” is not a correct abstraction. Result-domain slicing and reduction strip-mining must remain separate in both theory and implementation.

## Theorem A — pointwise result-domain slicing

Let $f$ be a shape-preserving pointwise operation, and let $\sigma$ be a slice operator on the result domain. Then

$$
\sigma(f(v_1,\dots,v_n)) =
f(\sigma_1(v_1),\dots,\sigma_n(v_n)),
$$

where:
- $\sigma_i = \sigma$ when operand $v_i$ depends on the sliced result dimension,
- $\sigma_i = \mathrm{id}$ when operand $v_i$ is independent of that dimension.

This theorem applies to:
- supported pointwise `arith` / `math`,
- `dyno.cast`,
- `dyno.select`,
- value-producing `dyno.mask`,
- `dyno.load` / `dyno.gather` after the corresponding operand slice is derived.

It does **not** by itself justify reduction strip-mining.

## One-dimensional reduction semantics and strip-mining modes

Let $d$ be one reduced dimension of length $n = s_d$. For a fixed preserved coordinate $q$, define the source sequence

$$
x_q(i) = \llbracket v \rrbracket(\iota_{P,\{d\}}(q, i))
\quad \text{for } i \in [0,n).
$$

The canonical 1-D reduction is

$$
y(q) = \mathrm{foldl}_{\oplus}(a_0(q), [x_q(0), x_q(1), \dots, x_q(n-1)]).
$$

Dyno admits two distinct strip-mined implementation modes.

### Theorem B1 — parallel strip-mining

Choose a vector width $w > 0$. Partition the sequence into contiguous chunks

$$
C_j = [jw, \min((j+1)w, n))
\quad \text{for } j = 0,\dots,m-1
$$

with $m = \lceil n / w \rceil$.

Define the lane subsequences

$$
L_\lambda(q) = [x_q(\lambda), x_q(w + \lambda), x_q(2w + \lambda), \dots]
$$

restricted to indices $< n$, for each lane $\lambda \in [0,w)$.

The **parallel strip-mined** result is

$$
y_{\mathrm{par}}(q) =
\mathrm{foldl}_{\oplus}
\Bigl(
 a_0(q),
 [
   \mathrm{foldl}_{\oplus}(e_{\oplus}, L_0(q)),
   \dots,
   \mathrm{foldl}_{\oplus}(e_{\oplus}, L_{w-1}(q))
 ]
\Bigr).
$$

This corresponds exactly to:
1. maintaining a **vector accumulator** whose lanes accumulate chunk values lane-wise,
2. then performing **one final register reduction** across the accumulator lanes.

This transformation is valid only if all of the following hold:
1. $\oplus$ has the correct identity $e_\oplus$,
2. inactive tail lanes are masked so that they contribute exactly the identity / no contribution,
3. the final one-register reduction uses the same combiner $\oplus$,
4. $\oplus$ is parallel-stripmine-admissible under the active policy $\Pi$.

Under strict policy, this is exact only for reduction kinds that admit this regrouping exactly. Under relaxed policy, floating-point addition may be admitted if the user explicitly selected that policy.

### Theorem B2 — sequential strip-mining

Let the same dimension be partitioned into contiguous chunks

$$
C_0, C_1, \dots, C_{m-1}
$$

in increasing source order.

Define the ordered chunk-fold operator

$$
\mathrm{ordchunk}_{\oplus}(a, C_j) =
\mathrm{foldl}_{\oplus}(a, [x_q(i)]_{i \in C_j}^{\uparrow}).
$$

The **sequential strip-mined** result is

$$
y_{\mathrm{seq}}(q) =
\mathrm{ordchunk}_{\oplus}(
  \dots \mathrm{ordchunk}_{\oplus}(
    \mathrm{ordchunk}_{\oplus}(a_0(q), C_0),
    C_1),
  \dots,
  C_{m-1}).
$$

This corresponds to:
1. processing chunks in source order,
2. reducing the active lanes of each chunk **in lane order into the accumulator**,
3. carrying that updated accumulator directly to the next chunk,
4. performing **no final register reduction**.

This is the semantics of an ordered vector reduction, such as RVV-style ordered reduction instructions.

The transformation is exact provided:
1. chunks are processed in increasing source order,
2. active lanes within each chunk are accumulated in increasing lane order,
3. the accumulator is threaded across chunks without any final regrouping.

No factorization-admissibility or parallel-stripmine-admissibility assumption is required. Sequential strip-mining is therefore the exact default for strict floating-point reductions when precision requirements forbid reassociation.

## Higher-dimensional reductions

Higher-dimensional reductions are first-class Dyno semantics.

### Canonical semantics

For reduced-dimension set $R$, the source semantics are the ordered fold over the lexicographically enumerated sequence $\mathrm{Seq}_R(q)$ defined in Section 5.9.

### Theorem C1 — factorization into repeated 1-D reductions

Let $R = \{r_0 < \dots < r_{k-1}\}$. If $\oplus$ is factorization-admissible under policy $\Pi$, then an $R$-reduction may be implemented as repeated 1-D reductions in **descending source-dimension order**:

$$
\mathrm{reduction}^{\oplus}_{R}(v)
\approx_{\Pi}
\mathrm{reduction}^{\oplus}_{\{r'_0\}}
\Bigl(
\mathrm{reduction}^{\oplus}_{\{r'_1\}}
\bigl(
\dots
\mathrm{reduction}^{\oplus}_{\{r'_{k-1}\}}(v)
\dots
\bigr)
\Bigr)
$$

where $r'_0 > r'_1 > \dots > r'_{k-1}$ are the same reduced source dimensions in descending order.

Descending source-dimension order is required because it preserves the intended index renumbering discipline and matches the canonical lexicographic sequence.

This theorem depends on **factorization-admissibility**. It does **not** follow merely because the reduction is higher-dimensional.

Consequences:
- integer `add`, `mul`, `min`, `max`, etc. may be factorized under strict policy,
- strict floating-point `add` may **not** in general be factorized this way,
- relaxed floating-point `add` may be factorized only when the active policy explicitly allows it.

### Theorem C2 — exact ordered lowering without factorization

If $\oplus$ is **not** factorization-admissible under the active policy, then a higher-dimensional reduction must not be rewritten into a tree of independent partial reductions.

Instead, it must be lowered as an exact lexicographic traversal carrying one accumulator:

$$
a_{t+1} = a_t \oplus \mathrm{Seq}_{R}(q)[t]
$$

for the canonical ordered sequence $\mathrm{Seq}_{R}(q)$.

Operationally, this means:
1. generate nested loops over the reduced dimensions in canonical lexicographic order,
2. carry the accumulator across the full traversal,
3. optionally strip-mine only the current innermost frontier using **sequential** 1-D strip-mining,
4. do not create intermediate partial reductions whose later recombination changes parenthesization.

This is the exact higher-dimensional rule needed for strict floating-point ordered reductions.

### Theorem D — mixed higher-dimensional chunking

Suppose one reduced dimension is chunked while other reduced dimensions remain explicit.

- If the chunked dimension uses **parallel strip-mining**, legality requires the conditions of Theorem B1 plus whatever factorization is used around it.
- If the chunked dimension uses **sequential strip-mining**, legality requires the conditions of Theorem B2 and preservation of the global lexicographic traversal order.

Therefore, higher-dimensional reduction lowering must choose between:
- a **factorized path** for admissible combiners, and
- an **ordered traversal path** for exact non-factorizable cases.

## Required normal forms

### Dyno Core Normal Form (DCNF)

After front-end conversion and local canonicalization:
- no first-class `dyno.contract`, `dyno.matmul`, or `dyno.outer`,
- only the approved Dyno core basis remains,
- every `dyno.reduction` is type-legal,
- every reduction has either an explicit init or a valid implicit identity,
- broadcasts are represented only by prefix `dyno.splat`, views, or gather/scatter.

### Dyno Reduction Normal Form (DRNF)

Before target-specific reduction lowering:
- every reduction has an explicit policy context,
- reduction legality is decided using:
  - factorization-admissibility,
  - parallel-stripmine-admissibility,
- higher-dimensional reductions are handled by one of two paths:
  1. **factorized repeated 1-D reductions** if the combiner is factorization-admissible under the active policy,
  2. **ordered lexicographic traversal** otherwise,
- no pass is allowed to normalize a non-factorizable reduction into a partial-reduction tree.

### Dyno VP Preparation Form (DVPF)

Immediately before the VP-oriented lowering:
- effect roots operate on scalar or 1-D tiles only,
- any higher-rank computation has already been converted to outer loops plus scalar/1-D inner kernels,
- any reduction reaching the VP boundary is in one of the following forms:
  1. a 1-D reduction eligible for **parallel** strip-mining,
  2. a 1-D reduction eligible for **sequential** strip-mining,
  3. an ordered loop-carried scalar or 1-lane accumulation produced from a non-factorizable higher-dimensional reduction,
- no illegal higher-dimensional strict floating-point reduction remains disguised as repeated 1-D partial sums.

### VP Conversion Input Form (VP-CIF)

Inside the VP converter:
- only scalar or 1-D Dyno tiles are accepted,
- reduction lowering must distinguish **parallel** and **sequential** 1-D modes,
- the existing parallel reduction path must not be treated as ordered by accident,
- exact ordered reductions must either:
  - lower through an explicit ordered VP reduction form, or
  - stay in an explicit ordered loop structure until a backend-specific lowering handles them.

## Interface requirements

Dyno should expose or emulate the following interfaces.

### Access-like interface

Required queries:
- result domain,
- operand/result dependence on result dimensions,
- how to derive a sliced memref view or sliced index tiles.

### Pointwise-like interface

Required queries:
- all operands share a common domain,
- which operands depend on which result dimensions,
- how to rebuild the op on sliced operands.

### Reduction-like interface

Required queries:
- reduced dimension set,
- preserved dimension set,
- canonical ordered reduction sequence,
- result shape from source shape,
- supported combiner kinds,
- identity construction,
- factorization-admissibility,
- parallel-stripmine-admissibility,
- legal lowering modes under the active policy.

### Effect-like interface

Required queries:
- whether false mask lanes suppress writes,
- whether the op writes pointwise to memory or another effect target.

For upstream `arith` and `math` ops that cannot practically grow new Dyno-specific interfaces, Dyno should maintain a conservative whitelist utility.

## Verifier and conformance requirements

A conforming Dyno implementation must satisfy all of the following.

### Reduction requirements

- reduction dimensions are unique and in canonical form,
- result type matches the preserved dimensions,
- explicit init type matches the reduction result type,
- implicit identity is used only for supported reduction kinds,
- the identity value is correct for the combiner and element type,
- strict and relaxed modes are never inferred silently from the type alone.

### Mask requirements

- conditions are boolean,
- maskedoff values have the exact pointwise result type,
- false lanes suppress writes for effect-only masked operations.

### Access requirements

- sliced memrefs preserve the original memory space,
- gather/scatter index tiles share a common domain,
- any view transformation must preserve the original indexing semantics.

### Broadcast requirements

- `dyno.splat` remains prefix-only,
- non-prefix broadcast must be represented by other explicit mechanisms.

## Project-level consequences

The formal consequences for the Dyno project are:

1. the semantic core is small and contraction-free,
2. result-domain slicing and reduction strip-mining are separate subsystems,
3. reduction legality depends on explicit numerical policy,
4. 1-D reductions have two distinct lowering modes:
   - parallel strip-mining with final register reduction,
   - sequential strip-mining with ordered chunk accumulation and no final reduction,
5. higher-dimensional reductions cannot all be normalized the same way:
   - admissible ones may factorize into repeated 1-D reductions,
   - strict non-factorizable ones require ordered lexicographic lowering,
6. the VP path must reflect this distinction explicitly rather than assuming one reduction scheme is universally correct.

That is the intended Dyno semantics.
