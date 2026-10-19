---
name: paper-implementation-divergence-log
description: >
  Living record of all known divergences between the canonical distilled
  theory references (`Circle_STARKs.llm.md`, `Stwo_Whitepaper.llm.md`) and
  the production codebase. Agents MUST read this before modifying any
  theoretically-grounded component. Agents MUST update this when finding new
  divergences. Never resolve a divergence silently.
---

# Paper-Implementation Divergence Log

Last analyzed: 2026-10-19

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — notation map and source navigation
- `.agents/papers/llm/Circle_STARKs.llm.md` — circle FFT/FRI/AIR theory anchors
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — STWO protocol/soundness/parameters

## How to Use This Log

- **Before modifying**: Search this log for the component you are modifying.
  If a divergence exists, understand it before changing anything.
- **After discovering**: Add a new entry immediately. Do not proceed without documenting.
- **Resolution**: Only a human with cryptography expertise can close a divergence
  as RESOLVED-SAFE. Agents may propose resolution, never confirm it.

## Active Divergences

### DIVERGENCE-001: Dimension Gap Handling in FRI

Paper: `.agents/papers/llm/Circle_STARKs.llm.md` —
`prot:IOP:proximity` + quotient decomposition anchors (`e:decomposition:q`,
`lem:quotient:decomposition`) require an explicit dimension-gap treatment:
`g = f - lambda * v_n` from `L_N(F)` to `L'_N(F)`.

Code: `crates/stwo/src/core/fri.rs` and `crates/stwo/src/prover/fri.rs` —
The FRI verifier has a `first_layer` / `inner_layers` / `last_layer` structure.
The circle-to-line fold (first layer) handles the J-split. There is no explicit
lambda scalar transmission in the current protocol flow. The quotient decomposition
in `crates/stwo/src/core/pcs/quotients.rs` handles the dimension gap implicitly
through the DEEP quotient construction.

Type: Intentional deviation
Risk: NEUTRAL (the constraint system ensures polynomials are in L'_N by construction
via the quotient identity; the dimension gap parameter lambda = 0 when constraint
degree is odd, which is the common case)
Status: OPEN
Notes: Verify that even-degree constraint systems correctly handle lambda != 0.

### DIVERGENCE-002: FRI Folding Order

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
`prot:cFRI:multi` / `e:cFRI:multi:folding` describe first J-split then
subsequent projection folds.

Code: `crates/stwo/src/core/fri.rs` — First layer performs circle-to-line
fold (`fold_circle`), then inner layers perform line folds. The constant
`CIRCLE_TO_LINE_FOLD_STEP` is used for the first fold. Inner layer queries are
derived by folding the original queries.

Type: Optimization (batched structure)
Risk: NEUTRAL
Status: OPEN
Notes: The mathematical operation is equivalent; the batching structure differs
from the sequential presentation in the paper.

### DIVERGENCE-003: QM31 as 4 Base Field Polynomials

Paper: `.agents/papers/llm/Circle_STARKs.llm.md` —
AIR model keeps trace polynomials in `L'_N(F_p)` while composition/challenges
use extension-field structure.

Code: Throughout `crates/stwo/src/core/pcs/` — QM31 polynomials are decomposed
into 4 base field coordinate polynomials for commitment and FRI. The composition
polynomial is split into 2 * SECURE_EXTENSION_DEGREE = 8 coordinate polynomials
(see `verify()` in `crates/stwo/src/core/verifier.rs`).

Type: Intentional deviation (efficiency)
Risk: NEUTRAL (mathematically equivalent; reduces FRI to base field operations)
Status: OPEN
Notes: This is standard practice. The `from_partial_evals` method in
`crates/stwo/src/core/fields/qm31.rs:51-57` reconstructs the combined value.

### DIVERGENCE-004: Composition Polynomial Split

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
composition and cross-domain quotient construction (`prot:STARK:IOPP`,
`e:crossdomain:quotient`) requires split handling by degree/domain.

Code: `crates/stwo/src/core/verifier.rs` — `COMPOSITION_LOG_SPLIT: u32 = 1`
is hardcoded. The split produces `2 * SECURE_EXTENSION_DEGREE` columns.
A TODO in the module notes this should be configurable.

Type: Intentional deviation (simplified)
Risk: PERFORMANCE (limits flexibility for higher-degree constraints)
Status: OPEN
Notes: The TODO acknowledges this limitation. For production stwo-cairo usage,
the current split may be sufficient.

### DIVERGENCE-005: Pairwise LogUp Column Grouping

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
logUp construction in `prot:STARK:IOPP` and related constraint equations;
implementation applies pairwise grouping optimization.

Code: `crates/constraint-framework/src/logup.rs` and
`crates/constraint-framework/src/prover/logup.rs` — LogUp implementation
groups fractions pairwise. The `Batching` type in `lib.rs:47` controls
which entries are batched together.

Type: Optimization (matches paper)
Risk: NEUTRAL
Status: RESOLVED-SAFE
Notes: Implementation matches the optimization described in the whitepaper.

### DIVERGENCE-006: Non-Transposed Merkle Tree

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
cross-domain Merkle commitments (`s:cross:domain:merkle`, `alg:merkle`).

Code: `crates/stwo/src/core/vcs_lifted/` — Uses a "lifted" Merkle tree
variant where multiple polynomials of different sizes are committed in a
single tree by lifting smaller polynomials to the largest domain size.
The `lifting_log_size` parameter in `PcsConfig` controls this.

Type: Intentional deviation (efficiency)
Risk: NEUTRAL (reduces number of Merkle trees and proof size)
Status: OPEN
Notes: The `vcs_lifted` module is a newer addition alongside the original `vcs`.

### DIVERGENCE-007: Security Parameter Defaults

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
parameter/soundness targets (Section "6. Parameter Rules", `s:example:params`)
including 100-bit security regimes with ~26 grinding bits.

Code: `PcsConfig::default()` in `crates/stwo/src/core/pcs/mod.rs` — Default uses
`pow_bits: 10`, `log_blowup_factor: 1`, `n_queries: 3`. This yields
`security_bits() = 10 + 1*3 = 13` bits — far below production requirements.

Type: Intentional deviation (test defaults)
Risk: SOUNDNESS (if used in production without override)
Status: OPEN
Notes: The default config is clearly for testing. Production deployments
(e.g., stwo-cairo) must set appropriate parameters. Document this prominently.

### DIVERGENCE-008: Proof of Work Grinding Bit Limit

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` —
parameter guidance in Section "6. Parameter Rules" / `s:example:params`.

Code: `crates/stwo/src/prover/backend/simd/grind.rs` — TODO comment:
"support more than 32 bits." Current implementation limited to 32 PoW bits.

Type: Implementation limitation
Risk: NEUTRAL (26 < 32, so current limit is sufficient for target security)
Status: OPEN
Notes: The 32-bit limit is adequate for the 26-bit target but leaves no room
for future security parameter increases.

### DIVERGENCE-009: Poseidon2 Constants and Round Order

Paper: Poseidon2 paper (https://eprint.iacr.org/2023/323.pdf) Section 5.

Code: `crates/examples/src/poseidon/mod.rs` — Three critical TODOs:
- "Use poseidon's real constants" — placeholder constants in use
- Coefficients unverified against Section 5.3
- Round matrix may be applied in wrong order relative to paper

Type: Bug candidate
Risk: SOUNDNESS (for any system using this Poseidon2 implementation)
Status: OPEN — ESCALATION REQUIRED
Notes: This is in the examples crate, not the core prover. However, any
downstream user of this code (e.g., for a Poseidon2 hash proof) would
inherit these issues. The placeholder constants make this unsuitable for
production use.

### DIVERGENCE-010: LogUp Validation Missing in Blake Example

Code: `crates/examples/src/blake/round/constraints.rs` — TODO: "validate logup"
Code: `crates/examples/src/blake/scheduler/mod.rs` — TODO: "validate logup"

Type: Implementation gap
Risk: SOUNDNESS (for Blake example only — constraints may be under-specified)
Status: OPEN
Notes: Example code only, but users may copy patterns from examples.

### DIVERGENCE-011: The First FRI Round Folds by the Fold Step

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` — `prot:cFRI:multi`; and the
round-by-round FRI soundness accounting (soundcalc `pcs/fri.py`,
`_get_commit_phase_error`: `(f - 1) · ε_lin(k / F_i)` for a round of factor `f`).

Code: `crates/stwo/src/core/fri.rs` (`FriVerifier::commit`, `SparseEvaluation::fold_circle`)
and `crates/stwo/src/prover/fri.rs` (`commit_inner_layers`) — the first layer folds the
circle to the line with `α` and then folds the line `fold_step - 1` times with `α², α⁴, …`,
before the next commitment: one round of factor `2^fold_step`, the circle fold included.
Inner layers then fold by `fold_step` each, the last by what remains.

Type: Intentional deviation (one commitment saved)
Risk: SOUNDNESS ACCOUNTING — a model that counts the circle fold as its own round of
factor 2 and the line folds after it in rounds of `2^fold_step` undercounts the first
round's error: `(2^s - 1)·ε_lin(k/2^s)` against `ε_lin(k/2)` and `(2^s - 1)·ε_lin(k/2^(s+1))`.
For fold step 3 that is about 0.8 bits on the fold term.
Status: OPEN
Notes: Tamper tests at fold step 3 in `prover/fri.rs` (`fold_step_3_tests`).

### DIVERGENCE-012: Affine DEEP Batching (opt-in)

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` — the DEEP quotient batches the
sampled values' quotients by powers of one coefficient.

Code: `crates/stwo/src/core/pcs/quotients.rs` (`DeepCoefficients`) and
`crates/stwo/src/core/pcs/mod.rs` (`DeepBatching`, `PcsConfig::deep_batching`) —
with `DeepBatching::Affine` the first quotient's coefficient is 1 and every other's is
drawn independently from the channel (after the sampled values are mixed), in place of
`α^i`. `DeepBatching::Powers`, the default, is the paper's.

Type: Intentional extension (soundness)
Risk: SOUNDNESS — improves the batching term from `(B - 1)·ε` (curves, BCIKS20 Thm 6.2)
to `ε` (affine spaces, BCIKS20 Thm 1.6), as soundcalc `2af5e7df` counts it. The field is
not serialized: a verifier must set it from its own configuration, never from a proof.
Status: OPEN (needs cryptographer review of the citation and the count)
Notes: Tests in `prover/pcs/quotient_ops.rs` and the wide-Fibonacci example.

### DIVERGENCE-013: The DEEP Quotient's Degenerate Lines Refused, Not Asserted

Paper: `.agents/papers/llm/Stwo_Whitepaper.llm.md` — the DEEP quotient divides each sampled
column by the line through the sample point z and its conjugate z̄, evaluated at the queried
point; the protocol assumes z̄ ≠ z (z.y outside CM31) and the line nonzero at every domain point.

Code: `crates/stwo/src/core/constraints.rs` (`complex_conjugate_line_coeffs`) and
`crates/stwo/src/core/pcs/quotients.rs` (`denominator_inverses`, `column_line_coeffs`,
`quotient_constants`, `accumulate_row_quotients`, `fri_answers`) — upstream asserts
`z.y ≠ conj(z.y)` (`assert_ne!`, the TODO's 2^-62) and inverts the line's values with
`CM31::batch_inverse`, which asserts on zero. Here both return
`VerificationError::InvalidStructure`, which `fri_answers` passes to `verify_values`; the
prover's callers (`prover/backend/{cpu,simd}/quotients.rs`) `expect` them, panicking where
upstream asserted.

Type: Intentional deviation (robustness: a verifier refuses, never panics)
Risk: NEUTRAL — the accepted set is unchanged: a proof that reached either assert was never
accepted. z.y in CM31 is about 2^-62 a draw of z (the channel's, so a prover grinds it); a zero
line at a point of the base field's circle needs `Im(z)` isotropic and nonzero, which no point of
the QM31 circle has (the proof is in `denominator_inverses`' doc), so for z on the circle only
the first is reachable. Points off the circle reach the second (the unit test's).
Status: OPEN
Notes: Unit tests in `core/pcs/quotients.rs` (`a_sample_point_with_y_in_cm31_is_refused`,
`a_conjugate_line_through_the_queried_point_is_refused`), each from a constructed point.

## Resolved Divergences

### DIVERGENCE-005 (see above)
Resolved: Pairwise LogUp grouping matches STWO distilled protocol guidance.
