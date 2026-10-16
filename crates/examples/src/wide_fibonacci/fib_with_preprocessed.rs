#![allow(unused)]
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::{Backend, Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

pub type WideFibWithPpComponent<const N: usize> = FrameworkComponent<WideFibWithPpEval<N>>;

pub struct FibInput {
    pub a: BaseField,
    pub b: BaseField,
}

pub fn generate_preprocessed_trace<B: Backend>(
    log_size: u32,
) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut pp_col = Col::<B, BaseField>::zeros(1 << log_size);
    (0..1 << log_size).for_each(|i| pp_col.set(i, BaseField::from(i)));
    let domain = CanonicCoset::new(log_size).circle_domain();
    CircleEvaluation::<B, _, BitReversedOrder>::new(domain, pp_col)
}

pub fn generate_trace<const N: usize, B: Backend>(
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    assert!(inputs.len().is_power_of_two());
    let log_size = inputs.len().ilog2();
    let mut trace = (0..N)
        .map(|_| Col::<B, BaseField>::zeros(1 << log_size))
        .collect_vec();

    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].set(vec_index, a);
        trace[1].set(vec_index, b);
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (
                b,
                a.square() + b.square() + BaseField::from(vec_index).square(),
            );
            col.set(vec_index, b);
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<B, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// A component that at row n (starting from 0) enforces the sequence `aₙ = aₙ₋₁² + aₙ₋₂² + n²`.
#[derive(Clone)]
pub struct WideFibWithPpEval<const N: usize> {
    pub log_n_rows: u32,
}
impl<const N: usize> FrameworkEval for WideFibWithPpEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        let seq = eval.get_preprocessed_column(PreProcessedColumnId {
            id: String::from("seq"),
        });

        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square() + seq.square()));
            a = b;
            b = c;
        }
        eval
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sM31Channel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
    use stwo::core::verifier::verify;
    use stwo::prover::backend::simd::column::BaseColumn;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::{prove, CommitmentSchemeProver};
    use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
    use stwo_constraint_framework::TraceLocationAllocator;

    use super::{generate_preprocessed_trace, generate_trace, FibInput, WideFibWithPpEval};
    use crate::wide_fibonacci::fib_with_preprocessed::WideFibWithPpComponent;
    use crate::wide_fibonacci::{WideFibonacciComponent, WideFibonacciEval};

    const FIB_SEQUENCE_LENGTH: usize = 3;

    fn generate_test_inputs(log_n_instances: u32) -> Vec<FibInput> {
        (0..1 << log_n_instances)
            .map(|i| FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i as u32),
            })
            .collect_vec()
    }

    #[ignore]
    #[test_log::test]
    fn test_wide_fib_with_pp_prove_with_blake() {
        for log_n_instances in 4..=8 {
            let config = PcsConfig::default();
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            let preprocessed_trace = generate_preprocessed_trace(log_n_instances);
            tree_builder.extend_evals(vec![preprocessed_trace]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let component = WideFibWithPpComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibWithPpEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test_log::test]
    /// The preprocessed tree sits below the lifting height, so its query positions are mapped by
    /// `prepare_preprocessed_query_positions`, which does not preserve their order. A second,
    /// larger component raises the lifting height; many queries on a small domain make unsorted
    /// and non-adjacent duplicate preprocessed positions certain.
    fn test_wide_fib_with_pp_below_lifting_height() {
        let log_n_pp_rows = 4;
        let log_n_rows = log_n_pp_rows + 2;
        let config = PcsConfig {
            pow_bits: 0,
            fri_config: FriConfig::new(0, 1, 64, 1),
            lifting_log_size: None,
        };
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_n_rows + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Prove.
        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![generate_preprocessed_trace(log_n_pp_rows)]);
        tree_builder.commit(prover_channel);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(generate_trace::<FIB_SEQUENCE_LENGTH, _>(
            &generate_test_inputs(log_n_pp_rows),
        ));
        let large_inputs = (0..1 << log_n_rows)
            .map(|i| crate::wide_fibonacci::FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i),
            })
            .collect_vec();
        tree_builder.extend_evals(crate::wide_fibonacci::generate_trace::<
            FIB_SEQUENCE_LENGTH,
            _,
        >(&large_inputs));
        tree_builder.commit(prover_channel);
        let allocator = &mut TraceLocationAllocator::default();
        let pp_component = WideFibWithPpComponent::new(
            allocator,
            WideFibWithPpEval::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: log_n_pp_rows,
            },
            SecureField::zero(),
        );
        let large_component = WideFibonacciComponent::new(
            allocator,
            WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { log_n_rows },
            SecureField::zero(),
        );
        let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
            &[&pp_component, &large_component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sM31Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
        let sizes = TreeVec::concat_cols(
            [
                pp_component.trace_log_degree_bounds(),
                large_component.trace_log_degree_bounds(),
            ]
            .into_iter(),
        );
        commitment_scheme.commit(proof.commitments[0], &[log_n_pp_rows], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        assert!(commitment_scheme.trees[0].height < commitment_scheme.trees[1].height);
        verify(
            &[&pp_component, &large_component],
            verifier_channel,
            commitment_scheme,
            proof,
        )
        .unwrap();
    }

    #[test_log::test]
    fn test_wide_fib_with_unused_pp_prove_with_blake() {
        for log_n_instances in 4..=8 {
            let config = PcsConfig::default();
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            let mut preprocessed_trace = vec![generate_preprocessed_trace(log_n_instances)];
            // Build a long unused preprocessed column.
            let log_size_unused_pp = log_n_instances + 1;
            let domain = CanonicCoset::new(log_size_unused_pp).circle_domain();
            preprocessed_trace.push(CircleEvaluation::new(
                domain,
                BaseColumn::zeros(1 << log_size_unused_pp),
            ));
            tree_builder.extend_evals(preprocessed_trace);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[
                PreProcessedColumnId {
                    id: String::from("seq"),
                },
                PreProcessedColumnId {
                    id: String::from("large_unused"),
                },
            ]);
            let component = WideFibWithPpComponent::new(
                &mut allocator,
                WideFibWithPpEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(
                proof.commitments[0],
                &[log_n_instances, log_size_unused_pp],
                verifier_channel,
            );
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test_log::test]
    /// This test is equal to the previous one except it tests the hardcoding of lifting size.
    fn test_wide_fib_with_unused_pp_and_hardcoded_lifting() {
        for log_n_instances in 4..=8 {
            let mut config = PcsConfig::default();
            let log_size_unused_pp = log_n_instances + 3;
            // Set lifting log size to the largest preprocessed column (after LDE).
            config.lifting_log_size =
                Some(log_size_unused_pp + config.fri_config.log_blowup_factor);
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_size_unused_pp + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            let mut preprocessed_trace = vec![generate_preprocessed_trace(log_n_instances)];
            // Build a long unused preprocessed column.
            let domain = CanonicCoset::new(log_size_unused_pp).circle_domain();
            preprocessed_trace.push(CircleEvaluation::new(
                domain,
                BaseColumn::zeros(1 << log_size_unused_pp),
            ));
            tree_builder.extend_evals(preprocessed_trace);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[
                PreProcessedColumnId {
                    id: String::from("seq"),
                },
                PreProcessedColumnId {
                    id: String::from("large_unused"),
                },
            ]);
            let component = WideFibWithPpComponent::new(
                &mut allocator,
                WideFibWithPpEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let mut sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(
                proof.commitments[0],
                &[log_n_instances, log_size_unused_pp],
                verifier_channel,
            );
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }
}
