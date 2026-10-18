use std::iter::zip;

use itertools::Itertools;
use tracing::{span, Level};

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    build_samples_with_randomness_and_periodicity, ColumnSampleBatch, DeepCoefficients, PointSample,
};
use crate::core::pcs::TreeVec;
use crate::prover::backend::ColumnOps;
use crate::prover::poly::circle::{CircleEvaluation, PolyOps, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

pub trait QuotientOps: PolyOps {
    /// Receives a non-empty set of columns of the *same* size, and populates the vector
    /// `accumulated_numerators_vec` with their FRI numerators accumulations, across
    /// `sample_batches`.
    ///
    /// For each sample batch in `sample_batches`, accumulates the numerators of the columns
    /// involved in this batch over the evaluation subdomain (the first
    /// `column_size >> log_blowup_factor` rows in bit-reversed order) and pushes an
    /// `AccumulatedNumerators` object into `accumulated_numerators_vec`.
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    );

    /// Given a vector of `AccumulatedNumerators` (computed on evaluation subdomains), computes
    /// the full quotient on the subdomain of size `2^(lifting_log_size - log_blowup_factor)`,
    /// then interpolates and evaluates on the full domain of size `2^lifting_log_size`.
    ///
    /// For each subdomain point:
    /// * Computes the denominator inverse for each sample point.
    /// * Multiplies it by the accumulated numerator for that (subdomain point, sample point).
    /// * Sums across sample points.
    ///
    /// The result is then extended to the full evaluation domain via interpolation on the
    /// subdomain followed by evaluation on the full domain, using the provided `twiddles`.
    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureEvaluation<Self, BitReversedOrder>;
}

/// Helper struct that keeps track of the accumulation of the numerators involved in the FRI
/// quotients.
///
/// Note: `Clone` is derived only for benchmarking purposes.
#[derive(Clone)]
pub struct AccumulatedNumerators<B: ColumnOps<BaseField>> {
    /// One of the sample points received by the pcs.
    pub sample_point: CirclePoint<SecureField>,
    /// Stores a circle evaluation of the form:
    ///     p -> ∑ α^{k_i} * (cᵢ * f̃ᵢ(p) - bᵢ)
    /// where
    /// * p ∈ canonic coset of log size = l (where l is the log size of the column).
    /// * i runs over some column indices of the trace.
    /// * α is the random coefficient for the accumulation.
    /// * k_i is the randomness exponent for column i and sample point `sample_point`.
    /// * f̃ᵢ is the lift of the trace poly fᵢ to log size l.
    /// * (bᵢ, cᵢ) are the `b` and `c` line coefficients for column i and sample point
    ///   `sample_point`.
    pub partial_numerators_acc: SecureColumnByCoords<B>,
    /// Stores an accumulation of the form
    ///      ∑ α^{k_i} * aᵢ
    /// where
    /// * i runs over some column indices of the trace.
    /// * α and k_i are as in the previous docstring for `partial_numerators_acc`.
    /// * aᵢ is the `a` line coefficient for column i and sample point `sample_point`.
    ///
    /// The index set of the summation is equal to the index set of the summation referred in the
    /// previous docstring for `partial_numerators_acc`.
    pub first_linear_term_acc: SecureField,
}

pub fn compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &TreeVec<Vec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
    samples: &TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: impl Into<DeepCoefficients>,
    lifting_log_size: u32,
    twiddles: &TwiddleTree<B>,
    log_blowup_factor: u32,
) -> SecureEvaluation<B, BitReversedOrder> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    let mut accumulated_numerators_vec: Vec<AccumulatedNumerators<B>> = vec![];
    let samples_with_randomness = build_samples_with_randomness_and_periodicity(
        samples,
        columns
            .0
            .iter()
            .map(|x| x.iter().map(|c| c.domain.log_size()))
            .collect(),
        lifting_log_size,
        random_coeff,
    );

    // Populate `accumulated_numerators_vec`, per (log_size, sample_point). After this iteration,
    // `accumulated_numerators_vec` will have length equal to
    //
    //   ∑_k (# of distinct sample points per log size k).
    //
    zip(
        columns.iter().flatten(),
        samples_with_randomness.iter().flatten(),
    )
    .sorted_by_key(|(c, _)| c.domain.log_size())
    .group_by(|(c, _)| c.domain.log_size())
    .into_iter()
    .for_each(|(_, tuples)| {
        let (columns, samples_with_randomness): (Vec<_>, Vec<_>) = tuples.unzip();
        // TODO: slice.
        let sample_batches = ColumnSampleBatch::new_vec(&samples_with_randomness);
        B::accumulate_numerators(
            &columns,
            &sample_batches,
            &mut accumulated_numerators_vec,
            log_blowup_factor,
        )
    });

    // Group and accumulate the numerators per sample point: the accumulations (of different
    // lengths) get lifted and accumulated to a single vector. After this step, there is a single
    // accumulation per sample point.
    let accumulations_per_sample_point = accumulated_numerators_vec
        .into_iter()
        .sorted_by_key(|c| (c.sample_point.x, c.sample_point.y))
        .group_by(|c| c.sample_point)
        .into_iter()
        .map(|(sample_point, accumulations_per_log_size)| {
            let accumulations_per_log_size = accumulations_per_log_size.collect_vec();
            // Accumulate the `a` coefficients.
            let first_linear_term_acc: SecureField = accumulations_per_log_size
                .iter()
                .map(|x| x.first_linear_term_acc)
                .sum();
            // Lift and accumulate the partial numerators vectors.
            // `partial_numerators_acc` is already sorted increasingly by size as required by
            // `B::lift_and_accumulate`.
            let partial_numerators_acc = accumulations_per_log_size
                .into_iter()
                .map(|x| x.partial_numerators_acc)
                .collect_vec();
            let res = B::lift_and_accumulate(partial_numerators_acc).unwrap();

            AccumulatedNumerators {
                sample_point,
                partial_numerators_acc: res,
                first_linear_term_acc,
            }
        })
        .collect_vec();

    B::compute_quotients_and_combine(
        accumulations_per_sample_point,
        lifting_log_size,
        log_blowup_factor,
        twiddles,
    )
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::channel::{Blake2sChannel, Channel, MerkleChannel};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::M31;
    use crate::core::fri::FriConfig;
    use crate::core::pcs::quotients::{
        fri_answers, n_deep_values, CommitmentSchemeProof, DeepCoefficients, PointSample,
    };
    use crate::core::pcs::{CommitmentSchemeVerifier, DeepBatching, PcsConfig, TreeVec};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::vcs_lifted::blake2b_merkle::Blake2bMerkleChannel;
    use crate::core::verifier::VerificationError;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Backend, BackendForChannel, Column, CpuBackend};
    use crate::prover::pcs::quotient_ops::compute_fri_quotients;
    use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
    use crate::prover::{CommitmentSchemeProver, SecureField};

    #[test]
    fn test_quotients_are_low_degree() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 4;

        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(M31::from).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let eval = polynomial.evaluate(eval_domain);

        let sample_points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = sample_points
            .into_iter()
            .map(|x| PointSample {
                point: x,
                value: polynomial.eval_at_point(x),
            })
            .collect_vec();
        let rand_coeff =
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));
        let quot_eval = compute_fri_quotients(
            &TreeVec(vec![vec![&eval]]),
            &TreeVec(vec![vec![samples]]),
            rand_coeff,
            LOG_SIZE + LOG_BLOWUP_FACTOR,
            &CpuBackend::precompute_twiddles(eval_domain.half_coset),
            LOG_BLOWUP_FACTOR,
        );
        let mut coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.clone()).interpolate())
            .collect_vec();
        let zeros = coeffs[0].coeffs.split_off((1 << LOG_SIZE) - 1);

        assert!(zeros.iter().all(|c| c.is_zero()));
    }

    /// Generates a vector of random polynomials, such that the last one is of degree
    /// `LIFTING_LOG_SIZE - 1` and all the previous ones are of degree < `LIFTING_LOG_SIZE - 1`.
    fn prepare_polys<B: Backend, const N_COLS: usize, const LIFTING_LOG_SIZE: u32>(
    ) -> Vec<CircleCoefficients<B>> {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut polys: Vec<CircleCoefficients<B>> = (0..N_COLS - 1)
            .map(|_| {
                CircleCoefficients::new(
                    (0..1 << rng.gen_range(4..LIFTING_LOG_SIZE - 1))
                        .map(M31::from)
                        .collect(),
                )
            })
            .collect();
        polys.push(CircleCoefficients::new(
            (0..1 << LIFTING_LOG_SIZE).map(M31::from).collect(),
        ));
        polys
    }

    fn prove_and_verify_pcs<
        B: BackendForChannel<Blake2sMerkleChannel>,
        const STORE_COEFFS: bool,
    >() -> Result<(), VerificationError> {
        const N_COLS: usize = 10;
        const LIFTING_LOG_SIZE: u32 = 8;

        // Setup the prover side of the pcs.
        let mut channel = Blake2sChannel::default();
        let config = PcsConfig::default();
        let twiddles = B::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(config, &twiddles);
        if STORE_COEFFS {
            commitment_scheme.set_store_polynomials_coefficients();
        }
        let polys = prepare_polys::<B, N_COLS, LIFTING_LOG_SIZE>();
        let sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut channel);

        let mut rng = SmallRng::seed_from_u64(0);
        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let samples = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![(0..N_COLS)
            .zip(mask_structure.iter())
            .map(|(_, i)| samples.into_iter().take(*i).collect_vec())
            .collect_vec()];

        let proof = commitment_scheme.prove_values(TreeVec(sampled_points.clone()), &mut channel);

        // Verifier side of the pcs.
        let mut channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(proof.proof.commitments[0], &sizes, &mut channel);
        verifier.verify_values(TreeVec(sampled_points), proof.proof, &mut channel)
    }

    #[test]
    fn test_pcs_prove_and_verify_cpu() {
        assert!(prove_and_verify_pcs::<CpuBackend, true>().is_ok());
    }
    #[test]
    fn test_pcs_prove_and_verify_simd() {
        assert!(prove_and_verify_pcs::<SimdBackend, true>().is_ok());
    }
    #[test]
    fn test_pcs_prove_and_verify_simd_with_barycentric() {
        assert!(prove_and_verify_pcs::<SimdBackend, false>().is_ok());
    }

    /// Tests that SIMD quotient computation produces low-degree quotients even when the trace
    /// polynomial is very small (subdomain size < N_LANES).
    #[test]
    fn test_simd_quotients_are_low_degree_small_trace() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 4;

        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(M31::from).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let cpu_eval = polynomial.evaluate(eval_domain);
        let simd_eval = CircleEvaluation::new(eval_domain, BaseColumn::from_cpu(&cpu_eval.values));

        let sample_points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = sample_points
            .into_iter()
            .map(|x| PointSample {
                point: x,
                value: polynomial.eval_at_point(x),
            })
            .collect_vec();
        let rand_coeff =
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));
        let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);
        let quot_eval = compute_fri_quotients(
            &TreeVec(vec![vec![&simd_eval]]),
            &TreeVec(vec![vec![samples]]),
            rand_coeff,
            LOG_SIZE + LOG_BLOWUP_FACTOR,
            &twiddles,
            LOG_BLOWUP_FACTOR,
        );
        let mut coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.to_cpu()).interpolate())
            .collect_vec();
        let zeros = coeffs[0].coeffs.split_off((1 << LOG_SIZE) - 1);

        assert!(zeros.iter().all(|c| c.is_zero()));
    }

    /// Under either batching, the function the verifier's answers sample (the batched DEEP
    /// quotient, at every point of the domain) is of low degree when the sampled values are the
    /// polynomials' values, and a single wrong value makes it high degree: what FRI refuses.
    /// The prover's own quotient agrees with it on correct values.
    #[test]
    fn quotients_are_low_degree_exactly_when_the_values_are_right() {
        const LOG_SIZE: u32 = 5;
        const LOG_BLOWUP_FACTOR: u32 = 2;
        const LOG_DOMAIN: u32 = LOG_SIZE + LOG_BLOWUP_FACTOR;
        let mut rng = SmallRng::seed_from_u64(1);
        let random_felt = |rng: &mut SmallRng| {
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())))
        };
        let polys = (0..3)
            .map(|_| {
                CpuCirclePoly::new(
                    (0..1 << LOG_SIZE)
                        .map(|_| M31::from(rng.gen::<u32>()))
                        .collect(),
                )
            })
            .collect_vec();
        let domain = CanonicCoset::new(LOG_DOMAIN).circle_domain();
        let evals = polys.iter().map(|p| p.evaluate(domain)).collect_vec();
        let points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        // Columns sampled at one point, at two (with the periodicity sample), and at one.
        let samples = |wrong: bool| {
            TreeVec(vec![polys
                .iter()
                .zip([1, 2, 1])
                .enumerate()
                .map(|(column, (poly, n_points))| {
                    points[..n_points]
                        .iter()
                        .enumerate()
                        .map(|(i, &point)| PointSample {
                            point,
                            value: poly.eval_at_point(point)
                                + if wrong && column == 1 && i == 0 {
                                    SecureField::one()
                                } else {
                                    SecureField::zero()
                                },
                        })
                        .collect_vec()
                })
                .collect_vec()])
        };
        assert_eq!(n_deep_values(&samples(false)), 5);
        let verifier_answers = |samples, coefficients| {
            fri_answers(
                TreeVec(vec![vec![LOG_DOMAIN; 3]]),
                samples,
                coefficients,
                &(0..1 << LOG_DOMAIN).collect_vec(),
                TreeVec(vec![evals.iter().map(|e| e.values.clone()).collect_vec()]),
                LOG_DOMAIN,
            )
            .unwrap()
        };
        let degree_bound = |answers: &[SecureField]| {
            (0..4)
                .map(|coordinate| {
                    let column = answers
                        .iter()
                        .map(|a| a.to_m31_array()[coordinate])
                        .collect_vec();
                    let coeffs = CpuCircleEvaluation::new(domain, column)
                        .interpolate()
                        .coeffs;
                    coeffs
                        .iter()
                        .rposition(|c| !c.is_zero())
                        .map_or(0, |d| d + 1)
                })
                .max()
                .unwrap()
        };
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        for coefficients in [
            DeepCoefficients::Powers(random_felt(&mut rng)),
            DeepCoefficients::Affine((0..4).map(|_| random_felt(&mut rng)).collect()),
        ] {
            let right = verifier_answers(samples(false), coefficients.clone());
            assert!(degree_bound(&right) < 1 << LOG_SIZE);
            let prover = compute_fri_quotients(
                &TreeVec(vec![evals.iter().collect_vec()]),
                &samples(false),
                coefficients.clone(),
                LOG_DOMAIN,
                &twiddles,
                LOG_BLOWUP_FACTOR,
            );
            assert_eq!(prover.values.to_vec(), right);
            let wrong = verifier_answers(samples(true), coefficients);
            assert!(degree_bound(&wrong) > 1 << LOG_SIZE);
        }
    }

    /// Commits `N_COLS` polynomials on the SIMD backend, proves their values at two points
    /// (columns sampled once or twice), changes the proof by `tamper`, and verifies it under
    /// `verifier_batching`.
    fn prove_and_verify_pcs_with<MC: MerkleChannel>(
        config: PcsConfig,
        verifier_batching: DeepBatching,
        tamper: impl FnOnce(&mut CommitmentSchemeProof<MC::H>),
    ) -> Result<(), VerificationError>
    where
        SimdBackend: BackendForChannel<MC>,
    {
        const N_COLS: usize = 10;
        const LIFTING_LOG_SIZE: u32 = 8;

        let mut channel = MC::C::default();
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, MC>::new(config, &twiddles);
        let polys = prepare_polys::<SimdBackend, N_COLS, LIFTING_LOG_SIZE>();
        let sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut channel);

        let mut rng = SmallRng::seed_from_u64(0);
        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![mask_structure
            .iter()
            .map(|&n| points.into_iter().take(n).collect_vec())
            .collect_vec()];
        let mut proof = commitment_scheme
            .prove_values(TreeVec(sampled_points.clone()), &mut channel)
            .proof;
        tamper(&mut proof);

        let mut channel = MC::C::default();
        let mut verifier = CommitmentSchemeVerifier::<MC>::new(PcsConfig {
            deep_batching: verifier_batching,
            ..config
        });
        verifier.commit(proof.commitments[0], &sizes, &mut channel);
        verifier.verify_values(TreeVec(sampled_points), proof, &mut channel)
    }

    fn affine(fold_step: u32) -> PcsConfig {
        PcsConfig {
            pow_bits: 5,
            fri_config: FriConfig::new(2, 3, 8, fold_step),
            lifting_log_size: None,
            deep_batching: DeepBatching::Affine,
        }
    }

    /// A proof under affine batching verifies on the Blake2s and Blake2b channels, at fold
    /// steps 1 and 3; it fails under powers, as a powers proof fails under affine batching.
    #[test]
    fn affine_batching_proves_and_verifies() {
        for fold_step in [1, 3] {
            let config = affine(fold_step);
            prove_and_verify_pcs_with::<Blake2sMerkleChannel>(config, DeepBatching::Affine, |_| {})
                .unwrap();
            prove_and_verify_pcs_with::<Blake2bMerkleChannel>(config, DeepBatching::Affine, |_| {})
                .unwrap();
            assert!(prove_and_verify_pcs_with::<Blake2bMerkleChannel>(
                config,
                DeepBatching::Powers,
                |_| {}
            )
            .is_err());
            let powers = PcsConfig {
                deep_batching: DeepBatching::Powers,
                ..config
            };
            prove_and_verify_pcs_with::<Blake2bMerkleChannel>(powers, DeepBatching::Powers, |_| {})
                .unwrap();
            assert!(prove_and_verify_pcs_with::<Blake2bMerkleChannel>(
                powers,
                DeepBatching::Affine,
                |_| {}
            )
            .is_err());
        }
    }

    /// Under affine batching, a changed sampled value (of a column sampled once, of one sampled
    /// twice) and a changed queried value are refused.
    #[test]
    fn affine_batching_refuses_a_tampered_proof() {
        let config = affine(3);
        for column in 0..10 {
            let result = prove_and_verify_pcs_with::<Blake2bMerkleChannel>(
                config,
                DeepBatching::Affine,
                |proof| proof.sampled_values[0][column][0] += SecureField::one(),
            );
            assert!(result.is_err(), "column {column}");
        }
        let result = prove_and_verify_pcs_with::<Blake2bMerkleChannel>(
            config,
            DeepBatching::Affine,
            |proof| proof.queried_values[0][3][0] += M31::from(1),
        );
        assert!(matches!(result, Err(VerificationError::Merkle(_))));
    }

    /// Affine batching draws `B − 1` coefficients after the sampled values, one each for every
    /// value but the first, and mixes one more element into the config's transcript; powers
    /// draw one coefficient and leave the transcript as it was.
    #[test]
    fn deep_coefficients_are_drawn_one_a_value() {
        let mut powers = Blake2sChannel::default();
        let mut by_affine = powers.clone();
        let DeepCoefficients::Powers(alpha) =
            DeepCoefficients::draw(&mut powers, DeepBatching::Powers, 7)
        else {
            panic!()
        };
        let drawn = DeepCoefficients::draw(&mut by_affine, DeepBatching::Affine, 7);
        let DeepCoefficients::Affine(coefficients) = &drawn else {
            panic!()
        };
        assert_eq!(coefficients.len(), 6);
        assert_eq!(
            drawn.iter().collect_vec(),
            std::iter::once(SecureField::one())
                .chain(coefficients.iter().copied())
                .collect_vec()
        );
        assert_eq!(
            DeepCoefficients::Powers(alpha).iter().take(3).collect_vec(),
            vec![SecureField::one(), alpha, alpha * alpha]
        );
        assert_eq!(
            DeepCoefficients::from(alpha),
            DeepCoefficients::Powers(alpha)
        );

        let config = affine(1);
        let mix = |config: PcsConfig| {
            let mut channel = Blake2sChannel::default();
            config.mix_into(&mut channel);
            channel.digest()
        };
        let powers_config = PcsConfig {
            deep_batching: DeepBatching::Powers,
            ..config
        };
        assert_ne!(mix(config), mix(powers_config));
        let mut channel = Blake2sChannel::default();
        channel.mix_felts(&[
            SecureField::from_u32_unchecked(5, 3, 8, 2),
            SecureField::from_u32_unchecked(1, 0, 0, 0),
        ]);
        assert_eq!(mix(powers_config), channel.digest());
    }
}
