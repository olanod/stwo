use hashbrown::HashMap;
use itertools::Itertools;
use num_traits::Zero;
use tracing::instrument;

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fri::{
    ExtendedFriLayerProof, ExtendedFriProof, FriConfig, FriLayerProof, FriLayerProofAux, FriProof,
    FriProofAux,
};
use crate::core::poly::line::LinePoly;
use crate::core::queries::{draw_queries, Queries};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::LOG_PACKED_LEAF_SIZE;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;

pub trait FriOps: ColumnOps<BaseField> + PolyOps + Sized + ColumnOps<SecureField> {
    /// Folds a degree `d` polynomial into a degree `d / 2^k` polynomial, where `k = alphas.len()`.
    ///
    /// For i ∈ [0, k), the i-th fold computes the evaluation of `f_{i+1} = f_{i,0} + alphas[i] *
    /// f_{i,1}` on `E_{i+1} = pi(E_i)`, where:
    /// * `f_i` is the folded polynomial at fold i (`f_0 = eval`), evaluated on line domain `E_i`
    ///   (`E_0 = eval.domain()`).
    /// * `pi(x) = 2x^2 - 1` is the doubling map.
    /// * `f_{i,0}` and `f_{i,1}` are the polynomials determined by the identity `2*f_i(x) =
    ///   f_{i,0}(pi(x)) + x * f_{i,1}(pi(x))`.
    ///
    /// # Panics
    ///
    /// Panics if `alphas` is empty.
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alphas: &[SecureField],
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self>;

    /// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate
    /// polynomial.
    ///
    /// Let `src` be the evaluation of a circle polynomial `f` on a
    /// [`CircleDomain`] `E`. This function computes evaluations of `f' = f0
    /// + alpha * f1` on the x-coordinates of `E` such that `2f(p) = f0(px) + py * f1(px)`. The
    /// evaluations of `f'` are accumulated into `dst` by the formula `dst = dst * alpha^2 + f'`.
    ///
    /// # Panics
    ///
    /// Panics if `src` is not double the length of `dst`.
    ///
    /// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
    // TODO(andrew): Make folding factor generic.
    // TODO(andrew): Fold directly into FRI layer to prevent allocation.
    fn fold_circle_into_line(
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self>;

    /// Decomposes a FRI-space polynomial into a polynomial inside the fft-space and the
    /// remainder term.
    /// FRI-space: polynomials of total degree n/2.
    /// Based on lemma #12 from the CircleStark paper: f(P) = g(P)+ lambda * alternating(P),
    /// where lambda is the cosset diff of eval, and g is a polynomial in the fft-space.
    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField);
}

/// Computes `len` folding alphas derived by repeated squaring: `[alpha, alpha^2, alpha^4, ...]`.
pub fn squared_alpha_powers(alpha: SecureField, len: u32) -> Vec<SecureField> {
    let mut alphas = Vec::with_capacity(len as usize);
    let mut alpha = alpha;
    for _ in 0..len {
        alphas.push(alpha);
        alpha = alpha * alpha;
    }
    alphas
}

pub struct FriDecommitResult<H: MerkleHasherLifted> {
    pub fri_proof: ExtendedFriProof<H>,
    pub query_positions: Vec<usize>,
    pub unsorted_query_locations: Vec<usize>,
}

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
pub struct FriProver<'a, B: FriOps + MerkleOpsLifted<MC::H>, MC: MerkleChannel> {
    config: FriConfig,
    first_layer: FriFirstLayerProver<'a, B, MC::H>,
    inner_layers: Vec<FriInnerLayerProver<B, MC::H>>,
    last_layer_poly: LinePoly,
}
impl<'a, B: FriOps + MerkleOpsLifted<MC::H>, MC: MerkleChannel> FriProver<'a, B, MC> {
    /// Runs the commitment phase of FRI on one circle evaluation over a canonic circle domain.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation is not from a sufficiently low degree circle polynomial.
    /// * The evaluation domain is not a canonic circle domain.
    #[instrument(skip_all)]
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        column: &'a SecureEvaluation<B, BitReversedOrder>,
        twiddles: &TwiddleTree<B>,
    ) -> Self {
        assert!(column.domain.is_canonic(), "not canonic");

        let first_layer = Self::commit_first_layer(channel, &config, column);
        let (inner_layers, last_layer_evaluation) =
            Self::commit_inner_layers(channel, config, column, twiddles);
        let last_layer_poly = Self::commit_last_layer(channel, config, last_layer_evaluation);

        Self {
            config,
            first_layer,
            inner_layers,
            last_layer_poly,
        }
    }

    /// Commits to the first FRI layer.
    fn commit_first_layer(
        channel: &mut MC::C,
        config: &FriConfig,
        column: &'a SecureEvaluation<B, BitReversedOrder>,
    ) -> FriFirstLayerProver<'a, B, MC::H> {
        // The circle-to-line fold is always equal to the config.fold_step.
        // TODO(Leo): consider support for smaller steps.
        let layer = FriFirstLayerProver::new(column, config.fold_step);
        MC::mix_root(channel, layer.merkle_tree.root());
        layer
    }

    /// Builds and commits to the inner FRI layers (all layers except the first and last).
    ///
    /// Returns all inner layers and the evaluation of the last layer.
    fn commit_inner_layers(
        channel: &mut MC::C,
        config: FriConfig,
        column: &SecureEvaluation<B, BitReversedOrder>,
        twiddles: &TwiddleTree<B>,
    ) -> (Vec<FriInnerLayerProver<B, MC::H>>, LineEvaluation<B>) {
        let mut layers = Vec::new();
        let folding_alpha = channel.draw_secure_felt();

        let mut layer_evaluation = B::fold_circle_into_line(column, folding_alpha, twiddles);
        let mut line_log_size = layer_evaluation.domain().log_size();

        // Apply any additional line folds requested for the first stage.
        if config.fold_step > 1 {
            let extra_line_folds = config.fold_step - 1;
            let alpha_sq_powers =
                squared_alpha_powers(folding_alpha * folding_alpha, extra_line_folds);
            layer_evaluation = B::fold_line(&layer_evaluation, &alpha_sq_powers, twiddles);
            line_log_size -= extra_line_folds;
        }

        let last_layer_log_domain_size = config.last_layer_domain_size().ilog2();
        assert!(
            line_log_size >= last_layer_log_domain_size,
            "The circle-to-line fold results in a smaller line domain than the last layer."
        );
        // If we're already at the last layer, there are no inner layers to compute.
        if line_log_size == last_layer_log_domain_size {
            return (layers, layer_evaluation);
        }
        // While we can, skip `config.fold_step` layers.
        while line_log_size > last_layer_log_domain_size + config.fold_step {
            let layer = FriInnerLayerProver::new(layer_evaluation, config.fold_step);
            MC::mix_root(channel, layer.merkle_tree.root());
            let folding_alpha = channel.draw_secure_felt();
            let alpha_sq_powers = squared_alpha_powers(folding_alpha, config.fold_step);
            layer_evaluation = B::fold_line(&layer.evaluation, &alpha_sq_powers, twiddles);
            layers.push(layer);
            line_log_size -= config.fold_step;
        }

        // Do one last fold (of size 0 < k <= config.fold_step) to reach the correct size.
        let last_fold_step = line_log_size - last_layer_log_domain_size;
        let layer = FriInnerLayerProver::new(layer_evaluation, last_fold_step);
        MC::mix_root(channel, layer.merkle_tree.root());
        let folding_alpha = channel.draw_secure_felt();
        let alpha_sq_powers = squared_alpha_powers(folding_alpha, last_fold_step);
        layer_evaluation = B::fold_line(&layer.evaluation, &alpha_sq_powers, twiddles);
        layers.push(layer);

        (layers, layer_evaluation)
    }

    /// Builds and commits to the last layer.
    ///
    /// The layer is committed to by sending the verifier all the coefficients of the remaining
    /// polynomial.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation domain size exceeds the maximum last layer domain size.
    /// * The evaluation is not of sufficiently low degree.
    fn commit_last_layer(
        channel: &mut MC::C,
        config: FriConfig,
        evaluation: LineEvaluation<B>,
    ) -> LinePoly {
        assert_eq!(evaluation.len(), config.last_layer_domain_size());

        let evaluation = evaluation.to_cpu();
        let mut coeffs = evaluation.interpolate().into_ordered_coefficients();

        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        let zeros = coeffs.split_off(last_layer_degree_bound);
        assert!(zeros.iter().all(SecureField::is_zero), "invalid degree");

        let last_layer_poly = LinePoly::from_ordered_coefficients(coeffs);
        channel.mix_felts(&last_layer_poly);

        last_layer_poly
    }

    /// Returns a FRI proof and the query positions.
    pub fn decommit(self, channel: &mut MC::C) -> FriDecommitResult<MC::H> {
        let first_layer_log_size = self.first_layer.column.domain.log_size();
        let unsorted_query_locations =
            draw_queries(channel, first_layer_log_size, self.config.n_queries);
        let queries = Queries::new(&unsorted_query_locations, first_layer_log_size);

        let fri_proof = self.decommit_on_queries(&queries);
        FriDecommitResult {
            fri_proof,
            query_positions: queries.positions,
            unsorted_query_locations,
        }
    }

    /// # Panics
    ///
    /// Panics if the queries were sampled on the wrong domain size.
    pub fn decommit_on_queries(self, queries: &Queries) -> ExtendedFriProof<MC::H> {
        let Self {
            config,
            first_layer,
            inner_layers,
            last_layer_poly,
        } = self;

        let first_layer_proof = first_layer.decommit(queries, config.fold_step);

        let inner_layer_proofs = inner_layers
            .into_iter()
            .scan(queries.fold(config.fold_step), |layer_queries, layer| {
                let fold_step = layer.fold_step;
                let layer_proof = layer.decommit(layer_queries);
                *layer_queries = layer_queries.fold(fold_step);
                Some(layer_proof)
            })
            .collect_vec();

        let (inner_proofs, inner_layers_aux): (Vec<_>, Vec<_>) = inner_layer_proofs
            .into_iter()
            .map(|p| (p.proof, p.aux))
            .unzip();

        ExtendedFriProof {
            proof: FriProof {
                first_layer: first_layer_proof.proof,
                inner_layers: inner_proofs,
                last_layer_poly,
            },
            aux: FriProofAux {
                first_layer: first_layer_proof.aux,
                inner_layers: inner_layers_aux,
            },
        }
    }
}

/// Commitment to the first FRI layer.
struct FriFirstLayerProver<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    column: &'a SecureEvaluation<B, BitReversedOrder>,
    merkle_tree: MerkleProverLifted<B, H>,
    pack_leaves: bool,
}

impl<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> FriFirstLayerProver<'a, B, H> {
    fn new(first_layer_column: &'a SecureEvaluation<B, BitReversedOrder>, fold_step: u32) -> Self {
        let pack_leaves =
            first_layer_column.values.len().ilog2() >= LOG_PACKED_LEAF_SIZE && fold_step > 1;
        let log_rows_per_leaf = if pack_leaves { LOG_PACKED_LEAF_SIZE } else { 0 };
        let merkle_tree = MerkleProverLifted::commit(
            first_layer_column.values.columns.iter().collect_vec(),
            first_layer_column.values.len().ilog2() - log_rows_per_leaf,
            log_rows_per_leaf,
        );

        FriFirstLayerProver {
            column: first_layer_column,
            merkle_tree,
            pack_leaves,
        }
    }

    fn decommit(self, queries: &Queries, fold_step: u32) -> ExtendedFriLayerProof<H> {
        assert_eq!(queries.log_domain_size, self.column.domain.log_size());

        let (decommitment_positions, column_witness, value_map) =
            compute_decommitment_positions_and_witness_evals(
                self.column,
                &queries.positions,
                fold_step,
            );

        let decommitment_positions = if self.pack_leaves {
            decommitment_positions
                .iter()
                .map(|position| position >> LOG_PACKED_LEAF_SIZE)
                .dedup()
                .collect_vec()
        } else {
            decommitment_positions
        };
        // We can pass an empty vector to the merkle decommit because we don't use its returned
        // opened values.
        // TODO(Leo): consider adding a method to merkle prover to decommit only the auth paths.
        let (_, decommitment) = self
            .merkle_tree
            .decommit(&decommitment_positions, Vec::<&Col<B, BaseField>>::new());
        let commitment = self.merkle_tree.root();

        ExtendedFriLayerProof {
            proof: FriLayerProof {
                fri_witness: column_witness,
                decommitment: decommitment.decommitment,
                commitment,
            },
            aux: FriLayerProofAux {
                all_values: vec![value_map],
                decommitment: decommitment.aux,
            },
        }
    }
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): Support different step sizes and update docs.
// TODO(andrew): The docs are wrong. Each leaf of the merkle tree commits to a single
// QM31 value. This is inefficient and should be changed.
struct FriInnerLayerProver<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    evaluation: LineEvaluation<B>,
    merkle_tree: MerkleProverLifted<B, H>,
    fold_step: u32,
    pack_leaves: bool,
}

impl<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> FriInnerLayerProver<B, H> {
    fn new(evaluation: LineEvaluation<B>, fold_step: u32) -> Self {
        let pack_leaves = evaluation.values.len().ilog2() >= LOG_PACKED_LEAF_SIZE && fold_step > 1;
        let log_rows_per_leaf = if pack_leaves { LOG_PACKED_LEAF_SIZE } else { 0 };
        let merkle_tree = MerkleProverLifted::commit(
            evaluation.values.columns.iter().collect_vec(),
            evaluation.values.len().ilog2() - log_rows_per_leaf,
            log_rows_per_leaf,
        );

        FriInnerLayerProver {
            evaluation,
            merkle_tree,
            fold_step,
            pack_leaves,
        }
    }

    fn decommit(self, queries: &Queries) -> ExtendedFriLayerProof<H> {
        let (decommitment_positions, fri_witness, value_map) =
            compute_decommitment_positions_and_witness_evals(
                &self.evaluation.values,
                queries,
                self.fold_step,
            );

        let decommitment_positions = if self.pack_leaves {
            decommitment_positions
                .iter()
                .map(|position| position >> LOG_PACKED_LEAF_SIZE)
                .dedup()
                .collect_vec()
        } else {
            decommitment_positions
        };
        // We can pass an empty vector to the merkle decommit because we don't use its returned
        // opened values.
        let (_, decommitment) = self
            .merkle_tree
            .decommit(&decommitment_positions, Vec::<&Col<B, BaseField>>::new());
        let commitment = self.merkle_tree.root();

        ExtendedFriLayerProof {
            proof: FriLayerProof {
                fri_witness,
                decommitment: decommitment.decommitment,
                commitment,
            },
            aux: FriLayerProofAux {
                all_values: vec![value_map],
                decommitment: decommitment.aux,
            },
        }
    }
}

/// Returns a column's merkle tree decommitment positions and the evals the verifier can't
/// deduce from previous computations but requires for decommitment and folding.
///
/// Returns a map from leaf index to its value.
fn compute_decommitment_positions_and_witness_evals(
    column: &SecureColumnByCoords<impl PolyOps>,
    query_positions: &[usize],
    fold_step: u32,
) -> (Vec<usize>, Vec<QM31>, HashMap<usize, QM31>) {
    let mut decommitment_positions = Vec::new();
    let mut witness_evals = Vec::new();
    let mut value_map = HashMap::new();

    // Group queries by the folding coset they reside in.
    for subset_queries in query_positions.chunk_by(|a, b| a >> fold_step == b >> fold_step) {
        let subset_start = (subset_queries[0] >> fold_step) << fold_step;
        let subset_decommitment_positions = subset_start..subset_start + (1 << fold_step);
        let mut subset_queries_iter = subset_queries.iter().peekable();

        for position in subset_decommitment_positions {
            // Add decommitment position.
            decommitment_positions.push(position);

            let eval = column.at(position);
            value_map.insert(position, eval);

            // Only add evals the verifier can't calculate.
            if subset_queries_iter.next_if_eq(&&position).is_none() {
                witness_evals.push(eval);
            }
        }
    }

    (decommitment_positions, witness_evals, value_map)
}

#[cfg(test)]
mod tests {

    use num_traits::One;

    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
    use crate::core::fri::FriConfig;
    use crate::core::poly::circle::CircleDomain;
    use crate::core::queries::Queries;
    use crate::core::test_utils::test_channel;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Col, Column, CpuBackend};
    use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::prover::vcs_lifted::ops::PackLeavesOps;

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;

    type FriProver<'a> = super::FriProver<'a, CpuBackend, Blake2sMerkleChannel>;

    #[test]
    #[should_panic = "invalid degree"]
    fn committing_high_degree_polynomial_fails() {
        const LOG_EXPECTED_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR;
        const LOG_INVALID_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR - 1;
        let config = FriConfig::new(2, LOG_EXPECTED_BLOWUP_FACTOR, 3, 1);
        let column = polynomial_evaluation(6, LOG_INVALID_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        FriProver::commit(&mut test_channel(), config, &column, &twiddles);
    }

    #[test]
    #[should_panic = "not canonic"]
    fn committing_column_from_invalid_domain_fails() {
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 3));
        assert!(!invalid_domain.is_canonic(), "must be an invalid domain");
        let config = FriConfig::new(2, 2, 3, 1);
        let column = SecureEvaluation::new(
            invalid_domain,
            [SecureField::one(); 1 << 4].into_iter().collect(),
        );
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        FriProver::commit(&mut test_channel(), config, &column, &twiddles);
    }

    /// Returns an evaluation of a random polynomial with degree `2^log_degree`.
    ///
    /// The evaluation domain size is `2^(log_degree + log_blowup_factor)`.
    fn polynomial_evaluation(
        log_degree: u32,
        log_blowup_factor: u32,
    ) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
        let poly = CpuCirclePoly::new(vec![BaseField::one(); 1 << log_degree]);
        let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
        let domain = CircleDomain::new(coset);
        let values = poly.evaluate(domain);
        SecureEvaluation::new(domain, values.into_iter().map(SecureField::from).collect())
    }

    #[test]
    fn test_fri_commit_decommit_with_jumps() {
        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, 2);
        let column = polynomial_evaluation(6, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        let prover = FriProver::commit(&mut test_channel(), config, &column, &twiddles);
        let queries = Queries::from_positions(vec![0, 3], 6 + LOG_BLOWUP_FACTOR);
        prover.decommit_on_queries(&queries);
    }

    #[test]
    fn test_fri_commit_decommit_with_packed_leaves() {
        for fold_step in 2..=4 {
            let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, fold_step);
            let column = polynomial_evaluation(8, LOG_BLOWUP_FACTOR);
            let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

            let prover = FriProver::commit(&mut test_channel(), config, &column, &twiddles);
            let queries = Queries::from_positions(vec![1, 6, 11], 8 + LOG_BLOWUP_FACTOR);
            prover.decommit_on_queries(&queries);
        }
    }

    #[test]
    fn test_fri_commit_decommit_with_packed_leaves_simd() {
        for fold_step in 2..=4 {
            let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, fold_step);
            let cpu_eval = polynomial_evaluation(8, LOG_BLOWUP_FACTOR);
            let column = SecureEvaluation::new(
                cpu_eval.domain,
                cpu_eval.values.to_vec().into_iter().collect(),
            );
            let twiddles = SimdBackend::precompute_twiddles(column.domain.half_coset);
            let prover = super::FriProver::<'_, SimdBackend, Blake2sMerkleChannel>::commit(
                &mut test_channel(),
                config,
                &column,
                &twiddles,
            );
            let queries = Queries::from_positions(vec![1, 6, 11], 8 + LOG_BLOWUP_FACTOR);
            prover.decommit_on_queries(&queries);
        }
    }

    #[test]
    fn test_fri_commit_decommit_keccak256_cpu() {
        use crate::core::channel::Keccak256Channel;
        use crate::core::vcs_lifted::keccak256_merkle::Keccak256MerkleChannel;

        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, 2);
        let column = polynomial_evaluation(6, LOG_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        let prover = super::FriProver::<'_, CpuBackend, Keccak256MerkleChannel>::commit(
            &mut Keccak256Channel::default(),
            config,
            &column,
            &twiddles,
        );
        let queries = Queries::from_positions(vec![0, 3], 6 + LOG_BLOWUP_FACTOR);
        prover.decommit_on_queries(&queries);
    }

    #[test]
    fn test_fri_commit_decommit_keccak256_simd() {
        use crate::core::channel::Keccak256Channel;
        use crate::core::vcs_lifted::keccak256_merkle::Keccak256MerkleChannel;

        let config = FriConfig::new(2, LOG_BLOWUP_FACTOR, 3, 2);
        let cpu_eval = polynomial_evaluation(8, LOG_BLOWUP_FACTOR);
        let column = SecureEvaluation::new(
            cpu_eval.domain,
            cpu_eval.values.to_vec().into_iter().collect(),
        );
        let twiddles = SimdBackend::precompute_twiddles(column.domain.half_coset);
        let prover = super::FriProver::<'_, SimdBackend, Keccak256MerkleChannel>::commit(
            &mut Keccak256Channel::default(),
            config,
            &column,
            &twiddles,
        );
        let queries = Queries::from_positions(vec![1, 6, 11], 8 + LOG_BLOWUP_FACTOR);
        prover.decommit_on_queries(&queries);
    }

    #[test]
    fn test_pack_leaves_input_simd_matches_cpu() {
        for log_size in 2..8 {
            let values = (0..1 << log_size).map(|i| {
                SecureField::from_m31_array(core::array::from_fn(|coord| {
                    BaseField::from_u32_unchecked((i * SECURE_EXTENSION_DEGREE + coord) as u32)
                }))
            });
            let values = SecureColumnByCoords::<SimdBackend>::from_iter(values);

            let cpu_values = values.to_cpu();
            let cpu_cols: [&Col<CpuBackend, BaseField>; SECURE_EXTENSION_DEGREE] =
                core::array::from_fn(|i| &cpu_values.columns[i]);
            let generic: [Col<CpuBackend, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
                <CpuBackend as PackLeavesOps>::pack_leaves_input(&cpu_cols);
            let simd_cols: [&Col<SimdBackend, BaseField>; SECURE_EXTENSION_DEGREE] =
                core::array::from_fn(|i| &values.columns[i]);
            let simd: [Col<SimdBackend, BaseField>; SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE] =
                <SimdBackend as PackLeavesOps>::pack_leaves_input(&simd_cols);

            for col in 0..SECURE_EXTENSION_DEGREE * PACKED_LEAF_SIZE {
                assert_eq!(
                    generic[col].to_cpu(),
                    simd[col].to_cpu(),
                    "mismatch at log_size={log_size}, col={col}"
                );
            }
        }
    }
}

/// Fold step 3, as the final wrap uses it: each check a cheating prover can reach refuses at the
/// layer it belongs to. The column is a random polynomial of degree bound `2^12` at blowup 4, so
/// FRI folds the circle and two lines into a line of degree bound `2^9` (first layer, packed
/// leaves), then `2^6` and `2^3` (inner layers 0 and 1, packed), then `2^2` (inner layer 2, one
/// fold, unpacked leaves), the last layer.
#[cfg(test)]
mod fold_step_3_tests {
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{squared_alpha_powers, FriFirstLayerProver, FriInnerLayerProver, FriOps};
    use crate::core::channel::{Blake2bChannel, Channel, MerkleChannel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fri::{
        CirclePolyDegreeBound, FriConfig, FriLayerProof, FriProof, FriVerificationError,
    };
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::poly::line::LinePoly;
    use crate::core::queries::Queries;
    use crate::core::vcs::blake2b_hash::Blake2bHash;
    use crate::core::vcs_lifted::blake2b_merkle::{Blake2bMerkleChannel, Blake2bMerkleHasher};
    use crate::core::vcs_lifted::verifier::MerkleVerificationError;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::CpuBackend;
    use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
    use crate::prover::poly::twiddles::TwiddleTree;
    use crate::prover::poly::BitReversedOrder;

    type MC = Blake2bMerkleChannel;
    type FriProver<'a> = super::FriProver<'a, CpuBackend, MC>;
    type FriVerifier = crate::core::fri::FriVerifier<MC>;
    type Column = SecureEvaluation<CpuBackend, BitReversedOrder>;

    const LOG_DEGREE: u32 = 12;
    const LOG_BLOWUP: u32 = 2;
    const CONFIG: FriConfig = FriConfig {
        log_blowup_factor: LOG_BLOWUP,
        log_last_layer_degree_bound: 2,
        n_queries: 3,
        fold_step: 3,
    };
    /// Positions in the first layer's domain of `2^14`; 4 and 7 share a coset of 8.
    const POSITIONS: [usize; 4] = [4, 7, 5000, 16000];

    fn column() -> (Column, TwiddleTree<CpuBackend>) {
        let mut rng = SmallRng::seed_from_u64(3);
        let poly = CpuCirclePoly::new(
            (0..1 << LOG_DEGREE)
                .map(|_| BaseField::from(rng.gen_range(0..1u32 << 30)))
                .collect(),
        );
        let domain = CanonicCoset::new(LOG_DEGREE + LOG_BLOWUP).circle_domain();
        let column = SecureEvaluation::new(
            domain,
            poly.evaluate(domain)
                .into_iter()
                .map(SecureField::from)
                .collect(),
        );
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        (column, twiddles)
    }

    fn queries() -> Queries {
        Queries::from_positions(POSITIONS.to_vec(), LOG_DEGREE + LOG_BLOWUP)
    }

    /// The first-layer values at the queries: the DEEP answers an honest PCS hands FRI.
    fn answers(column: &Column) -> Vec<SecureField> {
        queries().iter().map(|&p| column.at(p)).collect()
    }

    /// [`super::FriProver::commit`], except that inner layer `bad_layer`'s committed values at
    /// the given positions are changed by one; the next layer is folded from the honest values.
    /// Every root is mixed and every alpha drawn as in an honest proof, and every Merkle path
    /// opens what was committed: only the fold into the bad layer disagrees with it.
    fn commit_with_bad_layer<'a>(
        column: &'a Column,
        twiddles: &TwiddleTree<CpuBackend>,
        bad_layer: usize,
        bad_positions: &[usize],
    ) -> FriProver<'a> {
        let channel = &mut Blake2bChannel::default();
        let first_layer = FriFirstLayerProver::new(column, CONFIG.fold_step);
        MC::mix_root(channel, first_layer.merkle_tree.root());
        let alpha = channel.draw_secure_felt();
        let line = CpuBackend::fold_circle_into_line(column, alpha, twiddles);
        let mut evaluation = CpuBackend::fold_line(
            &line,
            &squared_alpha_powers(alpha * alpha, CONFIG.fold_step - 1),
            twiddles,
        );
        let last_log_size = CONFIG.last_layer_domain_size().ilog2();
        let mut inner_layers = Vec::new();
        while evaluation.domain().log_size() > last_log_size {
            let step = CONFIG
                .fold_step
                .min(evaluation.domain().log_size() - last_log_size);
            let mut committed = evaluation.clone();
            if inner_layers.len() == bad_layer {
                for &position in bad_positions {
                    let value = committed.values.at(position) + SecureField::one();
                    committed.values.set(position, value);
                }
            }
            let layer = FriInnerLayerProver::new(committed, step);
            MC::mix_root(channel, layer.merkle_tree.root());
            let alpha = channel.draw_secure_felt();
            evaluation =
                CpuBackend::fold_line(&evaluation, &squared_alpha_powers(alpha, step), twiddles);
            inner_layers.push(layer);
        }
        let last_layer_poly = FriProver::commit_last_layer(channel, CONFIG, evaluation);
        super::FriProver {
            config: CONFIG,
            first_layer,
            inner_layers,
            last_layer_poly,
        }
    }

    fn honest_proof(
        column: &Column,
        twiddles: &TwiddleTree<CpuBackend>,
    ) -> FriProof<<MC as MerkleChannel>::H> {
        FriProver::commit(&mut Blake2bChannel::default(), CONFIG, column, twiddles)
            .decommit_on_queries(&queries())
            .proof
    }

    fn verify(
        proof: FriProof<<MC as MerkleChannel>::H>,
        answers: Vec<SecureField>,
    ) -> Result<(), FriVerificationError> {
        let bound = CirclePolyDegreeBound::new(LOG_DEGREE);
        FriVerifier::commit(&mut Blake2bChannel::default(), CONFIG, proof, bound)?
            .decommit_on_queries(&queries(), answers)
    }

    /// The query positions in inner layer `layer`'s domain.
    fn layer_positions(layer: usize) -> Vec<usize> {
        let mut queries = queries().fold(CONFIG.fold_step);
        for _ in 0..layer {
            queries = queries.fold(CONFIG.fold_step);
        }
        queries.positions
    }

    #[test]
    fn the_honest_proof_verifies_with_the_expected_layers() {
        let (column, twiddles) = column();
        let proof = honest_proof(&column, &twiddles);
        assert_eq!(proof.inner_layers.len(), 3);
        assert_eq!(
            proof.last_layer_poly.len(),
            1 << CONFIG.log_last_layer_degree_bound
        );
        verify(proof, answers(&column)).unwrap();
        // The cheating prover's construction, with nothing changed, is the honest proof.
        let unchanged = commit_with_bad_layer(&column, &twiddles, usize::MAX, &[])
            .decommit_on_queries(&queries())
            .proof;
        verify(unchanged, answers(&column)).unwrap();
    }

    /// A committed value that is not the fold of the layer before it, at a queried position, is
    /// refused by that layer's opening, where the verifier's own fold stands in for it: after
    /// the circle fold and two line folds (inner layer 0), after three line folds (inner layer
    /// 1) and after one (inner layer 2).
    #[test]
    fn a_changed_folded_value_is_refused_at_its_layer() {
        let (column, twiddles) = column();
        for layer in 0..3 {
            let positions = layer_positions(layer);
            let proof = commit_with_bad_layer(&column, &twiddles, layer, &positions[..1])
                .decommit_on_queries(&queries())
                .proof;
            assert!(
                matches!(
                    verify(proof, answers(&column)),
                    Err(FriVerificationError::InnerLayerCommitmentInvalid {
                        inner_layer,
                        error: MerkleVerificationError::RootMismatch,
                    }) if inner_layer == layer
                ),
                "layer {layer}"
            );
        }
    }

    /// The same change at a position no query reaches goes unseen: the check is the fold at
    /// the queries, so the queries' number sets its soundness.
    #[test]
    fn a_changed_value_off_the_queries_is_not_opened() {
        let (column, twiddles) = column();
        let queried = layer_positions(1);
        let unqueried = (0..1 << 9)
            .find(|p| queried.iter().all(|q| q >> 3 != p >> 3))
            .unwrap();
        let proof = commit_with_bad_layer(&column, &twiddles, 1, &[unqueried])
            .decommit_on_queries(&queries())
            .proof;
        verify(proof, answers(&column)).unwrap();
    }

    /// DEEP answers other than the committed first layer are refused by its opening.
    #[test]
    fn a_wrong_first_layer_answer_is_refused() {
        let (column, twiddles) = column();
        let mut answers = answers(&column);
        answers[2] += SecureField::one();
        assert!(matches!(
            verify(honest_proof(&column, &twiddles), answers),
            Err(FriVerificationError::FirstLayerCommitmentInvalid {
                error: MerkleVerificationError::RootMismatch
            })
        ));
    }

    /// A packed leaf holds 4 adjacent values of a coset of 8: a changed value, two values of
    /// one leaf swapped, a changed sibling hash, and a witness value too few or too many are each
    /// refused, in the first layer and in a packed inner layer.
    #[test]
    fn a_wrong_packed_leaf_is_refused() {
        let (column, twiddles) = column();
        type Tamper = fn(&mut FriLayerProof<Blake2bMerkleHasher>);
        let tampers: [(&str, Tamper); 3] = [
            ("changed value", |layer| {
                layer.fri_witness[0] += SecureField::one()
            }),
            ("swapped values", |layer| layer.fri_witness.swap(0, 1)),
            ("changed sibling", |layer| {
                layer.decommitment.hash_witness[0] = Blake2bHash([7; 32])
            }),
        ];
        for (name, tamper) in tampers {
            let mut proof = honest_proof(&column, &twiddles);
            tamper(&mut proof.first_layer);
            assert!(
                matches!(
                    verify(proof, answers(&column)),
                    Err(FriVerificationError::FirstLayerCommitmentInvalid { .. })
                ),
                "first layer, {name}"
            );
            let mut proof = honest_proof(&column, &twiddles);
            tamper(&mut proof.inner_layers[1]);
            assert!(
                matches!(
                    verify(proof, answers(&column)),
                    Err(FriVerificationError::InnerLayerCommitmentInvalid { inner_layer: 1, .. })
                ),
                "inner layer, {name}"
            );
        }
        let mut proof = honest_proof(&column, &twiddles);
        proof.first_layer.fri_witness.pop();
        assert!(matches!(
            verify(proof, answers(&column)),
            Err(FriVerificationError::FirstLayerEvaluationsInvalid)
        ));
        let mut proof = honest_proof(&column, &twiddles);
        proof.inner_layers[1].fri_witness.push(SecureField::one());
        assert!(matches!(
            verify(proof, answers(&column)),
            Err(FriVerificationError::InnerLayerEvaluationsInvalid { inner_layer: 1 })
        ));
    }

    /// The last layer: a changed coefficient, a polynomial of a higher degree bound, a
    /// coefficient count other than the stated size (refused, not a panic), and a missing inner
    /// layer.
    #[test]
    fn a_wrong_last_layer_is_refused() {
        let (column, twiddles) = column();
        let mut proof = honest_proof(&column, &twiddles);
        proof.last_layer_poly[1] += SecureField::one();
        assert!(matches!(
            verify(proof, answers(&column)),
            Err(FriVerificationError::LastLayerEvaluationsInvalid)
        ));

        let mut proof = honest_proof(&column, &twiddles);
        proof.last_layer_poly = LinePoly::new(vec![SecureField::one(); 8]);
        assert!(matches!(
            verify(proof, answers(&column)),
            Err(FriVerificationError::LastLayerDegreeInvalid)
        ));

        for (len, log_size) in [(3, 2), (4, 1), (1, 64), (2, 65)] {
            let mut proof = honest_proof(&column, &twiddles);
            proof.last_layer_poly =
                LinePoly::from_raw_parts(vec![SecureField::one(); len], log_size);
            assert!(
                matches!(
                    verify(proof, answers(&column)),
                    Err(FriVerificationError::LastLayerDegreeInvalid)
                ),
                "{len} coefficients, log size {log_size}"
            );
        }

        let mut proof = honest_proof(&column, &twiddles);
        proof.inner_layers.pop();
        assert!(matches!(
            verify(proof, answers(&column)),
            Err(FriVerificationError::InvalidNumFriLayers)
        ));
    }
}
