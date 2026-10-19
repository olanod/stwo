use core::iter;
use core::ops::Add;

use itertools::{izip, zip_eq, Either, Itertools};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std_shims::{ToString, Vec};

use super::{DeepBatching, TreeVec};
use crate::core::channel::Channel;
use crate::core::circle::CirclePoint;
use crate::core::constraints::complex_conjugate_line_coeffs;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::fri::{FriProof, FriProofAux};
use crate::core::pcs::PcsConfig;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux};
use crate::core::verifier::VerificationError;
use crate::core::ColumnVec;
// Used for no_std support.
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, core::hash::BuildHasherDefault<fnv::FnvHasher>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentSchemeProof<H: MerkleHasherLifted> {
    pub config: PcsConfig,
    pub commitments: TreeVec<H::Hash>,
    pub sampled_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    pub decommitments: TreeVec<MerkleDecommitmentLifted<H>>,
    pub queried_values: TreeVec<ColumnVec<Vec<BaseField>>>,
    pub proof_of_work: u64,
    pub fri_proof: FriProof<H>,
}

/// Auxiliary data for a [CommitmentSchemeProof].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentSchemeProofAux<H: MerkleHasherLifted> {
    /// The indices of the queries in the ordered they were sampled, before sorting and
    /// deduplication.
    pub unsorted_query_locations: Vec<usize>,
    /// For each trace, the Merkle decommitment auxiliary data.
    pub trace_decommitment: TreeVec<MerkleDecommitmentLiftedAux<H>>,
    /// The FRI auxiliary data.
    pub fri: FriProofAux<H>,
}

pub struct ExtendedCommitmentSchemeProof<H: MerkleHasherLifted> {
    pub proof: CommitmentSchemeProof<H>,
    pub aux: CommitmentSchemeProofAux<H>,
}

/// A batch of column samplings at a point.
pub struct ColumnSampleBatch {
    /// The point at which the columns are sampled.
    pub point: CirclePoint<SecureField>,
    /// The sampled column indices, their values at the point, and the random coefficient power
    /// corresponding to the fri quotient associated with (point, column_index, value).
    pub cols_vals_randpows: Vec<NumeratorData>,
}

/// Helper struct used only in `ColumnSampleBatch`. For a sample point z in a `ColumnSampleBatch`,
/// this struct contains the index of a column with such sample, the value of the column poly at z,
/// and the random coefficient power corresponding to the fri quotient associated with (z,
/// column_index, value).
pub struct NumeratorData {
    pub column_index: usize,
    pub sample_value: SecureField,
    pub random_coeff: SecureField,
}

impl ColumnSampleBatch {
    /// Groups column samples by sampled point.
    /// # Arguments
    /// samples: For each column, a vector of samples.
    pub fn new_vec(samples_with_rand: &[&Vec<(PointSample, SecureField)>]) -> Vec<Self> {
        // Group samples by point, and create a ColumnSampleBatch for each point.
        // This should keep a stable ordering.
        let mut grouped_samples = IndexMap::default();
        for (column_index, samples) in samples_with_rand.iter().enumerate() {
            for (sample, rand_pow) in samples.iter() {
                grouped_samples
                    .entry(sample.point)
                    .or_insert_with(Vec::new)
                    .push(NumeratorData {
                        column_index,
                        sample_value: sample.value,
                        random_coeff: *rand_pow,
                    });
            }
        }
        grouped_samples
            .into_iter()
            .map(|(point, cols_vals_randpows)| ColumnSampleBatch {
                point,
                cols_vals_randpows,
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct PointSample {
    pub point: CirclePoint<SecureField>,
    pub value: SecureField,
}

/// For each query position, corresponding to a domain point `p`, compute the FRI quotients
/// ```plain
///     ∑ ∑ α^{k(i, z)} * (c(i, z) * f̃ᵢ(p) - b(i, z) - a(i, z)) / line(z,conj(z))(p)
/// ```
/// where (with `α^{k(i, z)}` the `k(i, z)`-th of `random_coeff`'s coefficients, see
/// [`DeepCoefficients`]):
/// * the outer sum is over the set of sample points `z`,
/// * the inner sum is over the set of columns (corresponding to index `i`),
/// * line(z,conj(z))(p) is the equation of the line through (z, conj(z)) evaluated at `p`
/// * f̃ᵢ is the lift of the trace poly fᵢ to the domain of maximal log size.
/// * (a(i, z), b(i, z), c(i, z)) are the coefficients of the line equation `cY - aX - b` through
///   (z.y, f̃ᵢ(z)), (conj(z.y), conj(f̃ᵢ(z)).
pub fn fri_answers(
    column_log_sizes: TreeVec<Vec<u32>>,
    samples: TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: impl Into<DeepCoefficients>,
    query_positions: &[usize],
    queried_values: TreeVec<ColumnVec<Vec<BaseField>>>,
    lifting_log_size: u32,
) -> Result<Vec<SecureField>, VerificationError> {
    let queried_values = queried_values.flatten();
    assert!(queried_values
        .iter()
        .all(|queries_per_col| queries_per_col.len() == query_positions.len()));
    let samples_with_randomness = build_samples_with_randomness_and_periodicity(
        &samples,
        column_log_sizes
            .0
            .into_iter()
            .map(|x| x.into_iter())
            .collect(),
        lifting_log_size,
        random_coeff,
    );
    let sample_batches =
        ColumnSampleBatch::new_vec(&samples_with_randomness.iter().flatten().collect::<Vec<_>>());
    let lifting_domain = CanonicCoset::new(lifting_log_size).circle_domain();
    // Compute the quotient constants for all batches.
    let quotient_constants = quotient_constants(&sample_batches)?;
    let mut res = Vec::with_capacity(query_positions.len());
    for (idx, position) in query_positions.iter().enumerate() {
        let queried_values_at_row = queried_values.iter().map(|col| col[idx]).collect_vec();
        let domain_point = lifting_domain.at(bit_reverse_index(*position, lifting_log_size));

        res.push(accumulate_row_quotients(
            &sample_batches,
            &queried_values_at_row,
            &quotient_constants,
            domain_point,
        )?);
    }
    Ok(res)
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    queried_values_at_row: &[BaseField],
    quotient_constants: &QuotientConstants,
    domain_point: CirclePoint<BaseField>,
) -> Result<SecureField, VerificationError> {
    let sample_points = sample_batches.iter().map(|b| b.point).collect_vec();
    let denominator_inverses = denominator_inverses(&sample_points, domain_point)?;
    let mut row_accumulator = SecureField::zero();
    for (sample_batch, line_coeffs, denominator_inverse) in izip!(
        sample_batches,
        &quotient_constants.line_coeffs,
        denominator_inverses
    ) {
        let mut numerator = SecureField::zero();
        for (NumeratorData { column_index, .. }, (a, b, c)) in
            zip_eq(&sample_batch.cols_vals_randpows, line_coeffs)
        {
            let value = queried_values_at_row[*column_index] * *c;
            // The numerator is a line equation passing through
            //   (sample_point.y, sample_value), (conj(sample_point), conj(sample_value))
            // evaluated at (domain_point.y, value).
            // When substituting a polynomial in this line equation, we get a polynomial with a root
            // at sample_point and conj(sample_point) if the original polynomial had the values
            // sample_value and conj(sample_value) at these points.
            let linear_term = *a * domain_point.y + *b;
            numerator += value - linear_term;
        }
        // Note that `denominator_inverse` is an element of CM31 (see the docs and comments in the
        // function `denominator_inverses`).
        row_accumulator += numerator.mul_cm31(denominator_inverse);
    }
    Ok(row_accumulator)
}

/// Computes the sum
///     ∑ α^{k_i} * (cᵢ * f̃ᵢ(p) - bᵢ)
/// where:
/// * i is an index into `queried_values_at_row` that runs over the columns involved in the batch.
/// * f̃ᵢ(p) is `queried_values_at_row[i]`.
pub fn accumulate_row_partial_numerators(
    batch: &ColumnSampleBatch,
    queried_values_at_row: &[BaseField],
    coeffs: &Vec<(SecureField, SecureField, SecureField)>,
) -> SecureField {
    let mut numerator = SecureField::zero();
    for (NumeratorData { column_index, .. }, (_, b, c)) in zip_eq(&batch.cols_vals_randpows, coeffs)
    {
        let value = queried_values_at_row[*column_index] * *c;
        numerator += value - *b;
    }
    numerator
}

/// A column's line coefficients `(a, b, c)` in a sample batch ([`column_line_coeffs`]).
pub type LineCoeffs = (SecureField, SecureField, SecureField);

/// Precomputes the complex conjugate line coefficients for each column in each sample batch.
///
/// For the `i`-th numerator term `curr_coeff_power * alpha^i * (c * F(p) - (a * p.y + b))`,
/// we precompute and return the constants: (`curr_coeff_power * alpha^i * a`, `curr_coeff_power *
/// alpha^i * b`, `curr_coeff_power * alpha^i * c`). The index `i` is zero-based and runs
/// monotonically across all sample batches (i.e. the index of the `m`-th column in the `n`-th batch
/// is `m + Σ len(batch_k)`, for `k < n`).
///
/// # Errors
///
/// A sample point whose `y` is its own conjugate ([`complex_conjugate_line_coeffs`]).
pub fn column_line_coeffs(
    sample_batches: &[ColumnSampleBatch],
) -> Result<Vec<Vec<LineCoeffs>>, VerificationError> {
    sample_batches
        .iter()
        .map(|sample_batch| {
            sample_batch
                .cols_vals_randpows
                .iter()
                .map(
                    |NumeratorData {
                         column_index: _,
                         sample_value,
                         random_coeff,
                     }| {
                        let sample = PointSample {
                            point: sample_batch.point,
                            value: *sample_value,
                        };
                        complex_conjugate_line_coeffs(&sample, *random_coeff)
                    },
                )
                .collect()
        })
        .collect()
}

/// For each sample point P, computes the equation of a line passing through P = (pₓ, pᵧ) ∈
/// QM31 x QM31 and its conjugate P̄ = (p̄ₓ, p̄ᵧ), where the conjugate of an element of QM31 is with
/// respect to CM31. Then evaluates the line equation at `domain_point`, and inverts it.
///
/// # Errors
///
/// [`VerificationError::InvalidStructure`] when the line passes through `domain_point`: its
/// value there is zero and has no inverse. For P on the circle with `Im(pᵧ) ≠ 0` this never
/// happens at a point of the base field's circle. Write P = R + u·I, R and I in CM31²: the value
/// vanishes at Q iff R − Q = s·I for some s in CM31, so |Q|² = |R|² + s²·|I|² (as R·I = 0 on the
/// circle) = 1 + (s² − (2 + i))·|I|², and Q is on the circle only if |I|² = 0 (2 + i is not a
/// square in CM31). An isotropic I ≠ 0 is (±i·a, a), and R·I = 0 then makes |R|² = 0, against
/// |R|² + (2 + i)·|I|² = 1. Points off the circle can reach it.
pub fn denominator_inverses(
    sample_points: &[CirclePoint<SecureField>],
    domain_point: CirclePoint<M31>,
) -> Result<Vec<CM31>, VerificationError> {
    let mut denominators = Vec::new();

    // To find the equation of the line through P and P̄: a point Q = (qₓ, qᵧ) is on the line iff P -
    // Q is parallel to P - P̄ = (pₓ - p̄ₓ, pᵧ - p̄ᵧ), which is a multiple of (Im(pₓ), Im(pᵧ)).
    // We have P - Q  = ((Re(pₓ) - qₓ) + u * Im(pₓ), (Re(pᵧ) - qᵧ) + u * Im(pᵧ)). The parallelism
    // check reduces to
    //      (Re(pₓ) - qₓ) * Im(pᵧ) - (Re(pᵧ) - qᵧ) * Im(pₓ) = 0.
    // Note that this expression, evaluated at an arbitrary Q with M31 coordinates, is an element of
    // CM31.
    for sample_point in sample_points {
        // Extract Re(pₓ), Re(pᵧ), Im(pₓ), Im(pᵧ).
        let prx = sample_point.x.0;
        let pry = sample_point.y.0;
        let pix = sample_point.x.1;
        let piy = sample_point.y.1;
        denominators.push((prx - domain_point.x) * piy - (pry - domain_point.y) * pix);
    }
    if denominators.iter().any(CM31::is_zero) {
        return Err(VerificationError::InvalidStructure(ToString::to_string(
            "a sample point's conjugate line passes through a queried point",
        )));
    }

    Ok(CM31::batch_inverse(&denominators))
}

/// The line coefficients of every sample batch ([`column_line_coeffs`]).
///
/// # Errors
///
/// A sample point whose `y` is its own conjugate ([`complex_conjugate_line_coeffs`]).
pub fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
) -> Result<QuotientConstants, VerificationError> {
    Ok(QuotientConstants {
        line_coeffs: column_line_coeffs(sample_batches)?,
    })
}

/// Holds the precomputed constant values used in each quotient evaluation.
pub struct QuotientConstants {
    /// The line coefficients for each quotient numerator term. For more details see
    /// [self::column_line_coeffs].
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
}

pub fn build_samples_with_randomness_and_periodicity(
    samples: &TreeVec<Vec<Vec<PointSample>>>,
    column_log_sizes: Vec<impl Iterator<Item = u32>>,
    lifting_log_size: u32,
    random_coeff: impl Into<DeepCoefficients>,
) -> TreeVec<Vec<Vec<(PointSample, SecureField)>>> {
    let coefficients = random_coeff.into();
    let mut random_pows = coefficients.iter();
    let mut res: Vec<Vec<Vec<(PointSample, SecureField)>>> = Vec::new();
    let lifting_domain_generator = CanonicCoset::new(lifting_log_size).step();
    for (samples_per_tree, sizes_per_tree) in samples.iter().zip(column_log_sizes.into_iter()) {
        let samples_with_randomness_and_periodicity = samples_per_tree
            .iter()
            .zip(sizes_per_tree)
            .map(|(samples_per_cols, log_size)| {
                if samples_per_cols.is_empty() {
                    return Vec::new();
                }
                let mut new_samples: Vec<(PointSample, SecureField)> = Vec::new();
                // If the column has two samples, we add a periodicity check. Note that this check
                // is added even when the column is at its maximal size, in which case the
                // periodicity sample coincides with the OOD point sample. This simplifies the
                // Verifier logic by avoiding the need to track column sizes.
                // `deep_values_of_column` counts the values this pushes.
                if let [_prev_point_sample, point_sample] = &samples_per_cols[..] {
                    let period_generator = lifting_domain_generator.repeated_double(log_size);
                    new_samples.push((
                        PointSample {
                            point: point_sample.point.add(period_generator.into_ef()),
                            value: point_sample.value,
                        },
                        random_pows.next().expect(ONE_COEFFICIENT_A_VALUE),
                    ));
                }
                for sample in samples_per_cols.iter() {
                    new_samples.push((
                        sample.clone(),
                        random_pows.next().expect(ONE_COEFFICIENT_A_VALUE),
                    ));
                }
                new_samples
            })
            .collect();
        res.push(samples_with_randomness_and_periodicity);
    }
    TreeVec(res)
}

const ONE_COEFFICIENT_A_VALUE: &str = "the coefficients are drawn one a sampled value";

/// The values the DEEP quotient batches for a column sampled at `n_samples` points: each sample,
/// and a periodicity sample when there are two.
pub const fn deep_values_of_column(n_samples: usize) -> usize {
    if n_samples == 2 {
        3
    } else {
        n_samples
    }
}

/// Every value the DEEP quotient batches, `B`: one coefficient each.
pub fn n_deep_values(samples: &TreeVec<Vec<Vec<PointSample>>>) -> usize {
    samples
        .iter()
        .flatten()
        .map(|column| deep_values_of_column(column.len()))
        .sum()
}

/// The coefficients the DEEP quotient batches its sampled values by, in the order
/// [`build_samples_with_randomness_and_periodicity`] enumerates the values: the `i`-th value's
/// quotient `Q_i` enters the function FRI tests, `Q = Σ_i c_i · Q_i`, with coefficient `c_i`.
/// The coefficients are drawn from the channel after every commitment and every sampled value
/// has been mixed into it, so they are independent of the `Q_i`.
///
/// # Soundness
///
/// The batching step's error is the probability, over the coefficients, that `Q` is within the
/// proximity parameter of the code while some `Q_i` is not (a failure of correlated agreement).
///
/// - **[`DeepCoefficients::Powers`]**, `c_i = α^i`: `Q` ranges over a curve of degree `B − 1` in
///   the `Q_i`, and the error is `(B − 1) · ε`, `ε` a line's error (BCIKS20, Theorem 6.2, for
///   curves).
/// - **[`DeepCoefficients::Affine`]**, `c_0 = 1` and `c_1 … c_{B−1}` independent and uniform in
///   QM31: `Q` is uniform over the affine space `Q_0 + span{Q_1, …, Q_{B−1}}`, and correlated
///   agreement over an affine space holds with a line's error `ε` whatever its dimension (BCIKS20,
///   Theorem 1.6). The `(B − 1)` factor leaves the batching term: about `log2(B − 1)` bits, 9.8 for
///   916 values.
///
/// BCIKS20 is Ben-Sasson, Carmon, Ishai, Kopparty, Saraf, "Proximity Gaps for Reed–Solomon
/// Codes" (FOCS 2020, eprint 2020/654). In the Johnson regime `ε` is the line error of Theorem
/// 4.2 of BCHKS25 (eprint 2025/2055), `ε_lin(k)` at the tested dimension `k`, and soundcalc
/// (`github.com/ethereum/soundcalc`, revision `2af5e7df`) counts the two batchings exactly so:
/// `pcs/pcs.py`, `_get_batching_error`, takes `get_error_powers` (`(B − 1) · ε_lin`) for powers
/// and `get_error_linear` (`ε_lin`) for "affine batching: `c_0 = 1, c_i = r_i` for independent
/// random values `r_i`" (`pcs/fri.py`; the theorems are cited in `proxgaps/proxgaps_regime.py`).
///
/// Each `r_i` is a uniform QM31 element: four M31 coordinates drawn by rejection sampling. The
/// verifier draws `B − 1` of them, `⌈(B − 1) / 2⌉` channel hashes on the Blake2s and Blake2b
/// channels, against one for powers. FRI's own folds keep powers of their `α`: each round is
/// counted as `(f − 1) · ε_lin` either way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeepCoefficients {
    /// `1, α, α², …`.
    Powers(SecureField),
    /// `r_1, …, r_{B−1}`; the first value's coefficient is 1.
    Affine(Vec<SecureField>),
}

impl DeepCoefficients {
    /// Draws the coefficients for `n_values` sampled values: one element for powers, `n_values
    /// − 1` for affine batching.
    pub fn draw(channel: &mut impl Channel, batching: DeepBatching, n_values: usize) -> Self {
        match batching {
            DeepBatching::Powers => Self::Powers(channel.draw_secure_felt()),
            DeepBatching::Affine => {
                Self::Affine(channel.draw_secure_felts(n_values.saturating_sub(1)))
            }
        }
    }

    /// The coefficients in order, the first 1; unbounded for powers.
    pub fn iter(&self) -> impl Iterator<Item = SecureField> + '_ {
        match self {
            Self::Powers(alpha) => {
                Either::Left(iter::successors(Some(SecureField::one()), move |c| {
                    Some(*c * *alpha)
                }))
            }
            Self::Affine(coefficients) => {
                Either::Right(iter::once(SecureField::one()).chain(coefficients.iter().copied()))
            }
        }
    }
}

impl From<SecureField> for DeepCoefficients {
    fn from(alpha: SecureField) -> Self {
        Self::Powers(alpha)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};
    use std_shims::{vec, Vec};

    use super::{denominator_inverses, fri_answers, PointSample, TreeVec};
    use crate::core::circle::{CirclePoint, M31_CIRCLE_GEN};
    use crate::core::constraints::complex_conjugate_line_coeffs;
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::{SecureField, QM31};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse_index;
    use crate::core::verifier::VerificationError;

    const LOG_SIZE: u32 = 5;
    const POSITION: usize = 3;

    /// The lifting domain's point at `position`, as `fri_answers` reads it.
    fn queried_point(position: usize) -> CirclePoint<M31> {
        CanonicCoset::new(LOG_SIZE)
            .circle_domain()
            .at(bit_reverse_index(position, LOG_SIZE))
    }

    /// The point `real + u·imaginary`, coordinate by coordinate.
    fn point(real: CirclePoint<M31>, imaginary: (CM31, CM31)) -> CirclePoint<SecureField> {
        CirclePoint {
            x: QM31(real.x.into(), imaginary.0),
            y: QM31(real.y.into(), imaginary.1),
        }
    }

    /// `fri_answers` over one column of log size [`LOG_SIZE`], sampled once at `sample`, and
    /// queried at [`POSITION`].
    fn answers(sample: CirclePoint<SecureField>) -> Result<Vec<SecureField>, VerificationError> {
        fri_answers(
            TreeVec(vec![vec![LOG_SIZE]]),
            TreeVec(vec![vec![vec![PointSample {
                point: sample,
                value: SecureField::one(),
            }]]]),
            SecureField::one(),
            &[POSITION],
            TreeVec(vec![vec![vec![M31::from(5)]]]),
            LOG_SIZE,
        )
    }

    fn refused(result: Result<Vec<SecureField>, VerificationError>) -> bool {
        matches!(result, Err(VerificationError::InvalidStructure(_)))
    }

    /// A sample point whose `y` is its own conjugate has no conjugate line: refused, where the
    /// line's coefficients asserted.
    #[test]
    fn a_sample_point_with_y_in_cm31_is_refused() {
        let in_cm31 = point(M31_CIRCLE_GEN, (CM31::from(M31::from(7)), CM31::zero()));
        let sample = PointSample {
            point: in_cm31,
            value: SecureField::one(),
        };
        assert!(matches!(
            complex_conjugate_line_coeffs(&sample, SecureField::one()),
            Err(VerificationError::InvalidStructure(_))
        ));
        assert!(refused(answers(in_cm31)));
        assert!(refused(answers(M31_CIRCLE_GEN.into_ef())));

        let off_cm31 = point(M31_CIRCLE_GEN, (CM31::zero(), CM31::from(M31::from(7))));
        assert!(complex_conjugate_line_coeffs(
            &PointSample {
                point: off_cm31,
                value: SecureField::one(),
            },
            SecureField::one()
        )
        .is_ok());
    }

    /// A sample point `q + u·I`, `q` the queried point: its conjugate line passes through `q`,
    /// so the DEEP quotient's denominator is zero there. Refused, where its inverse asserted; the
    /// same point queried elsewhere is answered.
    #[test]
    fn a_conjugate_line_through_the_queried_point_is_refused() {
        let queried = queried_point(POSITION);
        let imaginary = (
            CM31::from_u32_unchecked(3, 4),
            CM31::from_u32_unchecked(5, 6),
        );
        let through = point(queried, imaginary);
        assert!(matches!(
            denominator_inverses(&[through], queried),
            Err(VerificationError::InvalidStructure(_))
        ));
        assert!(refused(answers(through)));

        let elsewhere = point(queried_point(POSITION + 1), imaginary);
        assert!(denominator_inverses(&[elsewhere], queried).is_ok());
        assert!(answers(elsewhere).is_ok());
    }
}
