//! Blake2b-256 lifted Merkle trees on the SIMD backend: one scalar hash a leaf or node, the
//! leaves and nodes hashed in parallel.

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2b_hash::Blake2bHash;
use crate::core::vcs_lifted::blake2b_merkle::Blake2bMerkleHasher;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::parallel_iter;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl ColumnOps<Blake2bHash> for SimdBackend {
    type Column = Vec<Blake2bHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

/// The row of a column of `2^log_size` values that the leaf `index` of a tree lifted to
/// `2^lifting_log_size` leaves reads: the lifted column repeats each pair of adjacent values
/// `2^(lifting_log_size - log_size)` times (see the CPU backend's `build_leaves`).
const fn lifted_row(index: usize, log_size: u32, lifting_log_size: u32) -> usize {
    let log_ratio = lifting_log_size - log_size;
    ((index >> (log_ratio + 1)) << 1) + (index & 1)
}

impl MerkleOpsLifted<Blake2bMerkleHasher> for SimdBackend {
    /// The leaves the CPU backend's generic `build_leaves` computes: leaf `i` hashes every
    /// column's lifted value at `i`, in the columns' order (increasing in length).
    fn build_leaves(columns: &[&Col<Self, BaseField>], lifting_log_size: u32) -> Vec<Blake2bHash> {
        if columns.is_empty() {
            return vec![Blake2bMerkleHasher::default().finalize()];
        }
        let columns = columns
            .iter()
            .map(|column| {
                let values = column.as_slice();
                assert!(values.len() >= 2, "A column must be of length >= 2.");
                (values, values.len().ilog2())
            })
            .collect_vec();
        assert!(columns.windows(2).all(|w| w[0].1 <= w[1].1));
        assert!(columns.last().unwrap().1 <= lifting_log_size);
        parallel_iter!(0..1usize << lifting_log_size)
            .map(|index| {
                let mut hasher = Blake2bMerkleHasher::default();
                let row = columns
                    .iter()
                    .map(|(values, log_size)| {
                        values[lifted_row(index, *log_size, lifting_log_size)]
                    })
                    .collect_vec();
                hasher.update_leaf(&row);
                hasher.finalize()
            })
            .collect()
    }

    fn build_next_layer(prev_layer: &Vec<Blake2bHash>) -> Vec<Blake2bHash> {
        let half = prev_layer.len() / 2;
        parallel_iter!(0..half)
            .map(|i| Blake2bMerkleHasher::hash_children((prev_layer[2 * i], prev_layer[2 * i + 1])))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs_lifted::blake2b_merkle::Blake2bMerkleHasher;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::CpuBackend;
    use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

    /// The SIMD leaves and layers equal the CPU backend's generic ones, for columns of mixed
    /// sizes (some shorter than a SIMD vector) lifted to the largest and above it.
    #[test]
    fn simd_tree_matches_cpu() {
        let mut rng = SmallRng::seed_from_u64(5);
        let log_sizes = [1, 3, 3, 4, 6, 6, 6];
        let cpu_columns = log_sizes
            .iter()
            .map(|&log_size| {
                (0..1 << log_size)
                    .map(|_| BaseField::from(rng.gen_range(0..(1u32 << 30))))
                    .collect_vec()
            })
            .collect_vec();
        let simd_columns = cpu_columns
            .iter()
            .map(|column| BaseColumn::from_cpu(column))
            .collect_vec();
        for lifting_log_size in [6, 8] {
            let cpu = <CpuBackend as MerkleOpsLifted<Blake2bMerkleHasher>>::build_leaves(
                &cpu_columns.iter().collect_vec(),
                lifting_log_size,
            );
            let simd = <SimdBackend as MerkleOpsLifted<Blake2bMerkleHasher>>::build_leaves(
                &simd_columns.iter().collect_vec(),
                lifting_log_size,
            );
            assert_eq!(cpu, simd);
            assert_eq!(
                <CpuBackend as MerkleOpsLifted<Blake2bMerkleHasher>>::build_next_layer(&cpu),
                <SimdBackend as MerkleOpsLifted<Blake2bMerkleHasher>>::build_next_layer(&simd),
            );
        }
        assert_eq!(
            <CpuBackend as MerkleOpsLifted<Blake2bMerkleHasher>>::build_leaves(&[], 0),
            <SimdBackend as MerkleOpsLifted<Blake2bMerkleHasher>>::build_leaves(&[], 0),
        );
    }
}
