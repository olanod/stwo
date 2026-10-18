//! Lifted Merkle trees on Blake2b-256, and the Merkle channel that pairs them with
//! [`Blake2bChannel`].
//!
//! - **A leaf** is `Blake2b-256(0x00 ‖ v_0 ‖ v_1 ‖ …)`, each value a little-endian u32, in the
//!   order the lifted tree gives (the columns sorted by size): `⌈(1 + 4c) / 128⌉` compressions for
//!   `c` values.
//! - **A node** is `Blake2b-256(0x01 ‖ left ‖ right)`: 65 bytes, one compression.
//!
//! The leading byte keeps a leaf's message apart from a node's whatever the leaf's width (RFC
//! 6962's convention), and from every channel message (whose tags start at `0x10`).

use serde::{Deserialize, Serialize};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::channel::{Blake2bChannel, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2b_hash::{Blake2bHash, Blake2bHasher};

/// The leading byte of a leaf's message.
pub const LEAF_TAG: u8 = 0x00;
/// The leading byte of an inner node's message.
pub const NODE_TAG: u8 = 0x01;

/// A leaf hasher that starts with [`LEAF_TAG`] absorbed.
#[derive(Clone, Debug)]
pub struct Blake2bMerkleHasher {
    hasher: Blake2bHasher,
}

impl Default for Blake2bMerkleHasher {
    fn default() -> Self {
        let mut hasher = Blake2bHasher::new();
        hasher.update(&[LEAF_TAG]);
        Self { hasher }
    }
}

impl MerkleHasherLifted for Blake2bMerkleHasher {
    type Hash = Blake2bHash;

    fn hash_children((left, right): (Self::Hash, Self::Hash)) -> Self::Hash {
        Blake2bHasher::hash_parts(&[&[NODE_TAG], &left.0, &right.0])
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        for value in column_values {
            self.hasher.update(&value.0.to_le_bytes());
        }
    }

    fn finalize(self) -> Self::Hash {
        self.hasher.finalize()
    }
}

/// Blake2b-256 Merkle trees with the Blake2b-256 Fiat–Shamir channel.
#[derive(Default)]
pub struct Blake2bMerkleChannel;

impl MerkleChannel for Blake2bMerkleChannel {
    type C = Blake2bChannel;
    type H = Blake2bMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: Blake2bHash) {
        channel.mix_root(root);
    }
}

/// The hasher carries no data a proof needs; these impls let proof types that are generic over
/// the hasher derive `Serialize` and `Deserialize`.
impl Serialize for Blake2bMerkleHasher {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let state = serializer.serialize_struct("Blake2bMerkleHasher", 0)?;
        serde::ser::SerializeStruct::end(state)
    }
}

impl<'de> Deserialize<'de> for Blake2bMerkleHasher {
    fn deserialize<D: serde::Deserializer<'de>>(_deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::default())
    }
}

#[cfg(test)]
mod tests {
    use std_shims::ToString;

    use super::Blake2bMerkleHasher;
    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::blake2b_hash::Blake2bHash;
    use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;

    /// Leaves and nodes are Blake2b-256 of their tagged messages (vectors from Python's
    /// `hashlib.blake2b(digest_size=32)`), and a leaf hashed in pieces equals it hashed whole.
    #[test]
    fn hashes_match_reference_vectors() {
        let values = [1, 2, 3].map(BaseField::from);
        let mut whole = Blake2bMerkleHasher::default();
        whole.update_leaf(&values);
        let mut pieces = Blake2bMerkleHasher::default();
        pieces.update_leaf(&values[..1]);
        pieces.update_leaf(&values[1..]);
        let leaf = whole.finalize();
        assert_eq!(leaf, pieces.finalize());
        assert_eq!(
            leaf.to_string(),
            "7cd3e3b286ba00e71ed24aecdc3b6dc57041a74db852161f21e74e2e6ed0abc2"
        );
        assert_eq!(
            Blake2bMerkleHasher::default().finalize().to_string(),
            "03170a2e7597b7b7e3d84c05391d139a62b157e78786d8c082f29dcf4c111314"
        );
        let node =
            Blake2bMerkleHasher::hash_children((Blake2bHash([0x11; 32]), Blake2bHash([0x22; 32])));
        assert_eq!(
            node.to_string(),
            "da49cb31db2893bfb633ce0c94548e2f0dbf3f7fd33a0d9fc6c2ef1b92f6db5e"
        );
    }

    /// A 16-value leaf has a node's length plus the tag; the tags keep the two apart.
    #[test]
    fn a_leaf_is_never_a_node() {
        let left = Blake2bHash([7; 32]);
        let right = Blake2bHash([9; 32]);
        let values = left
            .0
            .chunks_exact(4)
            .chain(right.0.chunks_exact(4))
            .map(|c| BaseField::from_u32_unchecked(u32::from_le_bytes(c.try_into().unwrap())))
            .collect::<std_shims::Vec<_>>();
        let mut leaf = Blake2bMerkleHasher::default();
        leaf.update_leaf(&values);
        assert_ne!(
            leaf.finalize(),
            Blake2bMerkleHasher::hash_children((left, right))
        );
    }
}
