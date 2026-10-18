//! Blake2b-256: BLAKE2b (RFC 7693) with a 32-byte digest, no key and no personalization.
//!
//! Every hash here is that function of one message, so a host that serves only whole-message
//! Blake2b-256 (Substrate's `blake2_256`) or only the BLAKE2b compression function computes the
//! same digests.

use core::fmt;

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use serde::{Deserialize, Serialize};
use std_shims::Vec;

/// A Blake2b-256 digest.
#[derive(Clone, Copy, PartialEq, Default, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct Blake2bHash(pub [u8; 32]);

impl From<Blake2bHash> for Vec<u8> {
    fn from(value: Blake2bHash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Blake2bHash> for [u8; 32] {
    fn from(value: Blake2bHash) -> Self {
        value.0
    }
}

impl AsRef<[u8]> for Blake2bHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Blake2bHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for Blake2bHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Blake2bHash as fmt::Display>::fmt(self, f)
    }
}

impl super::hash::Hash for Blake2bHash {}

/// A streaming Blake2b-256 hasher.
#[derive(Clone, Debug, Default)]
pub struct Blake2bHasher {
    state: Blake2b<U32>,
}

impl Blake2bHasher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, data: &[u8]) {
        Digest::update(&mut self.state, data);
    }

    pub fn finalize(self) -> Blake2bHash {
        Blake2bHash(self.state.finalize().into())
    }

    /// The digest of the concatenation of `parts`.
    pub fn hash_parts(parts: &[&[u8]]) -> Blake2bHash {
        let mut hasher = Self::new();
        for part in parts {
            hasher.update(part);
        }
        hasher.finalize()
    }
}

/// The number of BLAKE2b compressions a message of `len` bytes costs: one per 128-byte block,
/// and one for the empty message.
pub const fn blake2b_compressions(len: usize) -> usize {
    if len == 0 {
        1
    } else {
        len.div_ceil(128)
    }
}

#[cfg(test)]
mod tests {
    use std_shims::ToString;

    use super::{blake2b_compressions, Blake2bHasher};

    /// RFC 7693's Blake2b with a 32-byte digest, as `b2sum -l 256` and Python's
    /// `hashlib.blake2b(digest_size=32)` compute it.
    #[test]
    fn blake2b_256_matches_the_standard_function() {
        assert_eq!(
            Blake2bHasher::hash_parts(&[]).to_string(),
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
        );
        assert_eq!(
            Blake2bHasher::hash_parts(&[b"a", b"bc"]).to_string(),
            "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319"
        );
    }

    #[test]
    fn compressions_count_whole_blocks() {
        assert_eq!(blake2b_compressions(0), 1);
        assert_eq!(blake2b_compressions(1), 1);
        assert_eq!(blake2b_compressions(128), 1);
        assert_eq!(blake2b_compressions(129), 2);
    }
}
