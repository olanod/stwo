//! A Fiat–Shamir channel on Blake2b-256.
//!
//! The channel's state is one digest. Every operation hashes one message that starts with a
//! one-byte tag naming the operation, then the digest, then the operation's data:
//!
//! | operation | message | bytes | compressions |
//! |---|---|---|---|
//! | mix `n` u32s | `MIX_U32S ‖ digest ‖ words (LE)` | `33 + 4n` | `⌈(33 + 4n) / 128⌉`; 1 for a u64 |
//! | mix `n` QM31s | `MIX_FELTS ‖ digest ‖ M31 coordinates (LE u32)` | `33 + 16n` | `⌈(33 + 16n) / 128⌉`; 1 up to 5 felts |
//! | mix a Merkle root | `MIX_ROOT ‖ digest ‖ root` | 65 | 1 |
//! | draw 8 u32s | `DRAW ‖ digest ‖ counter (LE u32)` | 37 | 1 |
//! | PoW prefix | `POW_PREFIX ‖ digest ‖ n_bits (LE u32)` | 37 | 1, once a check |
//! | PoW attempt | `POW_NONCE ‖ prefix ‖ nonce (LE u64)` | 41 | 1 |
//!
//! A mix resets the draw counter. A draw yields 8 u32s, so 8 base field elements or 2 secure
//! field elements a compression (rejection sampling retries with probability about `2^-28`).
//! The tags keep every operation's messages apart from every other's and from the Merkle
//! hasher's ([`crate::core::vcs_lifted::blake2b_merkle`]), whose tags are 0 and 1.

use core::iter;

use std_shims::Vec;

use super::Channel;
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::vcs::blake2b_hash::{Blake2bHash, Blake2bHasher};

/// The leading byte of each operation's message.
pub mod tag {
    pub const MIX_U32S: u8 = 0x10;
    pub const MIX_FELTS: u8 = 0x11;
    pub const MIX_ROOT: u8 = 0x12;
    pub const DRAW: u8 = 0x13;
    pub const POW_PREFIX: u8 = 0x14;
    pub const POW_NONCE: u8 = 0x15;
}

const U32S_PER_DRAW: usize = 8;

/// A Fiat–Shamir channel whose every hash is one Blake2b-256 call.
#[derive(Default, Clone, Debug)]
pub struct Blake2bChannel {
    digest: Blake2bHash,
    n_draws: u32,
}

impl Blake2bChannel {
    pub const fn digest(&self) -> Blake2bHash {
        self.digest
    }

    pub const fn update_digest(&mut self, new_digest: Blake2bHash) {
        self.digest = new_digest;
        self.n_draws = 0;
    }

    /// Mixes a Merkle root: `MIX_ROOT ‖ digest ‖ root`.
    pub fn mix_root(&mut self, root: Blake2bHash) {
        self.update_digest(Blake2bHasher::hash_parts(&[
            &[tag::MIX_ROOT],
            &self.digest.0,
            &root.0,
        ]));
    }

    /// The digest a PoW nonce is hashed with: `POW_PREFIX ‖ digest ‖ n_bits`.
    pub fn pow_prefix(&self, n_bits: u32) -> Blake2bHash {
        Blake2bHasher::hash_parts(&[&[tag::POW_PREFIX], &self.digest.0, &n_bits.to_le_bytes()])
    }

    /// Whether `POW_NONCE ‖ prefix ‖ nonce` has at least `n_bits` trailing zero bits in its first
    /// 16 bytes, read as a little-endian u128.
    pub fn pow_nonce_passes(prefix: &Blake2bHash, n_bits: u32, nonce: u64) -> bool {
        let hash = Blake2bHasher::hash_parts(&[&[tag::POW_NONCE], &prefix.0, &nonce.to_le_bytes()]);
        let low = u128::from_le_bytes(hash.0[..16].try_into().unwrap());
        low.trailing_zeros() >= n_bits
    }

    /// Draws 8 uniform base field elements, retrying while any of the 8 u32s is `2P` or more.
    fn draw_base_felts(&mut self) -> [BaseField; U32S_PER_DRAW] {
        loop {
            let words = self.draw_words();
            if words.iter().all(|word| *word < 2 * P) {
                return words.map(|word| BaseField::reduce(word as u64));
            }
        }
    }

    fn draw_words(&mut self) -> [u32; U32S_PER_DRAW] {
        let hash =
            Blake2bHasher::hash_parts(&[&[tag::DRAW], &self.digest.0, &self.n_draws.to_le_bytes()]);
        self.n_draws += 1;
        core::array::from_fn(|i| u32::from_le_bytes(hash.0[4 * i..4 * i + 4].try_into().unwrap()))
    }
}

impl Channel for Blake2bChannel {
    const BYTES_PER_HASH: usize = 32;

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let mut hasher = Blake2bHasher::new();
        hasher.update(&[tag::MIX_FELTS]);
        hasher.update(&self.digest.0);
        for felt in felts {
            for coordinate in felt.to_m31_array() {
                hasher.update(&coordinate.0.to_le_bytes());
            }
        }
        self.update_digest(hasher.finalize());
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        let mut hasher = Blake2bHasher::new();
        hasher.update(&[tag::MIX_U32S]);
        hasher.update(&self.digest.0);
        for word in data {
            hasher.update(&word.to_le_bytes());
        }
        self.update_digest(hasher.finalize());
    }

    fn mix_u64(&mut self, value: u64) {
        self.mix_u32s(&[value as u32, (value >> 32) as u32]);
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let felts = self.draw_base_felts();
        SecureField::from_m31_array(felts[..SECURE_EXTENSION_DEGREE].try_into().unwrap())
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        let mut felts = iter::from_fn(|| Some(self.draw_base_felts())).flatten();
        iter::from_fn(|| {
            Some(SecureField::from_m31_array([
                felts.next()?,
                felts.next()?,
                felts.next()?,
                felts.next()?,
            ]))
        })
        .take(n_felts)
        .collect()
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        self.draw_words().to_vec()
    }

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        Self::pow_nonce_passes(&self.pow_prefix(n_bits), n_bits, nonce)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use std_shims::{BTreeSet, Vec};

    use super::Blake2bChannel;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::core::vcs::blake2b_hash::Blake2bHash;
    use crate::m31;

    fn hex(hash: Blake2bHash) -> std_shims::String {
        std_shims::ToString::to_string(&hash)
    }

    #[test]
    fn draws_count_and_reset_on_mix() {
        let mut channel = Blake2bChannel::default();
        channel.draw_u32s();
        assert_eq!(channel.n_draws, 1);
        // 9 secure felts are 36 base felts: 5 draws of 8.
        channel.draw_secure_felts(9);
        assert_eq!(channel.n_draws, 6);
        channel.mix_u64(7);
        assert_eq!(channel.n_draws, 0);
    }

    #[test]
    fn successive_draws_differ() {
        let mut channel = Blake2bChannel::default();
        assert_ne!(channel.draw_u32s(), channel.draw_u32s());
        assert_ne!(channel.draw_secure_felt(), channel.draw_secure_felt());
        let mut felts = channel.draw_secure_felts(5);
        felts.extend(channel.draw_secure_felts(4));
        assert_eq!(felts.len(), felts.iter().collect::<BTreeSet<_>>().len());
    }

    #[test]
    fn mixing_changes_the_digest() {
        let mut channel = Blake2bChannel::default();
        let initial = channel.digest;
        channel.mix_felts(
            &(0..2)
                .map(|i| SecureField::from(m31!(i + 1923782)))
                .collect_vec(),
        );
        assert_ne!(initial, channel.digest);
    }

    #[test]
    fn mix_u64_is_two_little_endian_words() {
        let mut channel = Blake2bChannel::default();
        channel.mix_u64(0x1111222233334444);
        let mut words = Blake2bChannel::default();
        words.mix_u32s(&[0x33334444, 0x11112222]);
        assert_eq!(channel.digest, words.digest);
    }

    /// Mixing u32s and mixing felts of the same bytes, and mixing a root of the same bytes, give
    /// three different digests.
    #[test]
    fn operations_are_domain_separated() {
        let felt = SecureField::from_m31_array([m31!(1), m31!(2), m31!(3), m31!(4)]);
        let mut by_felts = Blake2bChannel::default();
        by_felts.mix_felts(&[felt, felt]);
        let mut by_words = Blake2bChannel::default();
        by_words.mix_u32s(&[1, 2, 3, 4, 1, 2, 3, 4]);
        let mut by_root = Blake2bChannel::default();
        let root: [u8; 32] = core::array::from_fn(|i| [1, 0, 0, 0][i % 4] + (i / 4 % 4) as u8);
        by_root.mix_root(Blake2bHash(root));
        let digests = [by_felts.digest, by_words.digest, by_root.digest];
        assert_eq!(digests.iter().collect::<BTreeSet<_>>().len(), 3);
    }

    /// The channel's hashes are Blake2b-256 of the documented messages: vectors from Python's
    /// `hashlib.blake2b(digest_size=32)`, so a verifier that hashes through a host's
    /// `blake2_256` or a BLAKE2b compression precompile draws the same.
    #[test]
    fn transcript_matches_reference_vectors() {
        let mut channel = Blake2bChannel::default();
        channel.mix_u32s(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            hex(channel.digest),
            "cd8a2484971be8536899c7250ce5f38fcb49b958b9b28cc87b349ce692cb03ac"
        );
        let draw = Blake2bHash(
            hex::decode("a6e2b85d6e6d3dad8af030792e6f0654c177be7bf96d3e8206750c94e1a8eae9")
                .unwrap()
                .try_into()
                .unwrap(),
        );
        let words = draw
            .0
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(channel.draw_u32s(), words);
        channel.mix_root(Blake2bHash([0xab; 32]));
        assert_eq!(
            hex(channel.digest),
            "c69684339b59fb2949b4190e7b660866a3c3b15062d50f6738faea52ebeb2b4a"
        );
        channel.mix_felts(&[SecureField::from_m31_array([
            m31!(5),
            m31!(6),
            m31!(7),
            m31!(8),
        ])]);
        assert_eq!(
            hex(channel.digest),
            "c0ade6381e7b384d9dbc4ef1a65267f91ea98e1a8bfc7ad74517748a81b628d5"
        );
        assert_eq!(
            hex(channel.pow_prefix(10)),
            "b3aa93330fbeee34d02a33dbf70d86147540b7aa30d6ee4dc7a5cbd1f1311c8e"
        );
    }

    #[test]
    fn pow_nonce_check() {
        let mut channel = Blake2bChannel::default();
        channel.mix_u64(0x1111222233334344);
        let n_bits = 8;
        let nonce = (0..)
            .find(|&n| channel.verify_pow_nonce(n_bits, n))
            .unwrap();
        assert!((0..nonce).all(|n| !channel.verify_pow_nonce(n_bits, n)));
        assert!(channel.verify_pow_nonce(0, 0));
        // The check is the documented two hashes.
        let prefix = channel.pow_prefix(n_bits);
        assert!(Blake2bChannel::pow_nonce_passes(&prefix, n_bits, nonce));
        // The prefix binds the bit count.
        assert_ne!(prefix, channel.pow_prefix(n_bits + 1));
    }
}
