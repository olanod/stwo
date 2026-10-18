//! Implements a FRI polynomial commitment scheme.
//!
//! This is a protocol where the prover can commit on a set of polynomials and then prove their
//! opening on a set of points.
//! Note: This implementation is not really a polynomial commitment scheme, because we are not in
//! the unique decoding regime. This is enough for a STARK proof though, where we only want to imply
//! the existence of such polynomials, and are ok with having a small decoding list.
//! Note: Opened points cannot come from the commitment domain.

pub mod quotients;
pub mod utils;
mod verifier;

use serde::{Deserialize, Serialize};
use std_shims::vec;

pub use self::utils::TreeVec;
pub use self::verifier::CommitmentSchemeVerifier;
use super::channel::Channel;
use super::fields::qm31::SecureField;
use super::fri::FriConfig;

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub struct TreeSubspan {
    pub tree_index: usize,
    pub col_start: usize,
    pub col_end: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
/// Configuration parameters for the committment scheme prover.
pub struct PcsConfig {
    /// The number of proof of work bits before the FRI queries.
    pub pow_bits: u32,
    pub fri_config: FriConfig,
    /// An optional integer which controls the size of the lifting domain (This size includes the
    /// `log_blowup_factor`). When specified, the prover lifts all polynomials to the domain of
    /// given log size.
    /// If `None`, the prover lifts each tree’s polynomials to the largest domain within that tree
    /// (an implicit assumption here is that the largest domains are all of equal size across
    /// trees, except possibly for the preprocessed tree).
    pub lifting_log_size: Option<u32>,
    /// How the DEEP quotient batches the sampled values; [`DeepBatching::Powers`] unless a
    /// configuration opts in.
    ///
    /// Not serialized: a proof's bytes are the same under either batching, so a verifier sets it
    /// from its own configuration (the proof's format), never from a deserialized proof, whose
    /// field reads [`DeepBatching::Powers`]. A proof batched one way fails under the other.
    #[serde(skip)]
    pub deep_batching: DeepBatching,
}

/// How the DEEP quotient combines its sampled values into the one function FRI tests: the
/// coefficient of each value's quotient. See [`quotients::DeepCoefficients`] for the soundness
/// of each.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DeepBatching {
    /// `1, α, α², …`: powers of one coefficient drawn from the channel.
    #[default]
    Powers,
    /// `1, r_1, r_2, …`: an independent coefficient drawn from the channel for every value but
    /// the first.
    Affine,
}

impl PcsConfig {
    pub const fn security_bits(&self) -> u32 {
        self.pow_bits + self.fri_config.security_bits()
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        let PcsConfig {
            pow_bits,
            fri_config,
            lifting_log_size,
            deep_batching,
        } = self;
        let FriConfig {
            log_blowup_factor,
            n_queries,
            log_last_layer_degree_bound,
            fold_step,
        } = fri_config;

        let mut felts = vec![
            SecureField::from_u32_unchecked(
                *pow_bits,
                *log_blowup_factor,
                *n_queries as u32,
                *log_last_layer_degree_bound,
            ),
            SecureField::from_u32_unchecked(*fold_step, lifting_log_size.unwrap_or(0), 0, 0),
        ];
        // Affine batching adds a third element, so the default's transcript is unchanged.
        if *deep_batching == DeepBatching::Affine {
            felts.push(SecureField::from_u32_unchecked(1, 0, 0, 0));
        }
        channel.mix_felts(&felts);
    }
}

impl Default for PcsConfig {
    fn default() -> Self {
        Self {
            pow_bits: 10,
            fri_config: FriConfig::new(0, 1, 3, 1),
            lifting_log_size: None,
            deep_batching: DeepBatching::Powers,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_security_bits() {
        let config = super::PcsConfig {
            pow_bits: 42,
            fri_config: super::FriConfig::new(10, 10, 70, 1),
            ..Default::default()
        };
        assert!(config.security_bits() == 10 * 70 + 42);
    }
}
