//! Times Blake2s Merkle leaf hashing with the default `compress16` against the portable `u32x16`
//! one, both on the SIMD backend's 16-lane layout.
//!
//! Usage: `blake2s_compress_bench [old|new|both|merkle] [log_n_rows] [n_cols] [reps]`
//!
//! * `old` hashes with [`compress16_portable`], `new` with [`compress16`] (the NEON implementation
//!   on aarch64, the portable one elsewhere); `both` runs the two, alternating, and checks that
//!   their digests agree.
//! * `merkle` times the backend's `commit_on_layer` for the Blake2s-M31 Merkle hasher, which uses
//!   [`compress16`].
//!
//! Single-threaded; pin it to one core (e.g. with `taskset`) for stable numbers.
#![feature(portable_simd)]

use std::array;
use std::hint::black_box;
use std::simd::u32x16;
use std::time::Instant;

use stwo::core::fields::m31::BaseField;
use stwo::core::vcs::blake2_merkle::Blake2sM31MerkleHasher;
use stwo::prover::backend::simd::blake2s::{compress16, compress16_portable, INITIAL_STATE};
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::vcs::ops::MerkleOps;

type Compress = fn([u32x16; 8], [u32x16; 16], u32x16, u32x16, u32x16, u32x16) -> [u32x16; 8];

/// Hashes the rows of the columns in `chunks` (each column a `Vec` of `u32x16` packs of 16 rows,
/// grouped 16 columns per chunk) the way the SIMD Merkle leaf layer does: one compression per
/// chunk, the last one finalized.
fn hash_leaves(chunks: &[&[Vec<u32x16>]], compress: Compress) -> Vec<[u32x16; 8]> {
    let n_packs = chunks[0][0].len();
    let zero = u32x16::splat(0);
    (0..n_packs)
        .map(|i| {
            let mut state = INITIAL_STATE;
            let mut t = 0u32;
            for (k, chunk) in chunks.iter().enumerate() {
                t += 4 * chunk.len() as u32;
                let msgs: [u32x16; 16] = array::from_fn(|j| chunk.get(j).map_or(zero, |c| c[i]));
                let last = if k + 1 == chunks.len() {
                    u32x16::splat(u32::MAX)
                } else {
                    zero
                };
                state = compress(state, msgs, u32x16::splat(t), zero, last, zero);
            }
            state
        })
        .collect()
}

fn digest(states: &[[u32x16; 8]]) -> u64 {
    states
        .iter()
        .flatten()
        .flat_map(|x| x.to_array())
        .fold(0xcbf29ce484222325, |h, w| {
            (h ^ w as u64).wrapping_mul(0x100000001b3)
        })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map_or("both", String::as_str);
    let log_n_rows: u32 = args.get(2).map_or(16, |s| s.parse().unwrap());
    let n_cols: usize = args.get(3).map_or(32, |s| s.parse().unwrap());
    let reps: usize = args.get(4).map_or(5, |s| s.parse().unwrap());
    assert!(log_n_rows >= 4 && n_cols >= 1);

    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as u32) & 0x7fff_ffff
    };
    let n_packs = 1usize << (log_n_rows - 4);
    let cols: Vec<Vec<u32x16>> = (0..n_cols)
        .map(|_| {
            (0..n_packs)
                .map(|_| u32x16::from_array(array::from_fn(|_| next())))
                .collect()
        })
        .collect();
    let n_compressions = n_packs * n_cols.div_ceil(16);
    println!(
        "arch={} rows=2^{log_n_rows} cols={n_cols} compress16 calls/rep={n_compressions}",
        std::env::consts::ARCH
    );

    let report = |name: &str, secs: f64| {
        println!(
            "{name:>6}: {:9.3} ms/rep  {:8.1} ns/compress16",
            secs * 1e3,
            secs * 1e9 / n_compressions as f64
        );
    };
    let chunks: Vec<&[Vec<u32x16>]> = cols.chunks(16).collect();
    let time = |compress: Compress| {
        let start = Instant::now();
        let res = black_box(hash_leaves(black_box(&chunks), compress));
        (start.elapsed().as_secs_f64(), digest(&res))
    };

    match mode {
        "old" | "new" => {
            let compress: Compress = if mode == "old" {
                compress16_portable
            } else {
                compress16
            };
            let best = (0..reps)
                .map(|_| time(compress).0)
                .fold(f64::INFINITY, f64::min);
            report(mode, best);
        }
        "both" => {
            let (mut best_old, mut best_new) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..reps {
                let (t_old, d_old) = time(compress16_portable);
                let (t_new, d_new) = time(compress16);
                assert_eq!(d_old, d_new, "old and new digests differ");
                best_old = best_old.min(t_old);
                best_new = best_new.min(t_new);
            }
            report("old", best_old);
            report("new", best_new);
            println!("speedup: {:.3}x (digests agree)", best_old / best_new);
        }
        "merkle" => {
            let columns: Vec<BaseColumn> = cols
                .iter()
                .map(|c| {
                    c.iter()
                        .flat_map(|x| x.to_array())
                        .map(BaseField::from)
                        .collect()
                })
                .collect();
            let refs: Vec<_> = columns.iter().collect();
            let best = (0..reps)
                .map(|_| {
                    let start = Instant::now();
                    black_box(
                        <SimdBackend as MerkleOps<Blake2sM31MerkleHasher>>::commit_on_layer(
                            log_n_rows, None, &refs,
                        ),
                    );
                    start.elapsed().as_secs_f64()
                })
                .fold(f64::INFINITY, f64::min);
            report("merkle", best);
        }
        _ => panic!("mode must be old, new, both or merkle"),
    }
}
