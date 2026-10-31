//! BLAKE2s compression of 16 instances on aarch64 NEON.
//!
//! A `u32x16` occupies four 128-bit NEON registers, so the 16-word state of [`compress16`] alone
//! needs 64 registers when kept in `u32x16` form, and the portable implementation spills
//! heavily. Here the 16 instances are compressed as four independent groups of four lanes: the
//! state of one group is 16 `uint32x4_t` registers, which, with the rotation table and a few
//! temporaries, fits in NEON's 32 registers; message words that do not fit are reloaded from
//! memory. Lane `4 * g + k` of every input and output vector belongs to group `g`, lane `k`.
//!
//! [`compress16`]: super::blake2s::compress16

use core::arch::aarch64::{
    uint32x4_t, uint8x16_t, vaddq_u32, vdupq_n_u32, veorq_u32, vld1q_u32, vld1q_u8, vqtbl1q_u8,
    vreinterpretq_u16_u32, vreinterpretq_u32_u16, vreinterpretq_u32_u8, vreinterpretq_u8_u32,
    vrev32q_u16, vshlq_n_u32, vsriq_n_u32, vst1q_u32,
};
use std::array;
use std::simd::u32x16;

use super::blake2s::{IV, SIGMA};

/// Byte shuffle rotating every 32-bit word right by 8 bits.
const ROT8_TABLE: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];

#[inline(always)]
fn add(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    unsafe { vaddq_u32(a, b) }
}

#[inline(always)]
fn xor(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    unsafe { veorq_u32(a, b) }
}

#[inline(always)]
fn rot16(x: uint32x4_t) -> uint32x4_t {
    unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
}

#[inline(always)]
fn rot12(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32::<12>(vshlq_n_u32::<20>(x), x) }
}

#[inline(always)]
fn rot8(x: uint32x4_t, table: uint8x16_t) -> uint32x4_t {
    unsafe { vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), table)) }
}

#[inline(always)]
fn rot7(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32::<7>(vshlq_n_u32::<25>(x), x) }
}

/// Four BLAKE2s G functions in lockstep, on the words `a[i]`, `b[i]`, `c[i]`, `d[i]` with the
/// message words `x[i]`, `y[i]`, for `i` in `0..4`.
#[inline(always)]
fn g4(
    v: &mut [uint32x4_t; 16],
    m: &[uint32x4_t; 16],
    table: uint8x16_t,
    [a, b, c, d]: [[usize; 4]; 4],
    [x, y]: [[usize; 4]; 2],
) {
    for i in 0..4 {
        v[a[i]] = add(add(v[a[i]], m[x[i]]), v[b[i]]);
    }
    for i in 0..4 {
        v[d[i]] = rot16(xor(v[d[i]], v[a[i]]));
    }
    for i in 0..4 {
        v[c[i]] = add(v[c[i]], v[d[i]]);
    }
    for i in 0..4 {
        v[b[i]] = rot12(xor(v[b[i]], v[c[i]]));
    }
    for i in 0..4 {
        v[a[i]] = add(add(v[a[i]], m[y[i]]), v[b[i]]);
    }
    for i in 0..4 {
        v[d[i]] = rot8(xor(v[d[i]], v[a[i]]), table);
    }
    for i in 0..4 {
        v[c[i]] = add(v[c[i]], v[d[i]]);
    }
    for i in 0..4 {
        v[b[i]] = rot7(xor(v[b[i]], v[c[i]]));
    }
}

#[inline(always)]
fn round(v: &mut [uint32x4_t; 16], m: &[uint32x4_t; 16], table: uint8x16_t, r: usize) {
    let s = SIGMA[r].map(|i| i as usize);
    g4(
        v,
        m,
        table,
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        [[s[0], s[2], s[4], s[6]], [s[1], s[3], s[5], s[7]]],
    );
    g4(
        v,
        m,
        table,
        [[0, 1, 2, 3], [5, 6, 7, 4], [10, 11, 8, 9], [15, 12, 13, 14]],
        [[s[8], s[10], s[12], s[14]], [s[9], s[11], s[13], s[15]]],
    );
}

/// Lanes `4 * g..4 * g + 4` of `x`.
#[inline(always)]
fn group(x: &u32x16, g: usize) -> uint32x4_t {
    assert!(g < 4);
    // SAFETY: `x` holds 16 `u32`s and `4 * g + 4 <= 16`.
    unsafe { vld1q_u32(x.as_array().as_ptr().add(4 * g)) }
}

/// Compresses the four instances of lane group `g`, writing that group's lanes of `out`.
#[inline(never)]
fn compress4(
    h_vecs: &[u32x16; 8],
    msg_vecs: &[u32x16; 16],
    counters: &[u32x16; 4],
    g: usize,
    out: &mut [u32x16; 8],
) {
    let m: [uint32x4_t; 16] = array::from_fn(|i| group(&msg_vecs[i], g));
    let table = unsafe { vld1q_u8(ROT8_TABLE.as_ptr()) };
    let h = |i: usize| group(&h_vecs[i], g);
    let iv = |i: usize| unsafe { vdupq_n_u32(IV[i]) };
    let count = |i: usize| xor(iv(i + 4), group(&counters[i], g));
    let mut v = [
        h(0),
        h(1),
        h(2),
        h(3),
        h(4),
        h(5),
        h(6),
        h(7),
        iv(0),
        iv(1),
        iv(2),
        iv(3),
        count(0),
        count(1),
        count(2),
        count(3),
    ];
    // Unrolled so that the message indices are constants.
    round(&mut v, &m, table, 0);
    round(&mut v, &m, table, 1);
    round(&mut v, &m, table, 2);
    round(&mut v, &m, table, 3);
    round(&mut v, &m, table, 4);
    round(&mut v, &m, table, 5);
    round(&mut v, &m, table, 6);
    round(&mut v, &m, table, 7);
    round(&mut v, &m, table, 8);
    round(&mut v, &m, table, 9);
    for (i, o) in out.iter_mut().enumerate() {
        let res = xor(xor(h(i), v[i]), v[i + 8]);
        // SAFETY: `o` holds 16 `u32`s and `4 * g + 4 <= 16`.
        unsafe { vst1q_u32(o.as_mut_array().as_mut_ptr().add(4 * g), res) };
    }
}

/// Compresses 16 BLAKE2s instances; same contract as [`super::blake2s::compress16`].
pub fn compress16(
    h_vecs: [u32x16; 8],
    msg_vecs: [u32x16; 16],
    count_low: u32x16,
    count_high: u32x16,
    lastblock: u32x16,
    lastnode: u32x16,
) -> [u32x16; 8] {
    let counters = [count_low, count_high, lastblock, lastnode];
    let mut out = [u32x16::splat(0); 8];
    for g in 0..4 {
        compress4(&h_vecs, &msg_vecs, &counters, g, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::simd::u32x16;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::compress16;
    use crate::core::fields::m31::P;
    use crate::prover::backend::simd::blake2s::compress16_portable;
    use crate::prover::backend::simd::blake2s_ref::compress;
    use crate::prover::backend::simd::m31::reduce_to_m31_simd;

    /// Words at the edges of the u32 and M31 ranges.
    const EDGE_WORDS: [u32; 10] = [
        0,
        1,
        P - 1,
        P,
        P + 1,
        1 << 31,
        u32::MAX - 1,
        u32::MAX,
        0x5555_5555,
        0xAAAA_AAAA,
    ];

    struct Input {
        h: [u32x16; 8],
        m: [u32x16; 16],
        counters: [u32x16; 4],
    }

    /// Checks the NEON path against the portable path and, lane by lane, the scalar reference.
    fn check(input: &Input) {
        let Input { h, m, counters: c } = *input;
        let neon = compress16(h, m, c[0], c[1], c[2], c[3]);
        let portable = compress16_portable(h, m, c[0], c[1], c[2], c[3]);
        assert_eq!(neon, portable);
        for lane in 0..16 {
            let expected = compress(
                h.map(|x| x[lane]),
                m.map(|x| x[lane]),
                c[0][lane],
                c[1][lane],
                c[2][lane],
                c[3][lane],
            );
            assert_eq!(neon.map(|x| x[lane]), expected, "lane {lane}");
        }
        // The Blake2s-M31 digest reduction, composed with the compression.
        for (word, reduced) in neon.iter().zip(neon.map(reduce_to_m31_simd)) {
            assert_eq!(reduced.to_array(), word.to_array().map(|w| w % P));
        }
    }

    fn random_input(rng: &mut SmallRng) -> Input {
        Input {
            h: array::from_fn(|_| u32x16::from_array(rng.gen())),
            m: array::from_fn(|_| u32x16::from_array(rng.gen())),
            counters: array::from_fn(|_| u32x16::from_array(rng.gen())),
        }
    }

    #[test]
    fn neon_compress16_matches_reference_on_random_inputs() {
        let mut rng = SmallRng::seed_from_u64(0x6e656f6e);
        for _ in 0..500 {
            check(&random_input(&mut rng));
        }
    }

    #[test]
    fn neon_compress16_matches_reference_on_uniform_inputs() {
        for word in EDGE_WORDS {
            let w = u32x16::splat(word);
            check(&Input {
                h: [w; 8],
                m: [w; 16],
                counters: [w; 4],
            });
        }
    }

    #[test]
    fn neon_compress16_matches_reference_on_edge_words() {
        let mut rng = SmallRng::seed_from_u64(0xed6e);
        for _ in 0..500 {
            let mut edge = || {
                u32x16::from_array(array::from_fn(|_| {
                    EDGE_WORDS[rng.gen_range(0..EDGE_WORDS.len())]
                }))
            };
            check(&Input {
                h: array::from_fn(|_| edge()),
                m: array::from_fn(|_| edge()),
                counters: array::from_fn(|_| edge()),
            });
        }
    }

    /// Each lane computes on its own data: changing one word of one lane changes only that lane.
    #[test]
    fn neon_compress16_lanes_are_independent() {
        let mut rng = SmallRng::seed_from_u64(0x1a4e);
        let input = random_input(&mut rng);
        let base = compress16(
            input.h,
            input.m,
            input.counters[0],
            input.counters[1],
            input.counters[2],
            input.counters[3],
        );
        for lane in 0..16 {
            for word in 0..16 {
                let mut m = input.m;
                m[word][lane] ^= 1 << ((lane + word) % 32);
                let c = input.counters;
                let res = compress16(input.h, m, c[0], c[1], c[2], c[3]);
                for other in 0..16 {
                    let same = res.map(|x| x[other]) == base.map(|x| x[other]);
                    assert_eq!(
                        same,
                        other != lane,
                        "word {word}, lane {lane}, lane {other}"
                    );
                }
            }
        }
    }
}
