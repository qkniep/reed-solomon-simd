//! A collection of utility functions and helpers to facilitate the implementation of the [`Engine`] trait.
//!
//! [`Engine`]: crate::engine::Engine

use crate::engine::{fwht, tables, Engine, GfElement, ShardsRefMut, GF_BITS, GF_ORDER};
use core::iter::zip;

// ======================================================================
// FUNCTIONS - PUBLIC

/// Evaluate Polynomial using Fast Walsh-Hadamard Transform (FWHT).
///
/// This function is designed to be inlined and be compiled with SIMD
/// features enabled within an Engine's implementation of `eval_poly`.
///
/// See [`Avx2`] for an example on how to do this.
///
/// [`Avx2`]: crate::engine::Avx2
#[inline(always)]
pub fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
    eval_poly_out_truncated(erasures, truncated_size, GF_ORDER);
}

/// Like [`eval_poly`], but only computes the first `output_count` outputs
/// (output truncation of the final transform), leaving `erasures[output_count..]`
/// unspecified.
///
/// `truncated_size` is the number of non-zero `erasures` at the front (input
/// truncation of the first transform); `output_count` is the number of leading
/// outputs the caller actually reads. Decoding only reads a prefix of the
/// result, so this avoids the otherwise-fixed full `GF_ORDER` final transform.
#[inline(always)]
pub fn eval_poly_out_truncated(
    erasures: &mut [GfElement; GF_ORDER],
    truncated_size: usize,
    output_count: usize,
) {
    let log_walsh = tables::get_log_walsh();

    fwht::fwht_in_truncated(erasures, truncated_size);

    // The pointwise multiply by `log_walsh` and the output-truncation fold are
    // both `O(GF_ORDER)` passes. Fuse them: accumulate each element's product
    // straight into the first `k` outputs, instead of a full multiply pass
    // followed by a separate fold pass.
    let k = output_count.next_power_of_two();

    if k >= GF_ORDER {
        // No output truncation: multiply in place, then run the full transform.
        for (e, factor) in zip(erasures.iter_mut(), log_walsh.iter()) {
            *e = mul_mod(*e, *factor);
        }
        fwht::fwht(erasures, GF_ORDER);
        return;
    }

    // out[i] = Σ_q (erasures[q*k + i] * log_walsh[q*k + i]), folded into
    // `erasures[0..k]`. The `q = 0` term seeds the accumulator in place; later
    // terms read only indices `>= k`, which are never written, so this is safe.
    for i in 0..k {
        erasures[i] = mul_mod(erasures[i], log_walsh[i]);
    }
    for base in (k..GF_ORDER).step_by(k) {
        for i in 0..k {
            let product = mul_mod(erasures[base + i], log_walsh[base + i]);
            erasures[i] = add_mod(erasures[i], product);
        }
    }

    // Full radix-2 WHT over the `k` folded elements.
    fwht::wht_pow2(erasures, k);
}

/// `x[] ^= y[]`
#[inline(always)]
pub fn xor(xs: &mut [[u8; 64]], ys: &[[u8; 64]]) {
    debug_assert_eq!(xs.len(), ys.len());

    for (x_chunk, y_chunk) in zip(xs.iter_mut(), ys.iter()) {
        for (x, y) in zip(x_chunk.iter_mut(), y_chunk.iter()) {
            *x ^= y;
        }
    }
}

/// `data[x .. x + count] ^= data[y .. y + count]`
///
/// Ranges must not overlap.
#[inline(always)]
pub fn xor_within(data: &mut ShardsRefMut, x: usize, y: usize, count: usize) {
    let (xs, ys) = data.flat2_mut(x, y, count);
    xor(xs, ys);
}

// ======================================================================
// FUNCTIONS - CRATE - Galois field operations

/// Some kind of addition.
#[inline(always)]
pub(crate) fn add_mod(x: GfElement, y: GfElement) -> GfElement {
    let sum = u32::from(x) + u32::from(y);
    (sum + (sum >> GF_BITS)) as GfElement
}

/// Some kind of subtraction.
#[inline(always)]
pub(crate) fn sub_mod(x: GfElement, y: GfElement) -> GfElement {
    let dif = u32::from(x).wrapping_sub(u32::from(y));
    dif.wrapping_add(dif >> GF_BITS) as GfElement
}

/// Some kind of multiplication (used by [`eval_poly_out_truncated`]'s pointwise
/// step, where one operand is a precomputed `log_walsh` factor).
#[inline(always)]
pub(crate) fn mul_mod(x: GfElement, y: GfElement) -> GfElement {
    let product = u32::from(x) * u32::from(y);
    add_mod(product as GfElement, (product >> GF_BITS) as GfElement)
}

// ======================================================================
// FUNCTIONS - CRATE

/// FFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn fft_skew_end(
    engine: &impl Engine,
    data: &mut ShardsRefMut,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    engine.fft(data, pos, size, truncated_size, pos + size);
}

/// IFFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn ifft_skew_end(
    engine: &impl Engine,
    data: &mut ShardsRefMut,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    engine.ifft(data, pos, size, truncated_size, pos + size);
}

// Formal derivative.
pub(crate) fn formal_derivative(data: &mut ShardsRefMut) {
    for i in 1..data.len() {
        let width: usize = 1 << i.trailing_zeros();
        xor_within(data, i - width, i, width);
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // Un-fused reference: input transform, full pointwise multiply, output
    // transform, with no truncation.
    fn eval_poly_reference(erasures: &mut [GfElement; GF_ORDER]) {
        let log_walsh = tables::get_log_walsh();
        fwht::fwht(erasures, GF_ORDER);
        for (e, factor) in zip(erasures.iter_mut(), log_walsh.iter()) {
            *e = mul_mod(*e, *factor);
        }
        fwht::fwht(erasures, GF_ORDER);
    }

    #[test]
    fn eval_poly_out_truncated_matches_reference() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let random: Vec<GfElement> = (0..GF_ORDER).map(|_| rng.random()).collect();

        for (truncated_size, output_count) in [
            (1, 1),
            (64, 64),
            (128, 128),
            (200, 200),
            (256, 64),      // output smaller than input
            (64, 256),      // output larger than input
            (GF_ORDER, 64), // dense input, truncated output (LowRate shape)
            (64, GF_ORDER), // truncated input, full output
            (GF_ORDER, GF_ORDER),
        ] {
            let mut got = [0; GF_ORDER];
            got[..truncated_size].copy_from_slice(&random[..truncated_size]);
            let mut want = got;

            eval_poly_out_truncated(&mut got, truncated_size, output_count);
            eval_poly_reference(&mut want);

            assert_eq!(
                got[..output_count],
                want[..output_count],
                "mismatch for (truncated_size, output_count) = ({truncated_size}, {output_count})"
            );
        }
    }
}
