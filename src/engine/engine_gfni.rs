// The GFNI affine intrinsic `_mm256_gf2p8affine_epi64_epi8` is stable only from
// Rust 1.89, newer than the crate's 1.82 MSRV. That is deliberate: this module
// compiles only under the opt-in `gfni` feature, which raises the effective MSRV
// to 1.89 (see the feature note in Cargo.toml).
#![allow(clippy::incompatible_msrv)]

use core::iter::zip;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::engine::{
    tables::{self, MulGfni, MultiplyGfniT, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};

// ======================================================================
// Gfni - PUBLIC

/// Optimized [`Engine`] using GFNI (`GF2P8AFFINEQB`) instructions.
///
/// [`Gfni`] follows the same algorithm as [`Avx2`] but replaces the
/// `PSHUFB`-based split-table multiply with the GFNI affine transform:
/// multiply-by-constant in `GF(2^16)` is `GF(2)`-linear, so it decomposes into
/// four 8×8 `GF(2)` affine maps (see [`MultiplyGfniT`]), each a single
/// `GF2P8AFFINEQB`. This needs GFNI in addition to AVX2 (Ice Lake / Tremont and
/// later on Intel; Zen 4 and later on AMD).
///
/// [`Avx2`]: crate::engine::Avx2
/// [`MultiplyGfniT`]: crate::engine::tables::MultiplyGfniT
#[derive(Clone, Copy)]
pub struct Gfni {
    mul_gfni: &'static MulGfni,
    skew: &'static Skew,
}

impl Gfni {
    /// Creates new [`Gfni`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [tables]: crate::engine::tables
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul_gfni = tables::get_mul_gfni();
        let skew = tables::get_skew();

        Self { mul_gfni, skew }
    }
}

impl Engine for Gfni {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_gfni(data, pos, size, truncated_size, 0, skew_delta);
        }
    }

    fn fft_out_window(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        output_start: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_gfni(data, pos, size, truncated_size, output_start, skew_delta);
        }
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.ifft_private_gfni(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        unsafe {
            self.mul_gfni(x, log_m);
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_gfni(erasures, truncated_size) }
    }
}

// ======================================================================
// Gfni - IMPL Default

impl Default for Gfni {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Gfni - PRIVATE
//
//

/// The four affine matrices for one multiplier, each broadcast to all four
/// 64-bit lanes of a `__m256i` so `GF2P8AFFINEQB` applies it to every byte.
#[derive(Copy, Clone)]
struct LutGfni {
    lo_lo: __m256i,
    hi_lo: __m256i,
    lo_hi: __m256i,
    hi_hi: __m256i,
}

impl From<&MultiplyGfniT> for LutGfni {
    #[inline(always)]
    fn from(m: &MultiplyGfniT) -> Self {
        unsafe {
            Self {
                lo_lo: broadcast_matrix(m.lo_lo),
                hi_lo: broadcast_matrix(m.hi_lo),
                lo_hi: broadcast_matrix(m.lo_hi),
                hi_hi: broadcast_matrix(m.hi_hi),
            }
        }
    }
}

/// Broadcasts a packed 8×8 affine matrix into every 64-bit lane.
#[inline(always)]
unsafe fn broadcast_matrix(m: u64) -> __m256i {
    // Reinterpret the bit pattern; the sign of the i64 is irrelevant here.
    unsafe { _mm256_set1_epi64x(i64::from_ne_bytes(m.to_ne_bytes())) }
}

/// `{data_lo, data_hi} := m * {data_lo, data_hi}` in `GF(2^16)`.
#[inline(always)]
unsafe fn affine(data: __m256i, matrix: __m256i) -> __m256i {
    unsafe { _mm256_gf2p8affine_epi64_epi8::<0>(data, matrix) }
}

impl Gfni {
    #[target_feature(enable = "gfni,avx2")]
    unsafe fn mul_gfni(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutGfni::from(&self.mul_gfni[log_m as usize]);

        for chunk in x.iter_mut() {
            let x_ptr = chunk.as_mut_ptr().cast::<__m256i>();
            unsafe {
                let x_lo = _mm256_loadu_si256(x_ptr);
                let x_hi = _mm256_loadu_si256(x_ptr.add(1));
                let (prod_lo, prod_hi) = Self::mul_256(x_lo, x_hi, lut);
                _mm256_storeu_si256(x_ptr, prod_lo);
                _mm256_storeu_si256(x_ptr.add(1), prod_hi);
            }
        }
    }

    // {prod_lo, prod_hi} = {value_lo, value_hi} * m via four affine transforms.
    #[inline(always)]
    fn mul_256(value_lo: __m256i, value_hi: __m256i, lut: LutGfni) -> (__m256i, __m256i) {
        unsafe {
            let prod_lo =
                _mm256_xor_si256(affine(value_lo, lut.lo_lo), affine(value_hi, lut.lo_hi));
            let prod_hi =
                _mm256_xor_si256(affine(value_lo, lut.hi_lo), affine(value_hi, lut.hi_hi));
            (prod_lo, prod_hi)
        }
    }

    // {x_lo, x_hi} ^= {y_lo, y_hi} * m
    #[inline(always)]
    fn muladd_256(
        mut x_lo: __m256i,
        mut x_hi: __m256i,
        y_lo: __m256i,
        y_hi: __m256i,
        lut: LutGfni,
    ) -> (__m256i, __m256i) {
        let (prod_lo, prod_hi) = Self::mul_256(y_lo, y_hi, lut);
        unsafe {
            x_lo = _mm256_xor_si256(x_lo, prod_lo);
            x_hi = _mm256_xor_si256(x_hi, prod_hi);
        }
        (x_lo, x_hi)
    }
}

// ======================================================================
// Gfni - PRIVATE - FFT (fast Fourier transform)

impl Gfni {
    #[inline(always)]
    fn fftb_256(x: &mut [u8; 64], y: &mut [u8; 64], lut: LutGfni) {
        let x_ptr = x.as_mut_ptr().cast::<__m256i>();
        let y_ptr = y.as_mut_ptr().cast::<__m256i>();

        unsafe {
            let mut x_lo = _mm256_loadu_si256(x_ptr);
            let mut x_hi = _mm256_loadu_si256(x_ptr.add(1));

            let mut y_lo = _mm256_loadu_si256(y_ptr);
            let mut y_hi = _mm256_loadu_si256(y_ptr.add(1));

            (x_lo, x_hi) = Self::muladd_256(x_lo, x_hi, y_lo, y_hi, lut);

            _mm256_storeu_si256(x_ptr, x_lo);
            _mm256_storeu_si256(x_ptr.add(1), x_hi);

            y_lo = _mm256_xor_si256(y_lo, x_lo);
            y_hi = _mm256_xor_si256(y_hi, x_hi);

            _mm256_storeu_si256(y_ptr, y_lo);
            _mm256_storeu_si256(y_ptr.add(1), y_hi);
        }
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutGfni::from(&self.mul_gfni[log_m as usize]);

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            Self::fftb_256(x_chunk, y_chunk, lut);
        }
    }

    #[inline(always)]
    fn fft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            self.fft_butterfly_partial(s0, s2, log_m02);
            self.fft_butterfly_partial(s1, s3, log_m02);
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            self.fft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            self.fft_butterfly_partial(s2, s3, log_m23);
        }
    }

    #[target_feature(enable = "gfni,avx2")]
    unsafe fn fft_private_gfni(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        output_start: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.fft_private(data, pos, size, truncated_size, output_start, skew_delta);
    }

    #[inline(always)]
    fn fft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        output_start: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            // Blocks of `dist4` entirely below `output_start` only affect
            // outputs there, so skip them.
            let mut r = output_start & !(dist4 - 1);
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.fft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
                }

                r += dist4;
            }
            dist4 = dist;
            dist >>= 2;
        }

        // FINAL ODD LAYER

        if dist4 == 2 {
            let mut r = output_start & !1;
            while r < truncated_size {
                let log_m = self.skew[r + skew_delta];

                let (x, y) = data.dist2_mut(pos + r, 1);

                if log_m == GF_MODULUS {
                    utils::xor(y, x);
                } else {
                    self.fft_butterfly_partial(x, y, log_m);
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// Gfni - PRIVATE - IFFT (inverse fast Fourier transform)

impl Gfni {
    #[inline(always)]
    fn ifftb_256(x: &mut [u8; 64], y: &mut [u8; 64], lut: LutGfni) {
        let x_ptr = x.as_mut_ptr().cast::<__m256i>();
        let y_ptr = y.as_mut_ptr().cast::<__m256i>();

        unsafe {
            let mut x_lo = _mm256_loadu_si256(x_ptr);
            let mut x_hi = _mm256_loadu_si256(x_ptr.add(1));

            let mut y_lo = _mm256_loadu_si256(y_ptr);
            let mut y_hi = _mm256_loadu_si256(y_ptr.add(1));

            y_lo = _mm256_xor_si256(y_lo, x_lo);
            y_hi = _mm256_xor_si256(y_hi, x_hi);

            _mm256_storeu_si256(y_ptr, y_lo);
            _mm256_storeu_si256(y_ptr.add(1), y_hi);

            (x_lo, x_hi) = Self::muladd_256(x_lo, x_hi, y_lo, y_hi, lut);

            _mm256_storeu_si256(x_ptr, x_lo);
            _mm256_storeu_si256(x_ptr.add(1), x_hi);
        }
    }

    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutGfni::from(&self.mul_gfni[log_m as usize]);

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            Self::ifftb_256(x_chunk, y_chunk, lut);
        }
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            self.ifft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            self.ifft_butterfly_partial(s2, s3, log_m23);
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            self.ifft_butterfly_partial(s0, s2, log_m02);
            self.ifft_butterfly_partial(s1, s3, log_m02);
        }
    }

    #[target_feature(enable = "gfni,avx2")]
    unsafe fn ifft_private_gfni(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.ifft_private(data, pos, size, truncated_size, skew_delta);
    }

    #[inline(always)]
    fn ifft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist = 1;
        let mut dist4 = 4;
        while dist4 <= size {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.ifft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            let log_m = self.skew[dist + skew_delta - 1];
            if log_m == GF_MODULUS {
                utils::xor_within(data, pos + dist, pos, dist);
            } else {
                let (mut a, mut b) = data.split_at_mut(pos + dist);
                for i in 0..dist {
                    self.ifft_butterfly_partial(
                        &mut a[pos + i], // data[pos + i]
                        &mut b[i],       // data[pos + i + dist]
                        log_m,
                    );
                }
            }
        }
    }
}

// ======================================================================
// Gfni - PRIVATE - Evaluate polynomial

impl Gfni {
    #[target_feature(enable = "avx2")]
    unsafe fn eval_poly_gfni(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        utils::eval_poly(erasures, truncated_size);
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
