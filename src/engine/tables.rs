//! Lookup-tables used by [`Engine`]:s.
//!
//! All tables are global and each is initialized at most once.
//!
//! # Tables
//!
//! | Table        | Size    | Used in encoding | Used in decoding | By engines         |
//! | ------------ | ------- | ---------------- | ---------------- | ------------------ |
//! | [`Exp`]      | 128 kiB | yes              | yes              | all                |
//! | [`Log`]      | 128 kiB | yes              | yes              | all                |
//! | [`LogWalsh`] | 128 kiB | -                | yes              | all                |
//! | [`Mul16`]    | 8 MiB   | yes              | yes              | [`NoSimd`]         |
//! | [`Mul128`]   | 8 MiB   | yes              | yes              | [`Avx2`] [`Ssse3`] |
//! | [`Skew`]     | 128 kiB | yes              | yes              | all                |
//!
//! [`NoSimd`]: crate::engine::NoSimd
//! [`Avx2`]: crate::engine::Avx2
//! [`Ssse3`]: crate::engine::Ssse3
//! [`Engine`]: crate::engine
//!

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use once_cell::race::OnceBox;
#[cfg(feature = "std")]
use std::sync::LazyLock;

use crate::engine::{
    fwht, utils, GfElement, CANTOR_BASIS, GF_BITS, GF_MODULUS, GF_ORDER, GF_POLYNOMIAL,
};

// ======================================================================
// TYPE ALIASES - PUBLIC

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
/// [`Engine`]: crate::engine
pub type Exp = [GfElement; GF_ORDER];

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
/// [`Engine`]: crate::engine
pub type Log = [GfElement; GF_ORDER];

/// Used by [`Avx2`] and [`Ssse3`] engines for multiplications.
///
/// [`Avx2`]: crate::engine::Avx2
/// [`Ssse3`]: crate::engine::Ssse3
pub type Mul128 = [Multiply128lutT; GF_ORDER];

/// Elements of the Mul128 table
#[derive(Clone, Debug)]
pub struct Multiply128lutT {
    /// Lower half of `GfElements`
    pub lo: [u128; 4],
    /// Upper half of `GfElements`
    pub hi: [u128; 4],
}

/// Used by the [`Gfni`] engine for multiplications.
///
/// [`Gfni`]: crate::engine::Gfni
#[cfg(all(feature = "gfni", any(target_arch = "x86", target_arch = "x86_64")))]
pub type MulGfni = [MultiplyGfniT; GF_ORDER];

/// Elements of the [`MulGfni`] table.
///
/// Multiply-by-constant in `GF(2^16)` is `GF(2)`-linear, hence a 16×16 bit
/// matrix. Splitting each element into a low and high byte decomposes it into
/// four 8×8 `GF(2)` affine maps, each packed as a `u64` in the byte order
/// consumed by `GF2P8AFFINEQB`. Naming is `<output byte>_<input byte>`.
#[cfg(all(feature = "gfni", any(target_arch = "x86", target_arch = "x86_64")))]
#[derive(Clone, Copy, Debug)]
pub struct MultiplyGfniT {
    /// Low output byte from low input byte.
    pub lo_lo: u64,
    /// High output byte from low input byte.
    pub hi_lo: u64,
    /// Low output byte from high input byte.
    pub lo_hi: u64,
    /// High output byte from high input byte.
    pub hi_hi: u64,
}

/// Used by all [`Engine`]:s in [`Engine::eval_poly`].
///
/// [`Engine`]: crate::engine
/// [`Engine::eval_poly`]: crate::engine::Engine::eval_poly
pub type LogWalsh = [GfElement; GF_ORDER];

/// Used by [`NoSimd`] engine for multiplications.
///
/// [`NoSimd`]: crate::engine::NoSimd
pub type Mul16 = [[[GfElement; 16]; 4]; GF_ORDER];

/// Used by all [`Engine`]:s for FFT and IFFT.
///
/// [`Engine`]: crate::engine
pub type Skew = [GfElement; GF_MODULUS as usize];

// ======================================================================
// ExpLog - PUBLIC

/// Struct holding the [`Exp`] and [`Log`] lookup tables.
pub struct ExpLog {
    /// Exponentiation table.
    pub exp: Box<Exp>,
    /// Logarithm table.
    pub log: Box<Log>,
}

// ======================================================================
// STATIC - PUBLIC

/// Lazily initialized exponentiation and logarithm tables.
pub fn get_exp_log() -> &'static ExpLog {
    #[cfg(feature = "std")]
    {
        static EXP_LOG: LazyLock<ExpLog> = LazyLock::new(initialize_exp_log);
        &EXP_LOG
    }
    #[cfg(not(feature = "std"))]
    {
        static EXP_LOG: OnceBox<ExpLog> = OnceBox::new();
        EXP_LOG.get_or_init(|| Box::new(initialize_exp_log()))
    }
}

/// Lazily initialized logarithmic Walsh transform table.
pub fn get_log_walsh() -> &'static LogWalsh {
    #[cfg(feature = "std")]
    {
        static LOG_WALSH: LazyLock<Box<LogWalsh>> = LazyLock::new(initialize_log_walsh);
        &LOG_WALSH
    }
    #[cfg(not(feature = "std"))]
    {
        static LOG_WALSH: OnceBox<LogWalsh> = OnceBox::new();
        LOG_WALSH.get_or_init(initialize_log_walsh)
    }
}

/// Lazily initialized multiplication table for the `NoSimd` engine.
pub fn get_mul16() -> &'static Mul16 {
    #[cfg(feature = "std")]
    {
        static MUL16: LazyLock<Box<Mul16>> = LazyLock::new(initialize_mul16);
        &MUL16
    }
    #[cfg(not(feature = "std"))]
    {
        static MUL16: OnceBox<Mul16> = OnceBox::new();
        MUL16.get_or_init(initialize_mul16)
    }
}

/// Lazily initialized multiplication table for SIMD engines.
pub fn get_mul128() -> &'static Mul128 {
    #[cfg(feature = "std")]
    {
        static MUL128: LazyLock<Box<Mul128>> = LazyLock::new(initialize_mul128);
        &MUL128
    }
    #[cfg(not(feature = "std"))]
    {
        static MUL128: OnceBox<Mul128> = OnceBox::new();
        MUL128.get_or_init(initialize_mul128)
    }
}

/// Lazily initialized affine-matrix multiplication table for the `Gfni` engine.
#[cfg(all(feature = "gfni", any(target_arch = "x86", target_arch = "x86_64")))]
pub fn get_mul_gfni() -> &'static MulGfni {
    #[cfg(feature = "std")]
    {
        static MUL_GFNI: LazyLock<Box<MulGfni>> = LazyLock::new(initialize_mul_gfni);
        &MUL_GFNI
    }
    #[cfg(not(feature = "std"))]
    {
        static MUL_GFNI: OnceBox<MulGfni> = OnceBox::new();
        MUL_GFNI.get_or_init(initialize_mul_gfni)
    }
}

/// Lazily initialized skew table used in FFT and IFFT operations.
pub fn get_skew() -> &'static Skew {
    #[cfg(feature = "std")]
    {
        static SKEW: LazyLock<Box<Skew>> = LazyLock::new(initialize_skew);
        &SKEW
    }
    #[cfg(not(feature = "std"))]
    {
        static SKEW: OnceBox<Skew> = OnceBox::new();
        SKEW.get_or_init(initialize_skew)
    }
}

// ======================================================================
// FUNCTIONS - PUBLIC - math

/// Calculates `x * log_m` using [`Exp`] and [`Log`] tables.
#[inline(always)]
pub fn mul(x: GfElement, log_m: GfElement, exp: &Exp, log: &Log) -> GfElement {
    if x == 0 {
        0
    } else {
        exp[utils::add_mod(log[x as usize], log_m) as usize]
    }
}

// ======================================================================
// FUNCTIONS - PRIVATE - initialize tables

#[allow(clippy::needless_range_loop)]
fn initialize_exp_log() -> ExpLog {
    let mut exp = Box::new([0; GF_ORDER]);
    let mut log = Box::new([0; GF_ORDER]);

    // GENERATE LFSR TABLE

    let mut state = 1;
    for i in 0..GF_MODULUS {
        exp[state] = i;
        state <<= 1;
        if state >= GF_ORDER {
            state ^= GF_POLYNOMIAL;
        }
    }
    exp[0] = GF_MODULUS;

    // CONVERT TO CANTOR BASIS

    log[0] = 0;
    for i in 0..GF_BITS {
        let width = 1usize << i;
        for j in 0..width {
            log[j + width] = log[j] ^ CANTOR_BASIS[i];
        }
    }

    for i in 0..GF_ORDER {
        log[i] = exp[log[i] as usize];
    }

    for i in 0..GF_ORDER {
        exp[log[i] as usize] = i as GfElement;
    }

    exp[GF_MODULUS as usize] = exp[0];

    ExpLog { exp, log }
}

fn initialize_log_walsh() -> Box<LogWalsh> {
    let log = get_exp_log().log.as_slice();

    let mut log_walsh: Box<LogWalsh> = Box::new([0; GF_ORDER]);

    log_walsh.copy_from_slice(log);
    log_walsh[0] = 0;
    fwht::fwht(log_walsh.as_mut(), GF_ORDER);

    log_walsh
}

fn initialize_mul16() -> Box<Mul16> {
    let exp = &get_exp_log().exp;
    let log = &get_exp_log().log;
    let mut mul16 = vec![[[0; 16]; 4]; GF_ORDER];

    for log_m in 0..=GF_MODULUS {
        let lut = &mut mul16[log_m as usize];
        for i in 0..16 {
            lut[0][i] = mul(i as GfElement, log_m, exp, log);
            lut[1][i] = mul((i << 4) as GfElement, log_m, exp, log);
            lut[2][i] = mul((i << 8) as GfElement, log_m, exp, log);
            lut[3][i] = mul((i << 12) as GfElement, log_m, exp, log);
        }
    }

    mul16.into_boxed_slice().try_into().unwrap()
}

fn initialize_mul128() -> Box<Mul128> {
    // Based on:
    // https://github.com/catid/leopard/blob/22ddc7804998d31c8f1a2617ee720e063b1fa6cd/LeopardFF16.cpp#L375
    let exp = &get_exp_log().exp;
    let log = &get_exp_log().log;

    let mut mul128 = vec![
        Multiply128lutT {
            lo: [0; 4],
            hi: [0; 4],
        };
        GF_ORDER
    ];

    for log_m in 0..=GF_MODULUS {
        for i in 0..=3 {
            let mut prod_lo = [0u8; 16];
            let mut prod_hi = [0u8; 16];
            for x in 0..16 {
                let prod = mul((x << (i * 4)) as GfElement, log_m, exp, log);
                prod_lo[x] = prod as u8;
                prod_hi[x] = (prod >> 8) as u8;
            }
            mul128[log_m as usize].lo[i] = u128::from_le_bytes(prod_lo);
            mul128[log_m as usize].hi[i] = u128::from_le_bytes(prod_hi);
        }
    }

    mul128.into_boxed_slice().try_into().unwrap()
}

/// Packs one 8×8 `GF(2)` affine matrix as a `u64` in `GF2P8AFFINEQB` byte order.
///
/// `cols[k]` is the full `GF(2^16)` product contribution when bit `k` of the
/// relevant input byte is set; `hi_byte` selects which output byte this block
/// feeds. `GF2P8AFFINEQB` computes `out.bit[i] = parity(A.byte[7 - i] & x)`, so
/// row `7 - j` of the map (coefficients of output bit `7 - j`) is packed into
/// `A.byte[j]`.
#[cfg(all(feature = "gfni", any(target_arch = "x86", target_arch = "x86_64")))]
fn build_affine(cols: &[GfElement; 8], hi_byte: bool) -> u64 {
    let mut bytes = [0u8; 8];
    for (j, byte) in bytes.iter_mut().enumerate() {
        let out_bit = 7 - j;
        let mut b = 0u8;
        for (k, &col) in cols.iter().enumerate() {
            let col_byte = if hi_byte { (col >> 8) as u8 } else { col as u8 };
            b |= ((col_byte >> out_bit) & 1) << k;
        }
        *byte = b;
    }
    u64::from_le_bytes(bytes)
}

#[cfg(all(feature = "gfni", any(target_arch = "x86", target_arch = "x86_64")))]
fn initialize_mul_gfni() -> Box<MulGfni> {
    let exp = &get_exp_log().exp;
    let log = &get_exp_log().log;

    let mut mul_gfni = vec![
        MultiplyGfniT {
            lo_lo: 0,
            hi_lo: 0,
            lo_hi: 0,
            hi_hi: 0,
        };
        GF_ORDER
    ];

    for log_m in 0..=GF_MODULUS {
        // Product contributions of each input bit, split by input byte.
        let mut c_lo = [0; 8];
        let mut c_hi = [0; 8];
        for k in 0..8 {
            c_lo[k] = mul(1 << k, log_m, exp, log);
            c_hi[k] = mul(1 << (8 + k), log_m, exp, log);
        }
        mul_gfni[log_m as usize] = MultiplyGfniT {
            lo_lo: build_affine(&c_lo, false),
            hi_lo: build_affine(&c_lo, true),
            lo_hi: build_affine(&c_hi, false),
            hi_hi: build_affine(&c_hi, true),
        };
    }

    mul_gfni.into_boxed_slice().try_into().unwrap()
}

#[allow(clippy::needless_range_loop)]
fn initialize_skew() -> Box<Skew> {
    let exp = &get_exp_log().exp;
    let log = &get_exp_log().log;

    let mut skew = Box::new([0; GF_MODULUS as usize]);

    let mut temp = [0; GF_BITS - 1];

    for i in 1..GF_BITS {
        temp[i - 1] = 1 << i;
    }

    for m in 0..GF_BITS - 1 {
        let step: usize = 1 << (m + 1);

        skew[(1 << m) - 1] = 0;

        for i in m..GF_BITS - 1 {
            let s: usize = 1 << (i + 1);
            let mut j = (1 << m) - 1;
            while j < s {
                skew[j + s] = skew[j] ^ temp[i];
                j += step;
            }
        }

        temp[m] = GF_MODULUS - log[mul(temp[m], log[(temp[m] ^ 1) as usize], exp, log) as usize];

        for i in m + 1..GF_BITS - 1 {
            let sum = utils::add_mod(log[(temp[i] ^ 1) as usize], temp[m]);
            temp[i] = mul(temp[i], sum, exp, log);
        }
    }

    for i in 0..GF_MODULUS as usize {
        skew[i] = log[skew[i] as usize];
    }

    skew
}

// ======================================================================
// TESTS

#[cfg(all(
    test,
    feature = "gfni",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod gfni_tests {
    use super::{get_exp_log, get_mul_gfni, mul};
    use crate::engine::{GfElement, GF_MODULUS};

    /// Scalar model of one byte of `VGF2P8AFFINEQB` with `imm8 == 0`:
    /// `out.bit[i] = parity(A.byte[7 - i] & x)`. Mirrors the hardware
    /// instruction so the built [`MulGfni`] table can be checked without a
    /// GFNI-capable CPU.
    fn affine_byte(a: u64, x: u8) -> u8 {
        let ab = a.to_le_bytes();
        let mut y = 0u8;
        for i in 0..8 {
            let bit = u8::try_from((ab[7 - i] & x).count_ones() & 1).unwrap();
            y |= bit << i;
        }
        y
    }

    #[test]
    fn mul_gfni_matches_field_multiply() {
        let exp_log = get_exp_log();
        let (exp, log) = (&exp_log.exp, &exp_log.log);
        let table = get_mul_gfni();

        // Representative multipliers incl. the boundary log values.
        for &log_m in &[
            0,
            1,
            2,
            3,
            100,
            255,
            256,
            1000,
            32767,
            32768,
            GF_MODULUS - 1,
            GF_MODULUS,
        ] {
            let m = &table[log_m as usize];
            for v in 0..=GF_MODULUS {
                let lo = v as u8;
                let hi = (v >> 8) as u8;
                let prod_lo = affine_byte(m.lo_lo, lo) ^ affine_byte(m.lo_hi, hi);
                let prod_hi = affine_byte(m.hi_lo, lo) ^ affine_byte(m.hi_hi, hi);
                let got = GfElement::from(prod_lo) | (GfElement::from(prod_hi) << 8);
                assert_eq!(got, mul(v, log_m, exp, log), "log_m={log_m} v={v}");
            }
        }
    }
}
