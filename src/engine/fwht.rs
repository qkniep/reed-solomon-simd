use crate::engine::{utils, GfElement, GF_ORDER};

// ======================================================================
// FWHT (fast Walsh-Hadamard transform) - CRATE

/// Decimation in time (DIT) Fast Walsh-Hadamard Transform.
/// `m_truncated`: Number of non-zero elements in `data` (at the front).
#[inline(always)]
pub(crate) fn fwht(data: &mut [GfElement; GF_ORDER], m_truncated: usize) {
    // Note to self: fwht_8 is slightly faster on x86 (AMD Ryzen 5 3600),
    // but slower on ARM (Apple silicon M1).
    // fwht_16 is always slower. See branch: AndersTrier/FWHT_8_and_16
    let mut dist = 1;
    let mut dist4 = 4;
    while dist4 <= GF_ORDER {
        for r in (0..m_truncated).step_by(dist4) {
            for offset in r..r + dist {
                fwht_4(data, offset as u16, dist as u16);
            }
        }

        dist = dist4;
        dist4 <<= 2;
    }
}

/// Like [`fwht`], but folds the input truncation instead of pruning it.
///
/// `truncated_size` is the number of non-zero elements at the front of `data`;
/// the rest must be zero. When only the first `m` inputs are non-zero, the WHT
/// output is exactly periodic with period `k = m.next_power_of_two()`: each
/// output index's high-order bits never enter `popcount(i & j)` because the
/// corresponding bits of every non-zero `j` are zero, so `out[i] = out[i % k]`.
///
/// We therefore run a full WHT over just the first `k` elements in
/// `O(k log k)` and replicate that block across the array in `O(GF_ORDER)`
/// copies, instead of the `O(GF_ORDER)` butterflies the high-stride stages of
/// [`fwht`] would otherwise perform. The result is bit-identical to [`fwht`].
#[inline(always)]
pub(crate) fn fwht_in_truncated(data: &mut [GfElement; GF_ORDER], truncated_size: usize) {
    let k = truncated_size.next_power_of_two();

    if k >= GF_ORDER {
        fwht(data, truncated_size);
        return;
    }

    // Full radix-2 WHT over the `k` leading elements (`data[truncated_size..k]`
    // is already zero).
    let mut dist = 1;
    while dist < k {
        for r in (0..k).step_by(dist * 2) {
            for offset in r..r + dist {
                let (sum, dif) = fwht_2(data[offset], data[offset + dist]);
                data[offset] = sum;
                data[offset + dist] = dif;
            }
        }
        dist <<= 1;
    }

    // The full WHT output is periodic with period `k`; replicate the block by
    // repeated doubling (`k` and `GF_ORDER` are both powers of two).
    let mut filled = k;
    while filled < GF_ORDER {
        let (head, tail) = data.split_at_mut(filled);
        tail[..filled].copy_from_slice(&head[..filled]);
        filled <<= 1;
    }
}

/// Decimation in time (DIT) Fast Walsh-Hadamard Transform that only computes
/// the first `output_count` outputs (rounded up to a power of two); the rest of
/// `data` is left in an unspecified state.
///
/// Unlike [`fwht`], this assumes `data` is fully populated (no input
/// truncation). The WHT is a tensor power, so zeroing the high-order index bits
/// of the output equals GF-summing the input over those bits. We therefore fold
/// `data` down to `k = output_count.next_power_of_two()` elements in `O(GF_ORDER)`
/// and run a full WHT over just those, in `O(k log k)`. The first `output_count`
/// outputs equal [`fwht`]'s mod `GF_MODULUS`; the two evaluation orders may
/// pick different encodings of zero (`0` vs `GF_MODULUS`), which all consumers
/// of the transform treat identically.
#[inline(always)]
pub(crate) fn fwht_out_truncated(data: &mut [GfElement; GF_ORDER], output_count: usize) {
    let k = output_count.next_power_of_two();

    if k >= GF_ORDER {
        fwht(data, GF_ORDER);
        return;
    }

    // Fold the high-order index bits by GF-summing strided groups:
    //   fold[i] = Σ_q data[q * k + i]   for i in 0..k
    for base in (k..GF_ORDER).step_by(k) {
        for i in 0..k {
            data[i] = utils::add_mod(data[i], data[base + i]);
        }
    }

    // Full radix-2 WHT over the `k` folded elements.
    let mut dist = 1;
    while dist < k {
        for r in (0..k).step_by(dist * 2) {
            for offset in r..r + dist {
                let (sum, dif) = fwht_2(data[offset], data[offset + dist]);
                data[offset] = sum;
                data[offset + dist] = dif;
            }
        }
        dist <<= 1;
    }
}

// ======================================================================
// FWHT - PRIVATE

#[inline(always)]
fn fwht_2(a: GfElement, b: GfElement) -> (GfElement, GfElement) {
    let sum = utils::add_mod(a, b);
    let dif = utils::sub_mod(a, b);
    (sum, dif)
}

#[inline(always)]
fn fwht_4(data: &mut [GfElement; GF_ORDER], offset: u16, dist: u16) {
    // Indices. u16 additions and multiplication to avoid bounds checks
    // on array access. (GF_ORDER == (u16::MAX+1))
    let i0 = usize::from(offset);
    let i1 = usize::from(offset + dist);
    let i2 = usize::from(offset + dist * 2);
    let i3 = usize::from(offset + dist * 3);

    let (s0, d0) = fwht_2(data[i0], data[i1]);
    let (s1, d1) = fwht_2(data[i2], data[i3]);
    let (s2, d2) = fwht_2(s0, s1);
    let (s3, d3) = fwht_2(d0, d1);

    data[i0] = s2;
    data[i1] = s3;
    data[i2] = d2;
    data[i3] = d3;
}

// ======================================================================
// FWHT - TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GF_MODULUS;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // Reference implementation
    fn fwht_naive(data: &mut [GfElement; GF_ORDER]) {
        let mut dist = 1;
        let mut dist2 = 2;
        while dist2 <= data.len() {
            for r in (0..data.len()).step_by(dist2) {
                for offset in r..r + dist {
                    let (sum, dif) = fwht_2_naive(data[offset], data[offset + dist]);
                    data[offset] = sum;
                    data[offset + dist] = dif;
                }
            }

            dist = dist2;
            dist2 *= 2;
        }
    }

    fn fwht_2_naive(a: GfElement, b: GfElement) -> (GfElement, GfElement) {
        let (mut sum, sum_overflow) = a.overflowing_add(b);
        if sum_overflow {
            // `sum` got reduced mod 65536, but we want to
            // reduce it mod GF_MODULUS (65535) instead.
            sum += 1;
        }

        let (mut dif, dif_overflow) = a.overflowing_sub(b);
        if dif_overflow {
            dif -= 1;
        }

        (sum, dif)
    }

    #[test]
    fn test_full() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        let mut data1 = [(); GF_ORDER].map(|_| rng.random());
        let mut data2 = data1;

        fwht(&mut data1, GF_ORDER);
        fwht_naive(&mut data2);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_truncated() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let random: Vec<GfElement> = (0..GF_ORDER).map(|_| rng.random()).collect();

        for nonzero_count in [
            0,
            1,
            2,
            3,
            4,
            64,
            127,
            16384 - 1,
            16384 + 1,
            GF_ORDER / 2 - 1,
            GF_ORDER / 2,
            GF_ORDER / 2 + 1,
            GF_ORDER - 4,
            GF_ORDER - 3,
            GF_ORDER - 2,
            GF_ORDER - 1,
            GF_ORDER,
        ] {
            let mut data1 = [0; GF_ORDER];

            data1[..nonzero_count].copy_from_slice(&random[..nonzero_count]);
            let mut data2 = data1;

            fwht(&mut data1, nonzero_count);
            fwht_naive(&mut data2);

            assert_eq!(data1, data2);
        }
    }

    #[test]
    fn test_in_truncated() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let random: Vec<GfElement> = (0..GF_ORDER).map(|_| rng.random()).collect();

        for nonzero_count in [
            0,
            1,
            2,
            3,
            4,
            42,
            64,
            127,
            16384 - 1,
            16384 + 1,
            GF_ORDER / 2 - 1,
            GF_ORDER / 2,
            GF_ORDER / 2 + 1,
            GF_ORDER - 4,
            GF_ORDER - 3,
            GF_ORDER - 2,
            GF_ORDER - 1,
            GF_ORDER,
        ] {
            let mut data1 = [0; GF_ORDER];

            data1[..nonzero_count].copy_from_slice(&random[..nonzero_count]);
            let mut data2 = data1;

            fwht_in_truncated(&mut data1, nonzero_count);
            fwht_naive(&mut data2);

            assert_eq!(data1, data2, "mismatch for nonzero_count = {nonzero_count}");
        }
    }

    #[test]
    fn test_out_truncated() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let random: [GfElement; GF_ORDER] = [(); GF_ORDER].map(|_| rng.random());

        // Outputs are only canonical mod GF_MODULUS: `0` and `GF_MODULUS` both
        // encode zero, and the two evaluation orders may pick different
        // encodings, so compare canonicalized values.
        let canonical = |x: GfElement| if x == GF_MODULUS { 0 } else { x };

        for output_count in [
            0,
            1,
            2,
            3,
            4,
            42,
            64,
            127,
            16384 - 1,
            16384 + 1,
            GF_ORDER / 2,
            GF_ORDER - 1,
            GF_ORDER,
        ] {
            let mut full = random;
            let mut truncated = random;

            fwht(&mut full, GF_ORDER);
            fwht_out_truncated(&mut truncated, output_count);

            // The first `output_count` outputs must match the full transform.
            assert_eq!(
                full[..output_count]
                    .iter()
                    .copied()
                    .map(canonical)
                    .collect::<Vec<_>>(),
                truncated[..output_count]
                    .iter()
                    .copied()
                    .map(canonical)
                    .collect::<Vec<_>>(),
                "mismatch for output_count = {output_count}"
            );
        }
    }
}
