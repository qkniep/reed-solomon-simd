use crate::{
    engine::DefaultEngine,
    rate::{DefaultRate, DefaultRateDecoder, DefaultRateEncoder, Rate, RateDecoder, RateEncoder},
    DecoderResult, EncoderResult, Error,
};

// ======================================================================
// ReedSolomonEncoder - PUBLIC

/// Reed-Solomon encoder using [`DefaultEngine`] and [`DefaultRate`].
///
/// [`DefaultEngine`]: crate::engine::DefaultEngine
pub struct ReedSolomonEncoder(DefaultRateEncoder<DefaultEngine>);

impl ReedSolomonEncoder {
    /// Adds one original shard to the encoder.
    ///
    /// Original shards have indexes `0..original_count` corresponding to the order
    /// in which they are added and these same indexes must be used when decoding.
    ///
    /// See [basic usage](crate#basic-usage) for an example.
    pub fn add_original_shard<T: AsRef<[u8]>>(&mut self, original_shard: T) -> Result<(), Error> {
        self.0.add_original_shard(original_shard)
    }

    /// Encodes the added original shards returning [`EncoderResult`]
    /// which contains the generated recovery shards.
    ///
    /// When returned [`EncoderResult`] is dropped the encoder is
    /// automatically [`reset`] and ready for new round of encoding.
    ///
    /// See [basic usage](crate#basic-usage) for an example.
    ///
    /// [`reset`]: ReedSolomonEncoder::reset
    pub fn encode(&mut self) -> Result<EncoderResult<'_>, Error> {
        self.0.encode()
    }

    /// Creates new encoder with given configuration
    /// and allocates required working space.
    ///
    /// See [basic usage](crate#basic-usage) for an example.
    pub fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<Self, Error> {
        Ok(Self(DefaultRateEncoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            DefaultEngine::new(),
            None,
        )?))
    }

    /// Resets encoder to given configuration.
    ///
    /// - Added original shards are forgotten.
    /// - Existing working space is re-used if it's large enough
    ///   or re-allocated otherwise.
    pub fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        self.0.reset(original_count, recovery_count, shard_bytes)
    }

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use reed_solomon_simd::ReedSolomonEncoder;
    ///
    /// assert_eq!(ReedSolomonEncoder::supports(60_000, 4_000), true);
    /// assert_eq!(ReedSolomonEncoder::supports(60_000, 5_000), false);
    /// ```
    pub fn supports(original_count: usize, recovery_count: usize) -> bool {
        DefaultRate::<DefaultEngine>::supports(original_count, recovery_count)
    }
}

// ======================================================================
// ReedSolomonDecoder - PUBLIC

/// Reed-Solomon decoder using [`DefaultEngine`] and [`DefaultRate`].
///
/// [`DefaultEngine`]: crate::engine::DefaultEngine
pub struct ReedSolomonDecoder(DefaultRateDecoder<DefaultEngine>);

impl ReedSolomonDecoder {
    /// Adds one original shard to the decoder.
    ///
    /// - Shards can be added in any order.
    /// - Index must be the same that was used in encoding.
    ///
    /// See [basic usage](crate#basic-usage) for an example.
    pub fn add_original_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        original_shard: T,
    ) -> Result<(), Error> {
        self.0.add_original_shard(index, original_shard)
    }

    /// Adds one recovery shard to the decoder.
    ///
    /// - Shards can be added in any order.
    /// - Index must be the same that was used in encoding.
    ///
    /// See [basic usage](crate#basic-usage) for an example.
    pub fn add_recovery_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        recovery_shard: T,
    ) -> Result<(), Error> {
        self.0.add_recovery_shard(index, recovery_shard)
    }

    /// Decodes the added shards returning [`DecoderResult`]
    /// which contains the restored original shards.
    ///
    /// When returned [`DecoderResult`] is dropped the decoder is
    /// automatically [`reset`] and ready for new round of decoding.
    ///
    /// See [basic usage](crate#basic-usage) for an example.
    ///
    /// [`reset`]: ReedSolomonDecoder::reset
    pub fn decode(&mut self) -> Result<DecoderResult<'_>, Error> {
        self.0.decode()
    }

    /// Like [`decode`](ReedSolomonDecoder::decode), but also reconstructs any
    /// missing recovery shards as a by-product of the decode, avoiding a
    /// separate re-encode.
    ///
    /// The restored original shards are read as usual from the returned
    /// [`DecoderResult`]; the reconstructed recovery shards are read via
    /// [`DecoderResult::restored_recovery`] and
    /// [`DecoderResult::restored_recovery_iter`].
    ///
    /// Recovery shards are reconstructed **only when a decode actually runs**,
    /// i.e. when at least one original shard is missing. If all original shards
    /// are already present this does nothing extra and no recovery shards are
    /// reconstructed — encode instead.
    ///
    /// When the returned [`DecoderResult`] is dropped the decoder is
    /// automatically [`reset`] and ready for a new round of decoding.
    ///
    /// [`reset`]: ReedSolomonDecoder::reset
    pub fn decode_with_recovery(&mut self) -> Result<DecoderResult<'_>, Error> {
        self.0.decode_with_recovery()
    }

    /// Creates new decoder with given configuration
    /// and allocates required working space.
    ///
    /// See [basic usage](crate#basic-usage) for an example.
    pub fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<Self, Error> {
        Ok(Self(DefaultRateDecoder::new(
            original_count,
            recovery_count,
            shard_bytes,
            DefaultEngine::new(),
            None,
        )?))
    }

    /// Resets decoder to given configuration.
    ///
    /// - Added shards are forgotten.
    /// - Existing working space is re-used if it's large enough
    ///   or re-allocated otherwise.
    pub fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        self.0.reset(original_count, recovery_count, shard_bytes)
    }

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use reed_solomon_simd::ReedSolomonDecoder;
    ///
    /// assert_eq!(ReedSolomonDecoder::supports(60_000, 4_000), true);
    /// assert_eq!(ReedSolomonDecoder::supports(60_000, 5_000), false);
    /// ```
    pub fn supports(original_count: usize, recovery_count: usize) -> bool {
        DefaultRate::<DefaultEngine>::supports(original_count, recovery_count)
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use fixedbitset::FixedBitSet;

    use super::*;
    use crate::test_util;

    // ============================================================
    // HELPERS

    fn roundtrip(
        encoder: &mut ReedSolomonEncoder,
        decoder: &mut ReedSolomonDecoder,
        original_count: usize,
        recovery_hash: &str,
        decoder_original: &[usize],
        decoder_recovery: &[usize],
        seed: u8,
    ) {
        let original = test_util::generate_original(original_count, 1024, seed);

        for original in &original {
            encoder.add_original_shard(original).unwrap();
        }

        let result = encoder.encode().unwrap();
        let recovery: Vec<_> = result.recovery_iter().collect();

        test_util::assert_hash(&recovery, recovery_hash);

        let mut original_received = FixedBitSet::with_capacity(original_count);

        for i in decoder_original {
            decoder.add_original_shard(*i, &original[*i]).unwrap();
            original_received.set(*i, true);
        }

        for i in decoder_recovery {
            decoder.add_recovery_shard(*i, recovery[*i]).unwrap();
        }

        let result = decoder.decode().unwrap();
        let restored: BTreeMap<_, _> = result.restored_original_iter().collect();

        for i in 0..original_count {
            if !original_received[i] {
                assert_eq!(restored[&i], original[i]);
            }
        }
    }

    // ============================================================
    // ROUNDTRIP - TWO ROUNDS

    #[test]
    fn roundtrip_two_rounds_reset_low_to_high() {
        let mut encoder = ReedSolomonEncoder::new(2, 3, 1024).unwrap();
        let mut decoder = ReedSolomonDecoder::new(2, 3, 1024).unwrap();

        roundtrip(
            &mut encoder,
            &mut decoder,
            2,
            test_util::LOW_2_3,
            &[],
            &[0, 1],
            123,
        );

        encoder.reset(3, 2, 1024).unwrap();
        decoder.reset(3, 2, 1024).unwrap();

        roundtrip(
            &mut encoder,
            &mut decoder,
            3,
            test_util::HIGH_3_2,
            &[1],
            &[0, 1],
            132,
        );
    }

    // ==================================================
    // decode_with_recovery

    #[test]
    fn decode_with_recovery_reconstructs_missing_recovery() {
        let shard_bytes = 64;
        let original = test_util::generate_original(4, shard_bytes, 200);

        let mut encoder = ReedSolomonEncoder::new(4, 4, shard_bytes).unwrap();
        for o in &original {
            encoder.add_original_shard(o).unwrap();
        }
        let result = encoder.encode().unwrap();
        let recovery: Vec<Vec<u8>> = result.recovery_iter().map(<[u8]>::to_vec).collect();
        drop(result);

        // Lose originals {0, 2} and recovery {1, 3}.
        let mut decoder = ReedSolomonDecoder::new(4, 4, shard_bytes).unwrap();
        decoder.add_original_shard(1, &original[1]).unwrap();
        decoder.add_original_shard(3, &original[3]).unwrap();
        decoder.add_recovery_shard(0, &recovery[0]).unwrap();
        decoder.add_recovery_shard(2, &recovery[2]).unwrap();

        let result = decoder.decode_with_recovery().unwrap();

        assert_eq!(result.restored_original(0), Some(original[0].as_slice()));
        assert_eq!(result.restored_original(2), Some(original[2].as_slice()));

        let recovered: BTreeMap<usize, &[u8]> = result.restored_recovery_iter().collect();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[&1], recovery[1].as_slice());
        assert_eq!(recovered[&3], recovery[3].as_slice());
        assert_eq!(result.restored_recovery(0), None); // received
        assert_eq!(result.restored_recovery(2), None); // received
    }

    #[test]
    fn plain_decode_reports_no_recovery() {
        let shard_bytes = 64;
        let original = test_util::generate_original(3, shard_bytes, 201);

        let mut encoder = ReedSolomonEncoder::new(3, 3, shard_bytes).unwrap();
        for o in &original {
            encoder.add_original_shard(o).unwrap();
        }
        let result = encoder.encode().unwrap();
        let recovery: Vec<Vec<u8>> = result.recovery_iter().map(<[u8]>::to_vec).collect();
        drop(result);

        // Lose originals {0, 1} and recovery {2}; decode normally.
        let mut decoder = ReedSolomonDecoder::new(3, 3, shard_bytes).unwrap();
        decoder.add_original_shard(2, &original[2]).unwrap();
        decoder.add_recovery_shard(0, &recovery[0]).unwrap();
        decoder.add_recovery_shard(1, &recovery[1]).unwrap();

        let result = decoder.decode().unwrap();

        assert_eq!(result.restored_original(0), Some(original[0].as_slice()));
        assert_eq!(result.restored_original(1), Some(original[1].as_slice()));
        // Plain `decode` never reconstructs recovery, even though recovery 2 is missing.
        assert_eq!(result.restored_recovery(2), None);
        assert_eq!(result.restored_recovery_iter().count(), 0);
    }

    #[test]
    fn decode_with_recovery_noop_when_all_originals_present() {
        let shard_bytes = 64;
        let original = test_util::generate_original(3, shard_bytes, 202);

        let mut encoder = ReedSolomonEncoder::new(3, 3, shard_bytes).unwrap();
        for o in &original {
            encoder.add_original_shard(o).unwrap();
        }
        let result = encoder.encode().unwrap();
        let recovery: Vec<Vec<u8>> = result.recovery_iter().map(<[u8]>::to_vec).collect();
        drop(result);

        // All originals present, recovery {1, 2} missing — but no decode runs.
        let mut decoder = ReedSolomonDecoder::new(3, 3, shard_bytes).unwrap();
        for (i, o) in original.iter().enumerate() {
            decoder.add_original_shard(i, o).unwrap();
        }
        decoder.add_recovery_shard(0, &recovery[0]).unwrap();

        let result = decoder.decode_with_recovery().unwrap();

        // Nothing to decode, so nothing is reconstructed (documented behavior).
        assert!(result.restored_original_iter().next().is_none());
        assert_eq!(result.restored_recovery_iter().count(), 0);
        assert_eq!(result.restored_recovery(1), None);
        assert_eq!(result.restored_recovery(2), None);
    }

    // ==================================================
    // supports

    #[test]
    fn supports() {
        assert!(ReedSolomonEncoder::supports(4096, 61440));
        assert!(ReedSolomonEncoder::supports(61440, 4096));

        assert!(ReedSolomonDecoder::supports(4096, 61440));
        assert!(ReedSolomonDecoder::supports(61440, 4096));
    }
}
