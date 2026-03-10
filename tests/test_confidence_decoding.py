import pytest

np = pytest.importorskip("numpy")
stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from surface_code_in_stem.confidence_decoding import SyndromeBatch, WeightedMWPMDecoder


def test_syndrome_batch_validates_shapes_and_ranges():
    hard = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    confidence = np.array([[0.5, 1.0], [0.0, 0.2]], dtype=np.float64)

    batch = SyndromeBatch(hard_bits=hard, confidence=confidence)
    assert batch.hard_bits.shape == (2, 2)
    assert batch.confidence is not None

    with pytest.raises(ValueError, match="same shape"):
        SyndromeBatch(hard_bits=hard, confidence=np.array([[0.1], [0.2]]))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SyndromeBatch(hard_bits=hard, confidence=np.array([[1.2, 0.5], [0.5, 0.5]]))


def test_weighted_decoder_falls_back_to_hard_behavior_without_confidence():
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        before_round_data_depolarization=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    decoder = WeightedMWPMDecoder(dem, confidence_scale=1.5)

    sampler = circuit.compile_detector_sampler(seed=11)
    hard_bits, _ = sampler.sample(shots=40, separate_observables=True)
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)

    predicted_without_confidence = decoder.decode_batch(SyndromeBatch(hard_bits=hard_bits))

    confidence_all_ones = np.ones_like(hard_bits, dtype=np.float64)
    predicted_with_neutral_confidence = decoder.decode_batch(
        SyndromeBatch(hard_bits=hard_bits, confidence=confidence_all_ones)
    )

    np.testing.assert_array_equal(predicted_without_confidence, predicted_with_neutral_confidence)
