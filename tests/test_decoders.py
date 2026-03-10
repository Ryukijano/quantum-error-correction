import pytest

np = pytest.importorskip("numpy")
stim = pytest.importorskip("stim")

from surface_code_in_stem.decoders import (
    DecoderMetadata,
    MWPMDecoder,
    SparseBlossomDecoder,
    UnionFindDecoder,
)
from surface_code_in_stem.rl_nested_learning import _logical_error_rate
from surface_code_in_stem.surface_code import surface_code_circuit_string


def _sample_detector_data(*, distance: int = 3, rounds: int = 2, p: float = 0.001, shots: int = 12, seed: int = 99):
    circuit = stim.Circuit(surface_code_circuit_string(distance, rounds, p))
    detector_samples, observable_samples = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots,
        separate_observables=True,
    )
    metadata = DecoderMetadata(
        num_observables=circuit.num_observables,
        detector_error_model=circuit.detector_error_model(decompose_errors=True),
        circuit=circuit,
        seed=seed,
    )
    return detector_samples, observable_samples, metadata


def test_mwpm_decoder_output_is_deterministic_for_fixed_inputs():
    detector_samples, _, metadata = _sample_detector_data()
    decoder = MWPMDecoder()

    first = decoder.decode(detector_samples, metadata).logical_predictions
    second = decoder.decode(detector_samples, metadata).logical_predictions

    np.testing.assert_array_equal(first, second)


def test_union_find_and_sparse_blossom_match_mwpm_on_small_circuit():
    detector_samples, _, metadata = _sample_detector_data()

    mwpm_predictions = MWPMDecoder().decode(detector_samples, metadata).logical_predictions
    union_find_predictions = UnionFindDecoder().decode(detector_samples, metadata).logical_predictions
    sparse_predictions = SparseBlossomDecoder().decode(detector_samples, metadata).logical_predictions

    np.testing.assert_array_equal(union_find_predictions, mwpm_predictions)
    np.testing.assert_array_equal(sparse_predictions, mwpm_predictions)


def test_logical_error_rate_with_decoder_is_seed_deterministic():
    circuit_string = surface_code_circuit_string(3, 3, 0.001)

    first = _logical_error_rate(circuit_string, shots=16, seed=123, decoder=MWPMDecoder())
    second = _logical_error_rate(circuit_string, shots=16, seed=123, decoder=MWPMDecoder())

    assert first == second


@pytest.mark.parametrize(
    "detector_events",
    [
        np.zeros(4, dtype=np.bool_),
        np.zeros((2, 2, 2), dtype=np.bool_),
    ],
)
def test_mwpm_decoder_rejects_non_2d_detector_events(detector_events):
    decoder = MWPMDecoder()
    metadata = DecoderMetadata(num_observables=1)

    with pytest.raises(ValueError, match="detector_events must be a 2D bool array"):
        decoder.decode(detector_events, metadata)


def test_mwpm_decoder_rejects_zero_observables_metadata():
    decoder = MWPMDecoder()
    detector_events = np.zeros((2, 3), dtype=np.bool_)
    metadata = DecoderMetadata(num_observables=0)

    with pytest.raises(ValueError, match="num_observables must be > 0"):
        decoder.decode(detector_events, metadata)


def test_mwpm_decoder_raises_on_backend_shape_mismatch(monkeypatch):
    decoder = MWPMDecoder()
    detector_events = np.zeros((2, 1), dtype=np.bool_)
    metadata = DecoderMetadata(num_observables=1)

    def _raise_importerror(*_: object, **__: object):
        raise ImportError

    def _bad_shape(*_: object, **__: object):
        return np.zeros((detector_events.shape[0], metadata.num_observables + 1), dtype=np.bool_)

    monkeypatch.setattr(decoder, "_decode_with_pymatching", _raise_importerror)
    monkeypatch.setattr(decoder, "_fallback_decode", _bad_shape)

    with pytest.raises(ValueError, match="unexpected shape"):
        decoder.decode(detector_events, metadata)
