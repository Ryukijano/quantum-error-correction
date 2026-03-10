import pytest

np = pytest.importorskip("numpy")
stim = pytest.importorskip("stim")

from surface_code_in_stem.decoders import (
    DecoderMetadata,
    MWPMDecoder,
    SparseBlossomDecoder,
    UnionFindDecoder,
)
from surface_code_in_stem.decoders.mwpm import PymatchingConfigurationError
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


def test_mwpm_decoder_does_not_mask_unexpected_value_error(monkeypatch):
    decoder = MWPMDecoder()
    metadata = DecoderMetadata(num_observables=1)
    detector_events = np.zeros((2, 1), dtype=bool)

    def boom(*_: object) -> None:
        raise ValueError("unexpected pymatching failure")

    monkeypatch.setattr(decoder, "_decode_with_pymatching", boom)

    with pytest.raises(ValueError):
        decoder.decode(detector_events, metadata)


def test_mwpm_decoder_falls_back_on_expected_configuration_issue(monkeypatch):
    decoder = MWPMDecoder()
    metadata = DecoderMetadata(num_observables=2)
    detector_events = np.zeros((3, 1), dtype=bool)

    def missing_dem(*_: object) -> None:
        raise PymatchingConfigurationError("no detector error model provided")

    monkeypatch.setattr(decoder, "_decode_with_pymatching", missing_dem)

    output = decoder.decode(detector_events, metadata)

    np.testing.assert_array_equal(output.logical_predictions, np.zeros((3, 2), dtype=bool))
    assert output.diagnostics["backend"] == "fallback"
