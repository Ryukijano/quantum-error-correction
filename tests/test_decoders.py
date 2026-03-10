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


def test_mwpm_decoder_reports_pymatching_backend_when_available():
    pytest.importorskip("pymatching")

    detector_samples, observable_samples, metadata = _sample_detector_data(shots=32, p=0.01)
    output = MWPMDecoder().decode(detector_samples, metadata)

    assert output.diagnostics["backend"] == "pymatching"
    assert output.logical_predictions.shape == observable_samples.shape


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
