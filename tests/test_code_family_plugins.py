import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codes import CircuitGenerationConfig, benchmark_code_families, list_plugins
from codes.dual_rail_erasure import DualRailErasureCodePlugin
from codes.qldpc import QLDPCCodePlugin


def test_plugin_registry_includes_requested_code_families():
    families = set(list_plugins())
    assert {"surface", "qldpc", "bosonic", "dual_rail_erasure"}.issubset(families)


def test_qldpc_placeholder_error_message_is_explicit():
    plugin = QLDPCCodePlugin()
    with pytest.raises(NotImplementedError, match="Required parity-check inputs: 'hx' and 'hz'"):
        plugin.build_circuit(CircuitGenerationConfig(distance=3, rounds=2, physical_error_rate=0.001))


def test_dual_rail_placeholder_error_message_is_explicit():
    plugin = DualRailErasureCodePlugin()
    with pytest.raises(NotImplementedError, match="'parity_check' and 'erasure_map'"):
        plugin.build_circuit(CircuitGenerationConfig(distance=3, rounds=2, physical_error_rate=0.001))


def test_benchmark_harness_runs_surface_family_with_shared_api():
    np = pytest.importorskip("numpy")
    pytest.importorskip("stim")

    config = CircuitGenerationConfig(
        distance=3,
        rounds=2,
        physical_error_rate=0.001,
        extra_params={"variant": "static"},
    )
    results = benchmark_code_families(["surface"], config, shots=8, seed=5)

    assert set(results) == {"surface"}
    assert 0.0 <= results["surface"]["logical_error_rate"] <= 1.0
    assert np.isfinite(results["surface"]["logical_error_rate"])
    assert results["surface"]["decoder_metadata"]["syndrome_format"] == "stim_detector_sampler"
