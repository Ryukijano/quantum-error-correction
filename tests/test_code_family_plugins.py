import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codes import CircuitGenerationConfig, benchmark_code_families, list_plugins
from codes.dual_rail_erasure import DualRailErasureCodePlugin
from codes.bosonic import BosonicCodePlugin
from codes.qldpc import ClusteredCyclicCode, QLDPCCodePlugin


def test_plugin_registry_includes_requested_code_families():
    families = set(list_plugins())
    assert {"surface", "qldpc", "bosonic", "dual_rail_erasure"}.issubset(families)


def test_qldpc_raises_on_missing_inputs():
    # Should still raise if no variant/inputs provided
    plugin = QLDPCCodePlugin()
    with pytest.raises(NotImplementedError):
        plugin.build_circuit(CircuitGenerationConfig(distance=3, rounds=2, physical_error_rate=0.001))


def test_dual_rail_defaults_to_erasure_surface():
    # Now defaults to erasure_surface instead of raising error
    stim = pytest.importorskip("stim")
    plugin = DualRailErasureCodePlugin()
    circuit_str = plugin.build_circuit(CircuitGenerationConfig(distance=3, rounds=2, physical_error_rate=0.001))
    circuit = stim.Circuit(circuit_str)
    assert circuit.num_detectors > 0


def test_bosonic_defaults_to_gkp_surface():
    # Now defaults to gkp_surface
    stim = pytest.importorskip("stim")
    plugin = BosonicCodePlugin()
    circuit_str = plugin.build_circuit(CircuitGenerationConfig(distance=3, rounds=2, physical_error_rate=0.001))
    circuit = stim.Circuit(circuit_str)
    assert circuit.num_detectors > 0


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


def test_clustered_cyclic_code_builds_valid_stim_circuit():
    stim = pytest.importorskip("stim")

    circuit = ClusteredCyclicCode(distance=3, rounds=2, p=0.001, num_clusters=3, cluster_size=4, seed=11).build()

    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0


def test_qldpc_plugin_supports_clustered_cyclic_variant():
    stim = pytest.importorskip("stim")
    plugin = QLDPCCodePlugin()

    circuit_string = plugin.build_circuit(
        CircuitGenerationConfig(
            distance=3,
            rounds=2,
            physical_error_rate=0.001,
            extra_params={"variant": "clustered_cyclic", "num_clusters": 3, "cluster_size": 4, "seed": 11},
        )
    )

    circuit = stim.Circuit(circuit_string)
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0
