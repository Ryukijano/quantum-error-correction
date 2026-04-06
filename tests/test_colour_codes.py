"""Tests for colour code specific functionality.

Tests for ColorCodeStimBuilder, LoomColorCodeBuilder, ConcatenatedMWPMDecoder,
and the parallel threshold estimator with colour codes.
"""
from __future__ import annotations

import pytest
import numpy as np

try:
    from color_code_stim import ColorCode, NoiseModel as CCNoiseModel
    COLOR_CODE_AVAILABLE = True
except ImportError:
    COLOR_CODE_AVAILABLE = False

try:
    import loom
    LOOM_AVAILABLE = True
except ImportError:
    LOOM_AVAILABLE = False

from syndrome_net import CircuitSpec
from syndrome_net.container import get_container, reset_container


# Skip all tests if color-code-stim not available
pytestmark = pytest.mark.skipif(
    not COLOR_CODE_AVAILABLE,
    reason="color-code-stim not installed"
)


class TestColorCodeStimBuilder:
    """Tests for ColorCodeStimBuilder."""
    
    def test_builder_imports(self):
        """Builder must be importable from syndrome_net.codes."""
        from syndrome_net.codes import ColorCodeStimBuilder
        builder = ColorCodeStimBuilder()
        assert builder.name == "color_code"
    
    def test_supported_distances(self):
        """Builder must report supported distances."""
        from syndrome_net.codes import ColorCodeStimBuilder
        builder = ColorCodeStimBuilder()
        distances = builder.supported_distances
        assert len(distances) > 0
        assert all(isinstance(d, int) for d in distances)
    
    def test_build_triangular_circuit(self):
        """Must build triangular colour code circuit."""
        from syndrome_net.codes import ColorCodeStimBuilder
        builder = ColorCodeStimBuilder()
        
        spec = CircuitSpec(distance=5, rounds=5, error_probability=0.001, circuit_type="tri")
        circuit = builder.build(spec)
        
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1
    
    def test_build_rectangular_circuit(self):
        """Must build rectangular colour code circuit."""
        from syndrome_net.codes import ColorCodeStimBuilder
        builder = ColorCodeStimBuilder()
        
        spec = CircuitSpec(distance=6, rounds=6, error_probability=0.001, circuit_type="rec")
        circuit = builder.build(spec)
        
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
    
    def test_invalid_triangular_even_distance(self):
        """Triangular colour code must reject even distance."""
        from syndrome_net.codes import ColorCodeStimBuilder
        from syndrome_net import InvalidSpecError
        
        builder = ColorCodeStimBuilder()
        spec = CircuitSpec(distance=4, rounds=4, error_probability=0.001, circuit_type="tri")
        
        with pytest.raises(InvalidSpecError):
            builder.build(spec)
    
    def test_superdense_option(self):
        """Must support superdense circuit option."""
        from syndrome_net.codes import ColorCodeStimBuilder
        builder = ColorCodeStimBuilder()
        
        spec = CircuitSpec(
            distance=5, rounds=5, error_probability=0.001,
            circuit_type="tri", superdense=True
        )
        circuit = builder.build(spec)
        
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0


class TestLoomColorCodeBuilder:
    """Tests for LoomColorCodeBuilder."""
    
    @pytest.mark.skipif(not LOOM_AVAILABLE, reason="el-loom not installed")
    def test_builder_imports(self):
        """Builder must be importable from syndrome_net.codes."""
        from syndrome_net.codes import LoomColorCodeBuilder
        builder = LoomColorCodeBuilder()
        assert builder.name == "loom_color_code"
    
    @pytest.mark.skipif(not LOOM_AVAILABLE, reason="el-loom not installed")
    def test_supported_distances(self):
        """Builder must report supported distances."""
        from syndrome_net.codes import LoomColorCodeBuilder
        builder = LoomColorCodeBuilder()
        distances = builder.supported_distances
        assert len(distances) > 0
        assert all(isinstance(d, int) and d % 2 == 1 for d in distances)
    
    @pytest.mark.skipif(not LOOM_AVAILABLE, reason="el-loom not installed")
    def test_build_circuit(self):
        """Must build colour code circuit via Loom."""
        from syndrome_net.codes import LoomColorCodeBuilder
        builder = LoomColorCodeBuilder()
        
        spec = CircuitSpec(distance=5, rounds=5, error_probability=0.001)
        circuit = builder.build(spec)
        
        assert circuit.num_qubits > 0
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1
    
    @pytest.mark.skipif(not LOOM_AVAILABLE, reason="el-loom not installed")
    def test_invalid_even_distance(self):
        """Loom colour code must reject even distance."""
        from syndrome_net.codes import LoomColorCodeBuilder
        from syndrome_net import InvalidSpecError
        
        builder = LoomColorCodeBuilder()
        spec = CircuitSpec(distance=4, rounds=4, error_probability=0.001)
        
        with pytest.raises(InvalidSpecError):
            builder.build(spec)


class TestConcatenatedMWPMDecoder:
    """Tests for ConcatenatedMWPMDecoder."""
    
    def test_decoder_imports(self):
        """Decoder must be importable from syndrome_net.decoders."""
        from syndrome_net.decoders import ConcatenatedMWPMDecoder
        decoder = ConcatenatedMWPMDecoder()
        assert decoder.name == "concat_mwpm"
    
    def test_decode_with_color_code(self):
        """Decoder must decode colour code syndromes."""
        from syndrome_net.decoders import ConcatenatedMWPMDecoder
        from syndrome_net import DecoderMetadata
        
        # Build colour code circuit
        noise = CCNoiseModel.uniform_circuit_noise(0.01)
        cc = ColorCode(d=5, rounds=5, circuit_type="tri", noise_model=noise)
        circuit = cc.circuit
        
        # Create decoder
        decoder = ConcatenatedMWPMDecoder()
        decoder.attach_color_code(cc)
        
        # Sample syndrome
        sampler = circuit.compile_detector_sampler(seed=42)
        det, obs = sampler.sample(1, separate_observables=True)
        
        # Decode
        metadata = DecoderMetadata(
            num_observables=circuit.num_observables,
            detector_error_model=circuit.detector_error_model(decompose_errors=True),
            circuit=circuit,
            seed=42
        )
        
        result = decoder.decode(det, metadata)
        
        # Result should have correct shape
        assert result.logical_predictions.shape == (1, circuit.num_observables)
    
    def test_decoder_batch(self):
        """Decoder must support batch decoding."""
        from syndrome_net.decoders import ConcatenatedMWPMDecoder
        from syndrome_net import DecoderMetadata
        
        noise = CCNoiseModel.uniform_circuit_noise(0.01)
        cc = ColorCode(d=5, rounds=5, circuit_type="tri", noise_model=noise)
        circuit = cc.circuit
        
        decoder = ConcatenatedMWPMDecoder()
        decoder.attach_color_code(cc)
        
        sampler = circuit.compile_detector_sampler(seed=42)
        det, obs = sampler.sample(10, separate_observables=True)
        
        metadata = DecoderMetadata(
            num_observables=circuit.num_observables,
            detector_error_model=circuit.detector_error_model(decompose_errors=True),
            circuit=circuit,
            seed=42
        )
        
        result = decoder.decode_circuit(det, metadata)
        
        assert result.logical_predictions.shape == (10, circuit.num_observables)


class TestColourCodeRLEnvironments:
    """Tests for colour code RL environments."""
    
    def test_gym_env_imports(self):
        """Environments must be importable."""
        from surface_code_in_stem.rl_control.gym_env import ColourCodeGymEnv
        env = ColourCodeGymEnv(distance=3, rounds=3, physical_error_rate=0.001)
        assert env.distance == 3
        assert env.rounds == 3
    
    def test_gym_env_reset(self):
        """Gym environment must reset correctly."""
        from surface_code_in_stem.rl_control.gym_env import ColourCodeGymEnv
        env = ColourCodeGymEnv(distance=3, rounds=3, physical_error_rate=0.001)
        
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert len(obs) == env.num_detectors
        assert "binary_syndrome" in info
    
    def test_gym_env_step(self):
        """Gym environment must step correctly."""
        from surface_code_in_stem.rl_control.gym_env import ColourCodeGymEnv
        env = ColourCodeGymEnv(distance=3, rounds=3, physical_error_rate=0.001)
        
        obs, info = env.reset(seed=42)
        
        # Take action
        action = np.zeros(env.num_observables, dtype=np.int8)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert terminated is True  # Episode ends after one step
        assert reward in [1.0, -1.0]
    
    def test_discovery_env(self):
        """Discovery environment must work."""
        from surface_code_in_stem.rl_control.gym_env import ColourCodeDiscoveryEnv
        env = ColourCodeDiscoveryEnv(max_distance=13, target_threshold=0.005)
        
        obs, info = env.reset(seed=42)
        
        # Take a few actions
        for _ in range(5):
            action = 0  # inc_distance
            obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert step_info["distance"] > 3  # Distance should have increased
    
    def test_calibration_env(self):
        """Calibration environment must work."""
        from surface_code_in_stem.rl_control.gym_env import ColourCodeCalibrationEnv
        env = ColourCodeCalibrationEnv(distance=5, rounds=5, base_error_rate=0.001)
        
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert len(obs) == 3  # [mean_defect_rate, cycle_time_proxy, param_drift]


class TestParallelColourCodeEstimator:
    """Tests for ParallelColorCodeEstimator."""
    
    def test_estimator_imports(self):
        """Estimator must be importable."""
        from syndrome_net.parallel import ParallelColorCodeEstimator, ParallelConfig
        config = ParallelConfig(n_workers=2, use_jax=False)
        estimator = ParallelColorCodeEstimator(config)
        assert estimator is not None
    
    def test_circuit_cache(self):
        """Circuit cache must work."""
        from syndrome_net.parallel import CircuitCache
        from syndrome_net import CircuitSpec
        
        cache = CircuitCache(maxsize=10)
        
        spec = CircuitSpec(distance=5, rounds=5, error_probability=0.001, circuit_type="tri")
        
        # Initially empty
        assert cache.get(spec) is None
        
        # Add to cache
        noise = CCNoiseModel.uniform_circuit_noise(0.001)
        cc = ColorCode(d=5, rounds=5, circuit_type="tri", noise_model=noise)
        cache.put(spec, cc)
        
        # Should be retrievable
        cached = cache.get(spec)
        assert cached is not None
        assert len(cache) == 1
    
    def test_cache_lru_eviction(self):
        """Cache must evict LRU items."""
        from syndrome_net.parallel import CircuitCache
        from syndrome_net import CircuitSpec
        
        cache = CircuitCache(maxsize=2)
        
        specs = [
            CircuitSpec(distance=3, rounds=3, error_probability=0.001, circuit_type="tri"),
            CircuitSpec(distance=5, rounds=5, error_probability=0.001, circuit_type="tri"),
            CircuitSpec(distance=7, rounds=7, error_probability=0.001, circuit_type="tri"),
        ]
        
        noise = CCNoiseModel.uniform_circuit_noise(0.001)
        
        # Add three items to cache of size 2
        for i, spec in enumerate(specs):
            cc = ColorCode(d=spec.distance, rounds=spec.rounds, circuit_type="tri", noise_model=noise)
            cache.put(spec, cc)
        
        # First item should be evicted
        assert cache.get(specs[0]) is None
        # Last two should still be there
        assert cache.get(specs[1]) is not None
        assert cache.get(specs[2]) is not None


class TestContainerRegistration:
    """Tests for DI container registration."""
    
    def test_color_code_builder_registered(self):
        """ColorCodeStimBuilder must be registered in container."""
        reset_container()
        container = get_container()
        
        try:
            from syndrome_net.codes import ColorCodeStimBuilder
            builder = container.circuit_builders.get("color_code")
            assert builder is not None
        except ImportError:
            pytest.skip("ColorCodeStimBuilder not available")
    
    def test_loom_builder_registered(self):
        """LoomColorCodeBuilder must be registered in container."""
        reset_container()
        container = get_container()
        
        try:
            from syndrome_net.codes import LoomColorCodeBuilder
            builder = container.circuit_builders.get("loom_color_code")
            assert builder is not None
        except ImportError:
            pytest.skip("LoomColorCodeBuilder not available")
    
    def test_concat_decoder_registered(self):
        """ConcatenatedMWPMDecoder must be registered in container."""
        reset_container()
        container = get_container()
        
        try:
            from syndrome_net.decoders import ConcatenatedMWPMDecoder
            decoder = container.decoders.get("concat_mwpm")
            assert decoder is not None
        except (ImportError, KeyError):
            pytest.skip("ConcatenatedMWPMDecoder not available")
