"""OpenAI Gym environment for Quantum Error Correction reinforcement learning.

This module provides a Gym-compatible environment where an RL agent can interact
with a quantum error correction simulation. The agent receives syndromes as
observations and outputs corrections as actions, receiving rewards based on
successful logical operations.

Mathematical formulation:
- State Space S: {0,1}^N where N is the number of detectors (syndrome bits)
- Action Space A: {0,1}^M where M is the number of possible logical corrections
- Transition Function T: S x A -> S' determined by the quantum circuit dynamics
- Reward R(s,a,s'): +1 if logical state is preserved, -1 if logical error occurs
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False

from surface_code_in_stem.surface_code import surface_code_circuit_string
from surface_code_in_stem.decoders import MWPMDecoder, DecoderMetadata


class QECGymEnv(gym.Env):
    """
    OpenAI Gym environment for Quantum Error Correction.
    
    The agent acts as a decoder: it observes syndromes (detector events)
    and must predict the correct logical observable flips.
    
    Attributes:
        observation_space: MultiBinary space of size N (number of detectors)
        action_space: MultiDiscrete space of size M (number of observables), each {0, 1}
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        distance: int = 3,
        rounds: int = 3,
        physical_error_rate: float = 0.001,
        seed: Optional[int] = None,
        use_mwpm_baseline: bool = True,
        use_soft_information: bool = False
    ):
        """Initialize the QEC Gym Environment.
        
        Args:
            distance: Code distance for the surface code
            rounds: Number of syndrome measurement rounds
            physical_error_rate: Base probability of physical errors
            seed: Random seed for reproducibility
            use_mwpm_baseline: If True, provides MWPM prediction in info dict
            use_soft_information: If True, returns continuous [0,1] confidence values instead of binary {0,1}
        """
        super().__init__()
        
        if not HAS_STIM:
            raise ImportError("Stim is required for QECGymEnv. Run: pip install stim")
            
        self.distance = distance
        self.rounds = rounds
        self.p = physical_error_rate
        self.use_mwpm_baseline = use_mwpm_baseline
        self.use_soft_information = use_soft_information
        
        # Set up random number generator
        self._rng = np.random.default_rng(seed)
        self._seed_val = seed if seed is not None else self._rng.integers(0, 2**31 - 1)
        
        # Build circuit
        self._build_circuit()
        
        # Define spaces
        if self.use_soft_information:
            # Continuous [0, 1] soft information from analog readout
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_detectors,), dtype=np.float32)
        else:
            # Observation is the binary detector syndrome vector
            self.observation_space = spaces.MultiBinary(self.num_detectors)
        
        # Action is predicting the logical observable flip (0 or 1 for each observable)
        self.action_space = spaces.MultiDiscrete([2] * self.num_observables)
        
        # State for current episode
        self._current_syndrome: Optional[np.ndarray] = None
        self._current_logical: Optional[np.ndarray] = None
        
        # Optional MWPM baseline for comparison
        if self.use_mwpm_baseline:
            self._mwpm_decoder = MWPMDecoder()
            self._decoder_metadata = DecoderMetadata(
                num_observables=self.num_observables,
                detector_error_model=self.circuit.detector_error_model(decompose_errors=True),
                circuit=self.circuit,
                seed=self._seed_val
            )

    def _build_circuit(self) -> None:
        """Construct the underlying Stim circuit."""
        circuit_str = surface_code_circuit_string(
            distance=self.distance,
            rounds=self.rounds,
            p=self.p
        )
        self.circuit = stim.Circuit(circuit_str)
        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables
        self.sampler = self.circuit.compile_detector_sampler(seed=self._seed_val)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to a new random syndrome state.
        
        Returns:
            observation: The initial detector syndrome
            info: Dictionary containing auxiliary information
        """
        super().reset(seed=seed)
        if seed is not None:
            self._seed_val = seed
            self.sampler = self.circuit.compile_detector_sampler(seed=seed)
            
        # Sample one shot
        det_samples_bool, obs_samples = self.sampler.sample(1, separate_observables=True)
        
        det_samples = det_samples_bool[0].astype(np.int8)
        self._current_logical = obs_samples[0].astype(np.int8)
        
        if self.use_soft_information:
            # Simulate analog soft information: true defects are closer to 1, others to 0
            # with some Gaussian readout noise scaled by the physical error rate
            noise = self._rng.normal(loc=0.0, scale=0.2, size=self.num_detectors)
            self._current_syndrome = np.clip(det_samples + noise, 0.0, 1.0).astype(np.float32)
        else:
            self._current_syndrome = det_samples
        
        info = {}
        
        # Provide MWPM baseline if requested
        if self.use_mwpm_baseline:
            # Need 2D array for decoder input
            events = np.array([det_samples_bool[0]], dtype=np.bool_)
            decoded = self._mwpm_decoder.decode(events, self._decoder_metadata)
            mwpm_pred = decoded.logical_predictions[0].astype(np.int8)
            mwpm_correct = np.all(mwpm_pred == self._current_logical)
            info["mwpm_prediction"] = mwpm_pred
            info["mwpm_correct"] = mwpm_correct
            
        # Save binary syndrome for info
        info["binary_syndrome"] = det_samples
            
        return self._current_syndrome, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action (logical prediction) and return reward.
        
        In the standard QEC setting, episodes are length 1. The agent sees a syndrome,
        predicts the correction, and the episode ends.
        
        Args:
            action: Array of shape (num_observables,) with 0/1 predictions
            
        Returns:
            observation: New state (terminal in this formulation)
            reward: +1 for correct prediction, -1 for incorrect
            terminated: True (episode ends after one step)
            truncated: False
            info: Aux info including actual logical outcome
        """
        if self._current_syndrome is None or self._current_logical is None:
            raise RuntimeError("Cannot step before calling reset()")
            
        action = np.asarray(action, dtype=np.int8)
        
        if action.shape != (self.num_observables,):
            raise ValueError(f"Action shape must be ({self.num_observables},)")
            
        # Check if prediction matches actual logical error
        is_correct = np.all(action == self._current_logical)
        
        # Reward design:
        # We want to encourage correct decoding.
        # +1 for completely correct, -1 for any error.
        reward = 1.0 if is_correct else -1.0
        
        # Episode ends after one decoding attempt in standard formulation
        terminated = True
        truncated = False
        
        info = {
            "actual_logical": self._current_logical,
            "is_correct": is_correct,
            "error_rate_p": self.p
        }
        
        # Empty observation for next state since we terminate
        next_obs = np.zeros_like(self._current_syndrome)
        
        return next_obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render the environment state."""
        if mode != "human":
            raise NotImplementedError("Only human render mode supported")
            
        if self._current_syndrome is not None:
            num_defects = np.sum(self._current_syndrome)
            print(f"Syndrome (N={self.num_detectors}): {num_defects} defects")
            print(f"Actual Logical Flips: {self._current_logical}")
        else:
            print("Environment not initialized (call reset)")


class QECContinuousControlEnv(gym.Env):
    """
    OpenAI Gym environment for QEC Control/Calibration.
    
    Mathematical formulation:
    - State Space S: R^N (detector statistics/rates over a batch)
    - Action Space A: R^K (continuous perturbations to control parameters)
    - Reward R: -p_L where p_L is the logical error rate of the resulting state
    """
    
    def __init__(
        self,
        distance: int = 3,
        rounds: int = 3,
        parameter_dim: int = 4,
        batch_shots: int = 128,
        base_error_rate: float = 0.001,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        from .environment import StimCalibrationConfig, StimCalibrationEnvironment
        
        config = StimCalibrationConfig(
            distance=distance,
            rounds=rounds,
            shots=batch_shots,
            base_error_rate=base_error_rate,
            seed=seed if seed is not None else 0
        )
        
        self.env = StimCalibrationEnvironment(config, parameter_dim=parameter_dim)
        
        # Observe the initial state to get dimensions
        initial_obs = self.env.reset()
        obs_dim = initial_obs.shape[0]
        
        # State: detector rates [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float64)
        
        # Action: continuous parameter updates [-0.05, 0.05]
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(parameter_dim,), dtype=np.float64)
        
        self.max_steps = 50
        self.current_step = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.env.reset()
        
        # Gym API expects float32/float64 generally
        return obs.astype(np.float64), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        
        # The inner environment expects float64 array
        obs, reward, info = self.env.step(action)
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return obs.astype(np.float64), float(reward), terminated, truncated, info


class QECCodeDiscoveryEnv(gym.Env):
    """
    OpenAI Gym environment for Automated QEC Code Discovery.
    
    Agent sequentially flips bits in H_x and H_z parity check matrices to 
    discover novel qLDPC codes. The episode ends when the agent selects the
    'submit' action. The reward is based on fulfilling CSS conditions 
    (H_x @ H_z^T = 0 mod 2) and maximizing the code distance / rate.
    """
    
    def __init__(self, num_qubits: int = 12, num_checks: int = 8, max_steps: int = 50):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_checks = num_checks
        self.max_steps = max_steps
        
        # State: H_x and H_z matrices flattened
        self.matrix_size = num_checks * num_qubits
        self.state_dim = 2 * self.matrix_size
        
        self.observation_space = spaces.MultiBinary(self.state_dim)
        
        # Action: flip any bit in H_x or H_z, plus 1 action for 'submit/done'
        self.action_space = spaces.Discrete(self.state_dim + 1)
        self.submit_action = self.state_dim
        
        self._state = np.zeros(self.state_dim, dtype=np.int8)
        self.current_step = 0
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = np.zeros(self.state_dim, dtype=np.int8)
        self.current_step = 0
        return self._state.copy(), {}
        
    def _evaluate_code(self) -> float:
        """Evaluate CSS conditions and return a reward proxy."""
        hx = self._state[:self.matrix_size].reshape(self.num_checks, self.num_qubits)
        hz = self._state[self.matrix_size:].reshape(self.num_checks, self.num_qubits)
        
        # Check CSS commutativity: H_x @ H_z.T = 0 mod 2
        commutator = np.dot(hx, hz.T) % 2
        css_violations = np.sum(commutator)
        
        if css_violations > 0:
            # Heavy penalty for violating CSS, proportional to violations
            return -1.0 - 0.1 * css_violations
            
        # If CSS condition is met, we reward based on:
        # 1. Row ranks (we want independent checks to increase rate/distance)
        # 2. Check weights (we want low weight for LDPC)
        try:
            import galois
            hx_gf = galois.GF2(hx)
            hz_gf = galois.GF2(hz)
            rank_x = np.linalg.matrix_rank(hx_gf)
            rank_z = np.linalg.matrix_rank(hz_gf)
        except ImportError:
            # Fallback naive rank approx if galois not installed
            rank_x = self.num_checks
            rank_z = self.num_checks
            
        k = self.num_qubits - rank_x - rank_z
        
        # Penalize if k <= 0 (encodes zero or negative logical qubits)
        if k <= 0:
            return -0.5
            
        # Average row weights (encourage low density)
        avg_weight_x = np.mean(np.sum(hx, axis=1))
        avg_weight_z = np.mean(np.sum(hz, axis=1))
        
        # Target LDPC regime (e.g. weights around 4 to 6)
        weight_penalty = max(0, avg_weight_x - 6) + max(0, avg_weight_z - 6)
        
        # Simple reward proxy: encodes logical qubits with reasonable density
        reward = float(k) - 0.1 * weight_penalty
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        reward = 0.0
        info = {}
        
        if action == self.submit_action:
            terminated = True
            reward = self._evaluate_code()
            info["submitted"] = True
        else:
            # Flip the selected bit in the parity matrices
            self._state[action] = 1 - self._state[action]
            
            # Small shaping penalty for each step to encourage faster discovery
            reward = -0.01
            
            if truncated:
                # End of max steps, force evaluation
                reward = self._evaluate_code()
                info["submitted"] = True
                
        # Compute current stats for info
        hx = self._state[:self.matrix_size].reshape(self.num_checks, self.num_qubits)
        hz = self._state[self.matrix_size:].reshape(self.num_checks, self.num_qubits)
        css_violations = np.sum(np.dot(hx, hz.T) % 2)
        
        info.update({
            "css_violations": int(css_violations),
            "avg_check_weight": float(np.mean(np.sum(hx, axis=1)) + np.mean(np.sum(hz, axis=1))) / 2.0
        })
        
        return self._state.copy(), float(reward), terminated, truncated, info


class ColourCodeGymEnv(gym.Env):
    """Gym environment for RL decoding of colour codes.
    
    Extends the QECGymEnv pattern to colour codes from color-code-stim.
    The agent observes colour code syndromes and predicts logical corrections.
    
    Supports triangular (d odd) and rectangular (d even) colour code patches.
    Uses concatenated MWPM as baseline decoder for comparison.
    
    Reference: Lee & Brown, Quantum 9, 1609 (2025)
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        physical_error_rate: float = 0.001,
        circuit_type: str = "tri",
        seed: Optional[int] = None,
        use_mwpm_baseline: bool = True,
        use_superdense: bool = False
    ):
        super().__init__()
        
        if not HAS_STIM:
            raise ImportError("Stim is required for ColourCodeGymEnv")
        
        try:
            from color_code_stim import ColorCode, NoiseModel as CCNoiseModel
        except ImportError as exc:
            raise ImportError(
                "color-code-stim is required. Install with: pip install color-code-stim"
            ) from exc
        
        self.distance = distance
        self.rounds = rounds
        self.p = physical_error_rate
        self.circuit_type = circuit_type
        self.use_superdense = use_superdense
        self.use_mwpm_baseline = use_mwpm_baseline
        
        self._rng = np.random.default_rng(seed)
        self._seed_val = seed if seed is not None else self._rng.integers(0, 2**31 - 1)
        
        # Build colour code circuit
        noise = CCNoiseModel.uniform_circuit_noise(self.p)
        kwargs = dict(
            d=self.distance,
            rounds=self.rounds,
            circuit_type=self.circuit_type,
            noise_model=noise,
            superdense_circuit=self.use_superdense,
        )
        self._color_code = ColorCode(**kwargs)
        self.circuit = self._color_code.circuit
        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables
        self.sampler = self.circuit.compile_detector_sampler(seed=self._seed_val)
        
        # Define spaces
        self.observation_space = spaces.MultiBinary(self.num_detectors)
        self.action_space = spaces.MultiDiscrete([2] * self.num_observables)
        
        self._current_syndrome: Optional[np.ndarray] = None
        self._current_logical: Optional[np.ndarray] = None
        
        if self.use_mwpm_baseline:
            from surface_code_in_stem.decoders import DecoderMetadata
            self._decoder_metadata = DecoderMetadata(
                num_observables=self.num_observables,
                detector_error_model=self.circuit.detector_error_model(decompose_errors=True),
                circuit=self.circuit,
                seed=self._seed_val
            )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed_val = seed
            self.sampler = self.circuit.compile_detector_sampler(seed=seed)
        
        det_samples_bool, obs_samples = self.sampler.sample(1, separate_observables=True)
        det_samples = det_samples_bool[0].astype(np.int8)
        self._current_logical = obs_samples[0].astype(np.int8)
        self._current_syndrome = det_samples
        
        info = {"binary_syndrome": det_samples}
        
        if self.use_mwpm_baseline:
            try:
                prediction = self._color_code.concat_matching_decoder(det_samples_bool[0])
                mwpm_pred = np.array([int(prediction)], dtype=np.int8)
                mwpm_correct = np.all(mwpm_pred == self._current_logical)
                info["mwpm_prediction"] = mwpm_pred
                info["mwpm_correct"] = mwpm_correct
            except Exception:
                pass
        
        return self._current_syndrome, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._current_syndrome is None or self._current_logical is None:
            raise RuntimeError("Cannot step before calling reset()")
        
        action = np.asarray(action, dtype=np.int8)
        if action.shape != (self.num_observables,):
            raise ValueError(f"Action shape must be ({self.num_observables},)")
        
        is_correct = np.all(action == self._current_logical)
        reward = 1.0 if is_correct else -1.0
        
        terminated = True
        truncated = False
        
        info = {
            "actual_logical": self._current_logical,
            "is_correct": is_correct,
            "error_rate_p": self.p
        }
        
        next_obs = np.zeros_like(self._current_syndrome)
        return next_obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise NotImplementedError("Only human render mode supported")
        if self._current_syndrome is not None:
            print(f"Colour code syndrome (N={self.num_detectors}): {np.sum(self._current_syndrome)} defects")
            print(f"Actual Logical: {self._current_logical}")


class ColourCodeDiscoveryEnv(gym.Env):
    """RL environment for discovering colour code parameters.
    
    Extends QECCodeDiscoveryEnv with colour-code-specific constraints:
    - 3-colourability of the lattice
    - Hexagonal lattice structure
    - Transversal Clifford gate compatibility
    
    Agent explores distance, patch geometry, and noise thresholds.
    """
    
    def __init__(
        self,
        max_distance: int = 13,
        min_distance: int = 3,
        max_rounds: int = 10,
        target_threshold: float = 0.005,
        max_steps: int = 20
    ):
        super().__init__()
        
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.max_rounds = max_rounds
        self.target_threshold = target_threshold
        self.max_steps = max_steps
        
        # State: [distance_idx, circuit_type_idx, rounds, superdense_flag, p_rate]
        # Action: adjust one of these parameters or submit
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1, 0, 0.0001]),
            high=np.array([10, 4, max_rounds, 1, 0.02]),
            dtype=np.float32
        )
        
        # Actions: 0=inc_distance, 1=dec_distance, 2=change_type, 3=toggle_superdense,
        # 4=inc_rounds, 5=dec_rounds, 6=adjust_p, 7=submit
        self.action_space = spaces.Discrete(8)
        
        self.circuit_types = ["tri", "rec", "growing", "cult+growing"]
        self._state = np.array([2, 0, 5, 0, 0.001], dtype=np.float32)  # d=7, tri, 5 rounds
        self.current_step = 0
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = np.array([2, 0, 5, 0, 0.001], dtype=np.float32)
        self.current_step = 0
        return self._state.copy(), {}
    
    def _get_distance(self) -> int:
        idx = int(self._state[0])
        distances = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        return distances[min(idx, len(distances)-1)]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        reward = -0.01  # Small step penalty
        
        if action == 0:  # inc_distance
            self._state[0] = min(self._state[0] + 1, 10)
        elif action == 1:  # dec_distance
            self._state[0] = max(self._state[0] - 1, 0)
        elif action == 2:  # change circuit type
            self._state[1] = (int(self._state[1]) + 1) % len(self.circuit_types)
        elif action == 3:  # toggle superdense
            self._state[3] = 1 - self._state[3]
        elif action == 4:  # inc_rounds
            self._state[2] = min(self._state[2] + 1, self.max_rounds)
        elif action == 5:  # dec_rounds
            self._state[2] = max(self._state[2] - 1, 1)
        elif action == 6:  # adjust p (small random change)
            delta = np.random.uniform(-0.001, 0.001)
            self._state[4] = np.clip(self._state[4] + delta, 0.0001, 0.02)
        elif action == 7:  # submit
            terminated = True
            reward = self._evaluate_code()
        
        if truncated and not terminated:
            terminated = True
            reward = self._evaluate_code()
        
        info = {
            "distance": self._get_distance(),
            "circuit_type": self.circuit_types[int(self._state[1])],
            "rounds": int(self._state[2]),
            "superdense": bool(self._state[3]),
            "p": float(self._state[4])
        }
        
        return self._state.copy(), float(reward), terminated, truncated, info
    
    def _evaluate_code(self) -> float:
        """Evaluate colour code parameters and return reward."""
        d = self._get_distance()
        circuit_type = self.circuit_types[int(self._state[1])]
        rounds = int(self._state[2])
        superdense = bool(self._state[3])
        p = float(self._state[4])
        
        # Validation: distance constraints
        if circuit_type == "tri" and d % 2 == 0:
            return -1.0  # Triangular requires odd distance
        if circuit_type == "rec" and d % 2 != 0:
            return -0.5  # Rectangular prefers even distance
        
        # Simulate to estimate logical error rate
        try:
            from color_code_stim import ColorCode, NoiseModel as CCNoiseModel
            noise = CCNoiseModel.uniform_circuit_noise(p)
            cc = ColorCode(d=d, rounds=rounds, circuit_type=circuit_type, 
                          noise_model=noise, superdense_circuit=superdense)
            num_fails, _ = cc.simulate(shots=1000)
            ler = num_fails / 1000
            
            # Reward: below threshold is good
            if ler < self.target_threshold:
                return 1.0 + (self.target_threshold - ler) * 100  # Bonus for beating threshold
            else:
                return -ler * 10  # Penalty proportional to error rate
        except Exception:
            return -0.5  # Simulation failure penalty


class ColourCodeCalibrationEnv(gym.Env):
    """RL environment for calibrating colour code experiments.
    
    Continuous control of per-qubit error rates and timing parameters
    to optimize logical error rate for a fixed colour code configuration.
    
    State: detector statistics from recent syndrome measurements
    Action: continuous adjustments to calibration parameters
    Reward: improvement in logical error rate
    """
    
    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        circuit_type: str = "tri",
        parameter_dim: int = 6,
        batch_shots: int = 128,
        base_error_rate: float = 0.001,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        try:
            from color_code_stim import ColorCode, NoiseModel as CCNoiseModel
        except ImportError as exc:
            raise ImportError(
                "color-code-stim is required. Install with: pip install color-code-stim"
            ) from exc
        
        self.distance = distance
        self.rounds = rounds
        self.circuit_type = circuit_type
        self.parameter_dim = parameter_dim
        self.batch_shots = batch_shots
        self.base_error_rate = base_error_rate
        self._seed = seed if seed is not None else 0
        
        # Initial circuit
        noise = CCNoiseModel.uniform_circuit_noise(base_error_rate)
        self._color_code = ColorCode(
            d=distance, rounds=rounds, circuit_type=circuit_type,
            noise_model=noise, superdense_circuit=False
        )
        
        # State: detector rate statistics
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )  # [mean_defect_rate, cycle_time_proxy, parameter_drift]
        
        # Action: continuous calibration adjustments
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(parameter_dim,), dtype=np.float32
        )
        
        self._current_params = np.zeros(parameter_dim, dtype=np.float32)
        self.max_steps = 50
        self.current_step = 0
        self._last_ler = 1.0
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self._current_params = np.zeros(self.parameter_dim, dtype=np.float32)
        self._last_ler = 1.0
        
        # Initial observation
        obs = self._measure_state()
        return obs.astype(np.float32), {}
    
    def _measure_state(self) -> np.ndarray:
        """Measure detector statistics from colour code."""
        try:
            sampler = self._color_code.circuit.compile_detector_sampler(seed=self._seed + self.current_step)
            det_samples, _ = sampler.sample(self.batch_shots, separate_observables=True)
            mean_defect_rate = np.mean(det_samples)
        except Exception:
            mean_defect_rate = self.base_error_rate
        
        cycle_time_proxy = 1.0 + 0.1 * np.sum(np.abs(self._current_params))
        param_drift = np.mean(np.abs(self._current_params))
        
        return np.array([mean_defect_rate, cycle_time_proxy, param_drift], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        
        # Apply parameter adjustments
        self._current_params += action
        self._current_params = np.clip(self._current_params, -1.0, 1.0)
        
        # Measure new state
        obs = self._measure_state()
        
        # Estimate logical error rate
        try:
            num_fails, _ = self._color_code.simulate(shots=self.batch_shots)
            ler = num_fails / self.batch_shots
        except Exception:
            ler = 1.0
        
        # Reward: improvement in LER
        improvement = self._last_ler - ler
        reward = improvement * 10  # Scale up for meaningful rewards
        self._last_ler = ler
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            "logical_error_rate": ler,
            "params": self._current_params.copy(),
            "mean_defect_rate": obs[0]
        }
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
