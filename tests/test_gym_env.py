"""Tests for the OpenAI Gym QEC environment."""

import pytest
import numpy as np

pytest.importorskip("gym")
stim = pytest.importorskip("stim")

from surface_code_in_stem.rl_control.gym_env import QECGymEnv, QECContinuousControlEnv


def test_qec_gym_env_initialization():
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01)
    
    assert env.observation_space.shape[0] > 0
    assert env.action_space.shape[0] > 0
    
    # Check that it resets correctly
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert isinstance(obs, np.ndarray)
    assert "mwpm_prediction" in info


def test_qec_gym_env_step():
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01)
    env.reset(seed=42)
    
    # Take an arbitrary action (predict no logical errors)
    action = np.zeros(env.action_space.shape[0])
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert "actual_logical" in info
    assert "is_correct" in info


def test_qec_continuous_control_env():
    env = QECContinuousControlEnv(distance=3, rounds=2, parameter_dim=2, batch_shots=10)
    
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    
    action = np.array([0.01, -0.01])
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert reward <= 0.0  # Reward is negative logical error rate
    assert "logical_error_rate" in info
    assert "effective_p" in info
