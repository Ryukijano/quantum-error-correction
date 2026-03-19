"""Integration tests for the benchmark/training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("stim")
pytest.importorskip("gym")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codes import benchmark_code_families
from scripts.train_sota_rl import train_ppo_decoder
from surface_code_in_stem.rl_control.gym_env import QECGymEnv


def test_benchmark_plus_env_plus_training_history(benchmark_config):
    # 1) Shared benchmark harness should run surface plugin with deterministic metadata.
    results = benchmark_code_families(["surface"], benchmark_config, shots=8, seed=17)
    assert "surface" in results
    assert 0.0 <= results["surface"]["logical_error_rate"] <= 1.0

    # 2) Env path should support soft-information observations.
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01, use_soft_information=True)
    obs, info = env.reset(seed=17)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    assert "binary_syndrome" in info

    # 3) PPO helper should return per-episode history for downstream plotting.
    history = train_ppo_decoder(distance=3, rounds=2, physical_error_rate=0.01, episodes=4, batch_size=2)
    assert len(history) == 4
    assert all("episode" in row and "reward" in row for row in history)
