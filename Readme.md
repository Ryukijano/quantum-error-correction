# Syndrome-Net: Quantum Error Correction + RL Control

Syndrome-Net is a research-oriented framework for building, simulating, decoding, and optimizing quantum error-correction (QEC) workflows with Stim. It combines:

- Surface-code and dynamic-code circuit generation
- **Colour code support** (triangular, rectangular, growing, Loom-based hexagonal)
- Plugin-based support for multiple code families (`surface`, `qldpc`, `bosonic`, `dual_rail_erasure`, `color_code`)
- Classical and confidence-aware decoders
- Reinforcement-learning environments and agents for:
  - one-shot decoding (`QECGymEnv`, `ColourCodeGymEnv`)
  - continuous calibration (`QECContinuousControlEnv`, `ColourCodeCalibrationEnv`)
  - code discovery (`QECCodeDiscoveryEnv`, `ColourCodeDiscoveryEnv`)

## What’s New

- **Colour code integration** (Lee & Brown 2025, Entropica Loom):
  - `ColorCodeStimBuilder`: triangular, rectangular, growing circuits via color-code-stim
  - `LoomColorCodeBuilder`: hexagonal lattice circuits via el-loom Eka/Block abstractions
  - `ConcatenatedMWPMDecoder`: 6-matching decoder (2 per colour R/G/B)
  - Colour code RL environments (`ColourCodeGymEnv`, `ColourCodeDiscoveryEnv`, `ColourCodeCalibrationEnv`)
  - Parallel threshold estimation with circuit caching (`ParallelColorCodeEstimator`)
  - Hexagonal lattice visualization in Streamlit UI
- Gym-compatible QEC environments in `surface_code_in_stem/rl_control/gym_env.py`
- SOTA RL agent implementations in `surface_code_in_stem/rl_control/sota_agents.py`
  - Transformer/TITANS-backed PPO for discrete decoding
  - Continuous SAC for calibration control
- End-to-end training script in `scripts/train_sota_rl.py`
- Expanded code-family support:
  - Bosonic variants (`gkp_surface`, `cat_code`, `squeezed_state`)
  - qLDPC parity-matrix builders (toric, surface-derived, hypergraph-product, custom parity)

## Demo Screenshots

These screenshots were captured from the live Streamlit QEC dashboard and are
checked into the repo so GitHub renders them directly:

| Circuit Viewer | Threshold Explorer |
|---|---|
| ![Circuit Viewer demo](assets/demo/circuit_viewer.png) | ![Threshold Explorer demo](assets/demo/threshold_explorer.png) |

The demo app includes:

- Stim circuit visualization with static SVG and interactive Crumble views
- Detector error graph + syndrome heatmap
- Threshold sweeps with Plotly curves
- Live RL training controls for PPO / SAC runs
- **🎨 Colour Codes tab**: hexagonal lattice visualization, circuit generation, and RL training

## Repository Layout

- `surface_code_in_stem/`: core circuit builders, decoders, RL control, noise models
- `syndrome_net/`: protocol definitions, circuit builders (including colour codes), decoders, parallel utilities
- `app/`: Streamlit application (circuit viewer, RL training, threshold explorer, colour codes)
- `codes/`: plugin architecture and benchmarking harness across code families
- `scripts/`: runnable workflows (including SOTA RL training)
- `tests/`: targeted tests for gym envs, bosonic variants, qLDPC parity, and colour codes
- `surface_code_in_stem/DYNAMIC_CODES.md`: implementation notes mapped to Morvan et al. 2025

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For full colour code support:

```bash
pip install -r requirements-qec.txt
```

This installs `color-code-stim`, `el-loom`, and `galois` for colour code circuits and concatenated MWPM decoding.

Minimal install for RL/Gym experiments:

```bash
pip install stim gym numpy torch pytest
```

## Quickstart

### 1) Generate a surface-code circuit

```python
from surface_code_in_stem.surface_code import surface_code_circuit_string

circuit = surface_code_circuit_string(distance=3, rounds=3, p=0.001)
print(circuit[:500])
```

### 2) Run the new Gym environments

```python
from surface_code_in_stem.rl_control.gym_env import QECGymEnv, QECContinuousControlEnv
from surface_code_in_stem.rl_control.gym_env import ColourCodeGymEnv

decoding_env = QECGymEnv(distance=3, rounds=3, physical_error_rate=0.005)
obs, info = decoding_env.reset(seed=7)

control_env = QECContinuousControlEnv(distance=3, rounds=3, parameter_dim=4)
obs_ctrl, _ = control_env.reset(seed=7)

# Colour code decoding environment
colour_env = ColourCodeGymEnv(distance=5, rounds=5, physical_error_rate=0.001)
obs_colour, info_colour = colour_env.reset(seed=7)
```

### 2b) Build colour code circuits

```python
from syndrome_net import CircuitSpec
from syndrome_net.codes import ColorCodeStimBuilder, LoomColorCodeBuilder

# Triangular colour code via color-code-stim
spec = CircuitSpec(distance=5, rounds=5, error_probability=0.001, circuit_type="tri")
builder = ColorCodeStimBuilder()
circuit = builder.build(spec)

# Hexagonal colour code via el-loom
spec_loom = CircuitSpec(distance=5, rounds=5, error_probability=0.001)
loom_builder = LoomColorCodeBuilder()
circuit_loom = loom_builder.build(spec_loom)
```

### 3) Train SOTA RL agents

```bash
# Decoder (Transformer/TITANS + PPO)
python3 scripts/train_sota_rl.py --mode ppo --episodes 512

# Calibration (Continuous SAC)
python3 scripts/train_sota_rl.py --mode sac --episodes 512

# Both
python3 scripts/train_sota_rl.py --mode all --episodes 1000
```

### 4) Run tests

```bash
python3 -m pytest tests/test_gym_env.py
python3 -m pytest tests/test_bosonic.py
python3 -m pytest tests/test_qldpc_parity.py
python3 -m pytest tests/test_colour_codes.py  # Colour code tests
python3 -m pytest tests/test_circuit_determinism.py  # Includes colour code determinism
```

## High-Level Architecture

```mermaid
flowchart LR
    U[User / Script / Notebook] --> A[Code Family Plugins]
    U --> B[RL Gym Environments]

    subgraph Plugins
        A1[surface plugin]
        A2[qldpc plugin]
        A3[bosonic plugin]
        A4[dual_rail_erasure plugin]
        A5[color_code plugin]
    end

    A --> A1
    A --> A2
    A --> A3
    A --> A4
    A --> A5

    A1 --> C[Stim Circuit String]
    A2 --> C
    A3 --> C
    A4 --> C
    A5 --> C

    C --> D[stim.Circuit + sampler]
    D --> E[Syndromes + Observables]

    subgraph RL
        B1[QECGymEnv<br/>one-shot decoding]
        B2[QECContinuousControlEnv<br/>calibration control]
        B3[ColourCodeGymEnv<br/>colour code decoding]
        B4[ColourCodeDiscoveryEnv<br/>code discovery]
        B5[ColourCodeCalibrationEnv<br/>calibration]
        F1[Transformer/TITANS PPO]
        F2[Continuous SAC]
    end

    B --> B1
    B --> B2
    B --> B3
    B --> B4
    B --> B5
    E --> B1
    E --> B2
    E --> B3
    E --> B4
    E --> B5
    B1 --> F1
    B2 --> F2
    B3 --> F1
    B4 --> F1
    B5 --> F2

    F1 --> G[Decode action / logical prediction]
    F2 --> H[Continuous control action / theta update]

    G --> I[Reward / metrics]
    H --> I
```

## Documentation

- `docs/README.md`: docs index and reading path
- `docs/RL_QEC_ARCHITECTURE.md`: detailed architecture and algorithm internals with Mermaid diagrams
- `RL_EXPERIMENTS.md`: reproducibility notes
- `surface_code_in_stem/DYNAMIC_CODES.md`: dynamic-code implementation notes

## Colour Code References

The colour code implementation is based on:

- **Lee & Brown (2025)**: "High-threshold and low-overhead fault-tolerant quantum memory", [arXiv:2503.09704](https://arxiv.org/abs/2503.09704) — Breakthrough colour code architecture with concatenated MWPM decoding
- **Entropica Loom (2024)**: "Loom: A quantum error correction lattice surgery tool", [arXiv:2404.08663](https://arxiv.org/abs/2404.08663) — Eka/Block/Lattice abstractions for QEC circuit design
- **color-code-stim**: Python package for simulating and decoding colour code circuits using Stim — [GitHub](https://github.com/seokhyung-lee/color-code-stim)

The implementation includes:
- Two circuit builders: `ColorCodeStimBuilder` (color-code-stim) and `LoomColorCodeBuilder` (el-loom)
- Concatenated MWPM decoder with 6 matchings (2 per colour R/G/B)
- RL environments for decoding, discovery, and calibration
- Parallel threshold estimation with circuit caching
- Hexagonal lattice visualization in Streamlit UI

## Notes

- Current RL environments are implemented with `gym`; if desired, migration to `gymnasium` is straightforward.
- `QECGymEnv` is currently a one-step episode formulation for decoding, which is ideal for policy learning over syndrome-to-logical mapping and baseline comparison against MWPM.
