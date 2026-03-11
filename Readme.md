# Quantum Error Correction with Stim

This repository contains implementations and simulations of quantum error correction codes using Stim, including:

- Repetition codes
- Surface codes
- Error threshold analysis

## Structure
- `introduction_to_stim/`: Lab exercises and tutorials for Stim
- `surface_code_in_stem/`: Surface code implementation and simulations
- `codes/`: Code-family plugin system with shared interfaces and a benchmark harness
  to evaluate multiple families through the same decoder API.
  - `codes/surface/`: Surface-code plugin + compatibility wrappers around existing builders
  - `codes/qldpc/`: qLDPC placeholder plugin with explicit parity-check schema
  - `codes/bosonic/`: Bosonic placeholder plugin scaffold
  - `codes/dual_rail_erasure/`: Dual-rail erasure placeholder plugin with required parity inputs
- `surface_code_in_stem/dynamic/`: Stim circuit builders for the hexagonal,
  walking, and iSWAP dynamic surface codes demonstrated in Morvan et al.
  (Nature Physics, 2025). See `surface_code_in_stem/DYNAMIC_CODES.md` for a
  mapping from the paper to the implementation choices here. The legacy
  `dynamic_surface_codes.py` re-exports the same helpers for convenience.
- [getting_started.ipynb](cci:7://file:///home/ryukijano/quantum_error_correction/getting_started.ipynb:0:0-0:0): Introduction to Stim notebook

## Requirements

- Python 3.10+

## Installation

```bash
pip install stim numpy matplotlib
```

## Quickstart

### 1) Build a static surface-code circuit

```python
from surface_code_in_stem.surface_code import surface_code_circuit_string

stim_circuit = surface_code_circuit_string(distance=3, rounds=3, p=0.001)
print(stim_circuit[:400])
```

If Stim is missing, importing `rl_nested_learning` will defer the error until a Stim-powered feature is used, while providing a
clear installation hint so the module remains importable in lightweight environments.

## Project flow diagram

```mermaid
flowchart TD
    %% =========================
    %% Entry Points
    %% =========================
    A[User / Notebook / Script] --> B1[Static circuit workflow]
    A --> B2[Dynamic circuit workflow]
    A --> B3[RL comparison workflow]

    %% =========================
    %% Static Surface Code
    %% =========================
    subgraph S1[Static Surface Code Path]
        B1 --> C1[surface_code_in_stem.surface_code.surface_code_circuit_string]
        C1 --> C2[Geometry helpers<br/>data_coords / x_measure_coords / z_measure_coords]
        C1 --> C3[Cycle assembly<br/>initialization_step + rounds_step + final_step]
        C3 --> C4[Stim circuit string]
    end

    %% =========================
    %% Dynamic Surface Codes
    %% =========================
    subgraph S2[Dynamic Surface Code Path]
        B2 --> D0[surface_code_in_stem.dynamic.__init__]
        D0 --> D1[hexagonal_surface_code]
        D0 --> D2[walking_surface_code]
        D0 --> D3[iswap_surface_code]

        D1 --> DB[dynamic.base<br/>DynamicLayout + StimStringBuilder + stabilizer_cycle]
        D2 --> DB
        D3 --> DB

        DB --> D4[prepare_coords + adjacent_coords<br/>imported from static module]
        DB --> D5[Stim circuit string with<br/>QUBIT_COORDS / DETECTOR / OBSERVABLE_INCLUDE]
    end

    %% =========================
    %% RL / Comparison
    %% =========================
    subgraph S3[Policy Comparison / Evaluation]
        B3 --> E1[surface_code_in_stem.rl_nested_learning.compare_nested_policies]
        E1 --> E2[Build static circuit<br/>surface_code_circuit_string]
        E1 --> E3[Build dynamic circuit<br/>default: hexagonal_surface_code]
        E2 --> E4[_logical_error_rate]
        E3 --> E4
        E4 --> E5[stim.Circuit + detector sampler]
        E5 --> E6[logical_error_rate metrics]
        E6 --> E7[tabulate_comparison]
    end

    %% =========================
    %% Top-level compatibility modules
    %% =========================
    F1[rl_nested_learning.py at repo root] --> F2[Lazy optional stim import helper]
    F3[surface_code_in_stem/dynamic_surface_codes.py] --> D0
    F4[surface_code_in_stem/__init__.py] --> C1
    F4 --> D0
    F4 --> E1
```
