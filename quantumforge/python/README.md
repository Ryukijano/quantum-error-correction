# qhybrid-kernels: Python bindings for qhybrid

This package provides optional PyO3-backed Rust kernels behind the
`qhybrid` Python API used by Syndrome-Net's optional accelerated sampling
and noise stack.

## What this package provides

- Public API entry point: `qhybrid_kernels`
- Rust extension module name: `qhybrid_kernels.rust_kernels`
- Pure-Python fallback path when Rust extension is not available
- Conversions, Qiskit adapters, and high-throughput noise/kernel utilities

## Relationship with Syndrome-Net

Syndrome-Net consumes this package through:

- `surface_code_in_stem/accelerators/qhybrid_backend.py` (`qhybrid_backend`)
  - Imports `apply_pauli_channel_statevector` and `apply_kraus_1q_density_matrix`
  - Exposes `probe_capability()` and execution metadata used by runtime telemetry
- `surface_code_in_stem/accelerators/sampling_backends.py`
  - Chooses `qhybrid` via the sampling backend resolver
  - Uses runtime capability checks from `qhybrid_backend.probe_capability()`
- `surface_code_in_stem/rl_control/gym_env.py` + `app/rl_runner.py`
  - Controls sampler selection and auto-acceleration defaults

## Rust / PyO3 build prerequisites

Recommended local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r quantumforge/python/requirements.lock
cd quantumforge/python
pip install maturin
```

Prerequisites:

- Python 3.9+
- Rust toolchain (`rustc`, `cargo`)
- `maturin`
- `numpy` (runtime)

## Build the extension module

Editable build for local development:

```bash
cd quantumforge/python
maturin develop
```

Non-editable wheel build:

```bash
cd quantumforge/python
maturin build --release
```

The extension is imported from `qhybrid_kernels.rust_kernels`.

## How this package plugs into Syndrome-Net

Syndrome-Net discovers `qhybrid_kernels.rust_kernels` through a capability-first path:

1. `qhybrid_backend` probes module/runtime availability.
2. `surface_code_in_stem/accelerators/sampling_backends.py` builds the resolver chain (`qhybrid -> cuquantum -> qujax -> cudaq -> stim`).
3. RL runtime and benchmark paths record backend decision metadata:
   - `backend_id`, `backend_chain_tokens`, `backend_chain`
   - `contract_flags`, `profiler_flags`, `fallback_reason`

When the extension is absent or import fails, Syndrome-Net falls back to pure-Python paths while preserving backend telemetry.

## Merge playbook for `quantumforge` changes

When this module moves in parallel with a standalone `quantumforge` repository, prefer one of:

- snapshot mode: copy the updated tree into `syndrome-net/quantumforge` and sync lockfiles.
- package mode: publish a `qhybrid_kernels` wheel and consume via `requirements.*` in both repos.

Validation sequence:

1. Install with `maturin develop` in a clean venv.
2. Run `python -m pytest` for sampling/contract tests in the Syndrome-Net workspace.
3. Verify `backend_chain_tokens` shape in benchmark CSV/JSON outputs remains list-shaped in memory and JSON-encoded at serialization.

## Behaviour without the extension

If the Rust extension is not available, this package intentionally falls back.
The first import that requires kernels raises an explicit error:

```
rust_kernels extension is not available. Did you run `maturin develop` in quantumforge/python/?
```

And Syndrome-Net metadata consumers can still proceed via degraded/noisy-path fallbacks.

## Features

- **Fast Circuit Execution**: Replace slow Python statevector simulation paths
- **Advanced Noise Modeling**:
  - Pauli Channel (Monte Carlo trajectories)
  - Kraus Operator (Density Matrix)
  - Correlated Pauli Noise
  - Correlated CNOT errors
  - Pauli expectation evaluation
- **Qiskit Adapter**: Convert `qiskit.QuantumCircuit` objects to `qhybrid` JSON

## Example usage

```python
from qiskit import QuantumCircuit
from qhybrid_kernels import qiskit_to_qhybrid_json, execute_quantum_circuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

circuit_json = qiskit_to_qhybrid_json(qc)
statevector_ri = execute_quantum_circuit(circuit_json)
statevector = statevector_ri[:, 0] + 1j * statevector_ri[:, 1]
print(statevector)
```

Benchmark example:

```bash
cd quantumforge/python
python benchmarks/compare_simulators.py
```

## Syndrome-Net integration note

This package feeds the optional `qhybrid` acceleration path used by Syndrome-Net.
If the Rust extension is not available, Syndrome-Net falls back cleanly while preserving backend telemetry:

- `backend_chain` and `backend_chain_tokens` include attempted/resolved backends.
- `contract_flags` and `profiler_flags` remain present for CI parsing.
- fallback behavior remains deterministic for same seed and backend probe state.