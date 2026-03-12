# GitHub Actions Test Suite Design

Creates comprehensive, CPU-friendly test cases that run in GitHub Actions without GPU large runners.

## CI Constraints (MANDATORY)

- **No GPU**: All tests MUST run on `ubuntu-latest` CPU runners. Do not depend on GPU or large runners.
- **Fast**: Keep total suite under ~5 minutes. Use small problem sizes (e.g. `distance=3`, `rounds=2–3`, `shots≤32`).
- **Deterministic**: Use fixed seeds (`seed=...`) and explicit inputs. Avoid flaky assertions on stochastic metrics when possible.

## Test Categories to Cover

### 1. Circuits & Decoders

- Static surface code: `surface_code_circuit_string` produces valid Stim circuits for small `distance`/`rounds`/`p`.
- Dynamic codes: `hexagonal_surface_code`, `walking_surface_code`, `iswap_surface_code` produce valid circuits.
- **New Codes**: Verify `ClusteredCyclicCode` and `XYZ2HexagonalCode` produce valid Stim circuits with correct stabilizer counts.
- Decoders: MWPM, Union-Find, SparseBlossom agree on small circuits; outputs deterministic for fixed inputs.
- Logical error rate: `_logical_error_rate` is seed-deterministic; returns value in [0, 1].

### 2. Code Family Plugins

- `list_plugins()` includes expected families (e.g. `surface`, `qldpc`, `bosonic`, `dual_rail_erasure`).
- Placeholder plugins (qLDPC, dual_rail, bosonic) raise explicit errors with required parameter hints.
- `benchmark_code_families` runs `surface` and returns valid `logical_error_rate` in expected format.

### 3. RL / ML Components

- **Optimizers**: PEPG optimizer updates toward higher reward; finite outputs; `ask`/`tell` cycle works.
- **Masking**: `build_detector_parameter_mask`, `apply_masked_detector_weights` produce expected shapes.
- **Nested Learning**: `compare_nested_policies` returns expected keys; logical_error_rate in [0,1].
- **ML Models (CPU)**:
    - **TITANS**: Test memory module forward pass with tiny inputs (batch=1, seq_len=10) on CPU.
    - **GNNs**: Test Feedback GNN on small random graphs.
    - Use `torch.device('cpu')` or `jax.devices('cpu')` explicitly.
    - Mock heavy training loops or run only 1 step to verify pipeline mechanics.

### 4. Edge Cases & Invalid Inputs

- Invalid `distance`/`rounds`/`p` (e.g. negative, zero) raise clear errors.
- Missing required parameters for plugins raise `NotImplementedError` with instructive messages.
- Optional deps (e.g. `pymatching`): use `pytest.importorskip("pymatching")` so tests run when dep is absent.

## Pytest Patterns

```python
# Optional dependency – skip if not installed
pytest.importorskip("pymatching")

# Parametrize for multiple variants
@pytest.mark.parametrize("dynamic_builder", [hexagonal_surface_code, walking_surface_code, iswap_surface_code])
def test_all_builders(dynamic_builder): ...

# Deterministic seeds
compare_nested_policies(..., seed=123)
```

## GPU-Heavy Code (DO NOT run in CI)

Mark tests that require GPU and exclude them from the default run:

```python
@pytest.mark.gpu
def test_torch_model_on_gpu(): ...
```

In `pytest.ini` or `pyproject.toml`:

```ini
[pytest]
addopts = -m "not gpu"
```

Or in workflow: `pytest -m "not gpu"`.

## Workflow Checklist

When adding tests:

- [ ] No `cuda`, `gpu`, or `device=cuda` unless behind `@pytest.mark.gpu` and excluded in CI.
- [ ] Use `importorskip` for optional deps (pymatching, jax, torch).
- [ ] Small params: `distance=3`, `rounds≤3`, `shots≤64`.
- [ ] Fixed seeds for reproducibility.
- [ ] Assert structure (keys, shapes) before asserting stochastic values.
