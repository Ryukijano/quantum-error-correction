# Reinforcement Learning Experiments

This document records the reproducibility contract, benchmark definitions, and
recommended plotting workflow for the RL/QEC package.

## Reproducibility

The `compare_nested_policies` helper derives seeds from stable md5 hashes of
builder and policy names instead of Python's salted `hash` output. This keeps the
seed assigned to each `(builder, policy)` pair identical across interpreter
restarts and operating systems. Providing a `base_seed` still allows grouping
related runs while maintaining deterministic offsets for every individual
policy.

## Canonical Experiment Families

- **Decoder thresholds**
  - Script: `scripts/plot_threshold.py`
  - Metric: logical error rate vs physical error rate
  - Builders: `surface`, `hexagonal`, `walking`, `iswap`, `xyz2`
  - Decoders: `mwpm`, `union_find`, `sparse_blossom`

- **Decoder benchmarks**
  - Script: `scripts/benchmark_decoders.py`
  - Circuit suite: surface + dynamic code families
  - qLDPC suite: toric and hypergraph-product parity models
  - Metrics: logical error rate, block error rate, syndrome residual, runtime

- **RL training curves**
  - Script: `scripts/train_sota_rl.py`
  - PPO history: `ppo_history.json`
  - SAC history: `sac_history.json`
  - Plotter: `scripts/plot_training_curves.py`

### RL loop modes

The Streamlit UI and `scripts/train_sota_rl.py` share the same `RLTrainingStrategy`
implementations for PPO/SAC. The app adds an additional `pepg` pathway for
interactive tuning sessions; the script currently exposes `--mode ppo | sac | all`.

- `ppo` (and colour-code variants):
  - Environment: one-shot decoding (`QECGymEnv`, `ColourCodeGymEnv`)
  - Purpose: policy learning from syndrome payloads
  - Key runtime metrics: `policy_loss`, `value_loss`, `alpha_loss`, `learning_steps`, `policy_updates`, `rl_success_iqm`
- `sac` (and colour-code variants):
  - Environment: continuous calibration (`QECContinuousControlEnv`, `ColourCodeCalibrationEnv`, `QECCodeDiscoveryEnv`)
  - Purpose: continuous control over calibration and code-parameter schedules
  - Key runtime metrics: `policy_loss`, `value_loss`, `alpha_loss`, `alpha`, `policy_updates`, `rl_success_iqm`, `ler_ci`
- `pepg` (and colour-code variants):
  - Environment: continuous parameter optimization path
  - Purpose: evolutionary-style optimization loop for policy-free sweeps
  - Key runtime metrics: `policy_updates`, `grad_norm`, `rl_success_iqm`

### Resource footprint

- Script: `scripts/teraquop_footprint.py`
- Metric: estimated physical-qubit overhead for a target logical error rate

## Recommended Quick / Production Modes

- **Quick mode**
  - Use `--quick` for all plotting/benchmark scripts.
  - Intended for CI, smoke tests, and notebook iteration.
  - Small distance sets, fewer `p` samples, low shots.

- **Production mode**
  - Default execution path without `--quick`.
  - Intended for figure generation, appendix tables, and reports.
  - Larger distance sweep and more shots per point.

## Suggested Publishing Workflow

1. Run `scripts/plot_threshold.py --quick` to verify the plotting pipeline.
2. Run `scripts/benchmark_decoders.py --quick` to compare all decoders on a
   small sweep.
3. Run `scripts/train_sota_rl.py --mode all --episodes 200` and then
   `scripts/plot_training_curves.py --quick`.
4. Run the full sweeps for final figures and save the outputs as PDF + PNG.

## Benchmark and runtime contract fields

Both benchmark and runtime collectors should preserve metadata used by CI:

- Decoder benchmark rows (`scripts/benchmark_decoders.py`):
  - `backend`, `backend_enabled`, `backend_version`, `fallback_reason`
  - `sample_trace_id`, `backend_chain`, `backend_chain_tokens`
  - `contract_flags`, `profiler_flags`
- Runtime stream rows (`scripts/bench_runtime_contracts.py` and Streamlit training events):
  - `backend_id`, `backend_enabled`, `sample_us`, `backend_version`, `sample_rate`
  - `trace_tokens`, `backend_chain`, `contract_flags`, `profiler_flags`
  - `sample_trace_id`, `fallback_reason`, `details`, `ler_ci`

If one or more fields are missing, downstream parsers should fail fast in benchmark CI checks.

## Merge-aware experiment flow

Before sharing or shipping runs across repos, keep this sequence:

1. Sync `quantumforge` accelerator source and lockfiles if updated.
2. Run minimal contract checks for sampler and benchmarking paths:

```bash
python3 -m pytest \
  tests/test_sampling_backend_contracts.py \
  tests/test_benchmark_decoder_contracts.py \
  tests/test_cuda_q_decoder.py \
  tests/test_runtime_contracts.py
```

3. Run a smoke benchmark output on your intended target matrix:

```bash
python3 scripts/benchmark_decoders.py --quick --sampling-backends stim --suite circuit --output-dir artifacts/benchmarks
```

4. If any fallback/contract flags changed, include a short note in the experiment report and rerun the same command with the new suite arguments.

This flow prevents silent drift when two repositories evolve at different rates.

## Canonical benchmark artifact checklist

For each result row (CSV or JSON), preserve:

- `backend`
- `backend_enabled`
- `backend_version`
- `backend_chain`
- `backend_chain_tokens`
- `contract_flags`
- `profiler_flags`
- `fallback_reason`
- `sample_trace_id`

These fields are intentionally tracked even for non-accelerated fallback runs.

## Figure Quality Standard

- Use colorblind-safe palettes.
- Prefer bold titles and labels.
- Export both PDF and PNG.
- Keep all benchmark metadata in adjacent JSON and CSV files.
