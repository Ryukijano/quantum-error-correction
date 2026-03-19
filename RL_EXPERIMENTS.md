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

- **Resource footprint**
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

## Figure Quality Standard

- Use colorblind-safe palettes.
- Prefer bold titles and labels.
- Export both PDF and PNG.
- Keep all benchmark metadata in adjacent JSON and CSV files.
