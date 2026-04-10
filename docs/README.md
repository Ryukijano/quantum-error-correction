# Documentation Index

This directory contains deep-dive technical documentation for Syndrome-Net.

## Start Here

- `../Readme.md`: project overview, install, quickstart, and top-level architecture

## Architecture & RL

- `RL_QEC_ARCHITECTURE.md`: detailed architecture, mathematical formulation, and end-to-end RL training flow
- `../app/README.md`: Streamlit dashboard structure, tabs, and control surfaces
- `../README_HF.md`: deployment and Streamlit app dependency guidance
- `../docs/REPO_MERGE_AND_DEPLOYMENT.md`: repository topology/merge playbook and contract-check sequence
- `../docs/BENCHMARKING_CONTRACTS.md`: canonical schema for benchmark/runtime metadata outputs

## Related Notes

- `../RL_EXPERIMENTS.md`: reproducibility notes for RL experiments
- `../scripts/benchmark_decoders.py` and `../scripts/bench_runtime_contracts.py`: contract-field schemas for benchmark and runtime reporting
- `../surface_code_in_stem/DYNAMIC_CODES.md`: mapping from Morvan et al. (2025) dynamic codes to implementation

## Suggested Reading Path

1. Read `../Readme.md` for setup and quickstart.
2. Read `RL_QEC_ARCHITECTURE.md` for system internals and Mermaid diagrams.
3. Use `scripts/train_sota_rl.py` and tests in `../tests/` to run and validate.
