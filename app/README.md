---
title: Syndrome-Net QEC Lab
emoji: ⚛️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app/streamlit_app.py
pinned: true
license: mit
short_description: Interactive Quantum Error Correction + RL visualizer
---

# ⚛️ Syndrome-Net QEC Lab

An interactive visualizer for quantum error correction (QEC) circuits and
reinforcement learning decoders, built on top of the
[Syndrome-Net](https://github.com/Ryukijano/syndrome-net) research package.

## What's inside

| Tab | What you see |
|-----|-------------|
| 🔬 **Circuit Viewer** | Stim SVG circuit diagrams (timeline, detector-slice) + Crumble interactive editor + detector error graph |
| ⚡ **RL Live Training** | Real-time reward / success rate / syndrome heatmap while Transformer-PPO or Continuous-SAC trains |
| 📈 **Threshold Explorer** | Sweep physical error rates → log-scale logical error rate curves with crossing-point detection |
| 🛰 **Teraquop Footprint** | Physical qubit overhead for target logical error rate across code families |

## Running locally

```bash
conda activate syndrome-net      # or: conda activate jax
pip install -r app/requirements.lock
streamlit run app/streamlit_app.py
```

### Streamlit advanced controls (from RL Live Training)

The app exposes two expandable groups in the left sidebar:

- **Advanced backend controls**
  - `Sampling backend`: `auto`, `stim`, `qhybrid`, `cuquantum`, `qujax`, `cudaq`
  - `Enable backend trace payload`: include backend fallback trace columns in RL metric rows
  - `Trace token`: optional string recorded as `trace_tokens`
  - `Backend matrix window` and `backend sample window`: control how many recent trace events are retained for UI diagnostics

- **Advanced RL controls**
  - `Training seed`: deterministic run seeding for reproducibility
  - `Enable curriculum learning`: distance / error-rate curriculum scheduler toggle
  - `Early stopping` and window parameters: stop runs when reward or success stagnates
  - PPO/SAC/PEPG tuning knobs: gradient clipping, entropy terms, and PEPG perturbation strength

### Contract-aware payload keys (app path)

`app/services/rl_services.py` normalizes backend trace data before rendering charts and exports:

- `backend_id`
- `backend_enabled`
- `backend_chain`
- `backend_chain_tokens`
- `backend_version`
- `sample_us`
- `sample_rate`
- `sample_trace_id`
- `fallback_reason`
- `contract_flags`
- `profiler_flags`
- `trace_tokens`
- `trace_id`
- `details`
- `ler_ci`

Use these keys when wiring downstream data sinks, CI parsers, or custom exporters.

### Streamlit + merge-aware workflow

When syncing docs/code with the `quantumforge` tree:

- use `app/requirements.lock` for UI-only CI runs,
- keep root `requirements.lock` aligned for full benchmark/scripting tests,
- and ensure both `benchmark_decoders` CSV/JSON and profiler export formats preserve
  `backend_chain_tokens` as canonical list values in code (JSON-serialized only when writing files).

Training loop metadata from the app is normalized by `app/services/rl_services.py` with the metadata keys:

- `backend_id`, `backend_enabled`, `backend_chain`, `backend_version`
- `sample_us`, `sample_rate`, `sample_trace_id`, `fallback_reason`
- `contract_flags`, `profiler_flags`, `trace_tokens`, `trace_id`, `details`

For the full cross-repo merge playbook, see `../docs/REPO_MERGE_AND_DEPLOYMENT.md`.

## Deploying to Hugging Face Spaces

The Space is already configured via the YAML header above. Push the repository
(or the `app/` directory) to your HF Space and Spaces will auto-build from
`app/requirements.lock`.

```bash
git remote add space https://huggingface.co/spaces/Ryukijano/<space-name>
git push space main
```
