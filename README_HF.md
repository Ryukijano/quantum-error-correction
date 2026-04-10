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

Real-time interactive visualizer for **quantum error correction** circuits
and **reinforcement learning** decoders, powered by
[Stim](https://github.com/quantumlib/Stim) and
[Syndrome-Net](https://github.com/Ryukijano/syndrome-net).

## Tabs

| Tab | Description |
|-----|-------------|
| 🔬 **Circuit Viewer** | Stim SVG circuit diagrams (timeline / detector-slice) + Crumble interactive editor + detector error graph with fired-detector highlighting |
| ⚡ **RL Live Training** | Start Transformer-PPO or Continuous-SAC training and watch reward, success rate, and syndrome heatmaps update live |
| 📈 **Threshold Explorer** | Sweep physical error rates → log-scale p_L curves with automatic threshold crossing detection |
| 🛰 **Teraquop Footprint** | Physical qubit overhead estimator across surface / hexagonal / toric / hypergraph-product codes |

## Running locally (conda `jax` env)

```bash
git clone https://github.com/Ryukijano/syndrome-net
cd syndrome-net
conda activate jax
pip install -r app/requirements.lock
streamlit run app/streamlit_app.py
```

Or via the convenience wrapper (defaults to the `jax` env):

```bash
chmod +x run_app.sh
./run_app.sh
```

To run any analysis script in the same env:

```bash
chmod +x run_scripts.sh
./run_scripts.sh scripts/plot_threshold.py --quick
./run_scripts.sh scripts/benchmark_decoders.py --quick
```

### Optional accelerated backend stack

If you want to profile GPU-accelerated or optional Rust-accelerated sampling paths from the app UI:

- `qhybrid` backend: optional local build via `quantumforge/python`
  ```bash
  cd quantumforge/python
  python -m pip install -r requirements.lock
  pip install maturin
  maturin develop
  ```
- `cuquantum` backend: install from NVIDIA container/Conda package (`cuquantum`)
- `qujax` backend: JAX-enabled environment (`jax`, `jaxlib`)
- `cudaq` backend: `pip install cudaq` (or provider package)

The UI uses `Auto` backend probing by default and records fallback/profiling details in each metric payload (`backend_chain`, `backend_id`, `contract_flags`, `profiler_flags`).

### Backend dependency matrix (Syndrome-Net)

| Backend ID | Module/Package | Probe trigger | Default behaviour |
|---|---|---|---|
| `qhybrid` | `qhybrid_kernels.rust_kernels` | `qhybrid_backend.probe_capability()["enabled"]` | Fast path when Rust extension is installed; otherwise fallback to `stim` |
| `cuquantum` | `cuquantum.tensornet` | `from cuquantum import tensornet` succeeds | Optional; enters after qhybrid when probing |
| `qujax` | `jax` | `import jax` succeeds | Optional; currently wrapper-backed fallback path |
| `cudaq` | `cudaq` | `import cudaq` succeeds | Optional; enters after qujax when probing |
| `stim` | `stim` | `import stim` succeeds | Baseline path used as final fallback |

Note: `Dockerfile.hf` currently installs `quantumforge/python/requirements.lock` but does not run a Rust toolchain build step, so `qhybrid` is usually unavailable in vanilla Space runtime images unless pre-built wheels are provided separately.

### Merge and deployment strategy for accelerated stack

If you keep `quantumforge` in a separate canonical repository, treat this repo as a consumable consumer snapshot:

- keep `requirements.lock`, `app/requirements.lock`, and `quantumforge/python/requirements.lock` aligned.
- sync upstream `quantumforge` releases by snapshotting files into `quantumforge/` on a controlled branch.
- maintain optional backend compatibility checks in CI (`backend_contracts`, `runtime_contracts`) so consumers can see when `qhybrid`/`cuquantum`/`qujax`/`cudaq` behavior changes.

Suggested verification before pushing to HF:

```bash
python3 -m pytest tests/test_sampling_backend_contracts.py tests/test_benchmark_decoder_contracts.py tests/test_runtime_contracts.py
python3 scripts/benchmark_decoders.py --quick --sampling-backends stim --suite circuit --output-dir /tmp/hf-bench
```

For deployment documentation including command-level repo merge options, see
`docs/REPO_MERGE_AND_DEPLOYMENT.md`.

For full QEC and colour-code experiments, install:

```bash
pip install -r requirements-qec.txt
```

## Streamlit controls you should expect

The advanced sidebars in `app/streamlit_app.py` map to three key runtime surfaces:

- **Advanced backend controls**: sampling backend selection, backend probing trace token, and profiler payload toggle
- **Advanced RL controls**: curriculum, early stopping, PPO/SAC/PEPG-specific knobs
- **RL mode selection**: `ppo`, `sac`, or `pepg` loops from the RL Live Training tab

## Deploying to Hugging Face Spaces

Push this repository to your Space — the YAML header above tells HF Spaces
to use the Streamlit SDK and `app/requirements.lock` for app dependencies.

```bash
git remote add space https://huggingface.co/spaces/Ryukijano/<space-name>
git subtree push --prefix=. space main
```

## Reproducible Container Runtime

```bash
docker build -f Dockerfile.hf -t syndrome-net-hf .
docker run --rm -p 8501:8501 syndrome-net-hf
```

For deterministic builds, the container image installs dependencies from:

- `requirements.lock`
- `app/requirements.lock`
- `quantumforge/python/requirements.lock`

The Streamlit runtime reads `.streamlit/config.toml` for stable
server/theme defaults.
