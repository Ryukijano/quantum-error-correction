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

# Syndrome-Net QEC Lab

## App purpose

Syndrome-Net is a Streamlit dashboard for building and inspecting quantum error-correcting circuits plus RL-based decoding workflows. It runs Stim-driven workflow simulations in the browser and exposes:

- **Circuit Viewer** for detector-slice and timeline visualizations.
- **RL Live Training** for PPO/SAC/PEPG experiments.
- **Threshold Explorer** for logical-error-rate sweeps.
- **Teraquop Footprint** for overhead estimates across code families.

## Deployment context

This document is intentionally focused on deployment and Space-style usage. For full API, research, and development instructions, use the root `Readme.md`.

## Deploy quickstart

### Hugging Face Spaces

The app is configured via the YAML header above (`app_file: app/streamlit_app.py`) and `app/requirements.lock`.

```bash
git remote add space https://huggingface.co/spaces/<org>/<space-name>
git push space main
```

Use `main` as the deployment branch, or set your Space to track a different branch.

### Docker

```bash
docker build -f Dockerfile.hf -t syndrome-net-hf .
docker run --rm -p 8501:8501 syndrome-net-hf
```

### Local smokeable launch

```bash
conda activate jax
pip install -r app/requirements.lock
python3 scripts/run_streamlit_smoke.py --timeout 35
python3 -m streamlit run app/streamlit_app.py
```

Wrapper (repo helper):

```bash
./run_app.sh
```

`run_app.sh` defaults to conda env `ENV=jax` and exposes `PORT` to Streamlit.

## Environment prerequisites

- Python: 3.10+ (HF build uses project lockfiles, typically Python 3.11 in CI).
- Memory/storage: enough for your chosen circuit sizes and RL episode depth.
- Backend dependencies:
  - Minimum: `app/requirements.lock` (works in fallback mode).
  - Optional accelerators: `requirements.lock`, `quantumforge/python/requirements.lock`, and platform backends (`qhybrid`, `cuquantum`, `qujax`, `cudaq`) when installed.
- OS/runtime: CPU-only Space images are supported; optional CUDA paths only when packages and drivers are available.
- Optional script dependencies: for non-app scripts use normal package install + `python` from the same Python environment.

## Smoke checks

Run these before sharing a deployment:

```bash
python3 scripts/run_streamlit_smoke.py --timeout 35
python3 scripts/ci_contract_verification.sh
```

Reproducibility smoke:

```bash
python3 -m pytest tests/test_streamlit_backend_metadata.py tests/test_runtime_contracts.py
```

For Docker verification:

```bash
docker run --rm -d --name syndrome-net-hf -p 8501:8501 syndrome-net-hf
python3 - <<'PY'
import urllib.request
print("status:", urllib.request.urlopen("http://127.0.0.1:8501").status)
PY
docker stop syndrome-net-hf
```

## Supported branches and entrypoints

- **Preferred deploy branch**: `main`.
- **Space entrypoint**: `app/streamlit_app.py` (`app_file` in header).
- **Container entrypoint**: `streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501` (from `Dockerfile.hf`).
- **CI entrypoint**: `scripts/run_streamlit_smoke.py`.
- **Auxiliary runners**: `run_scripts.sh` for other repo scripts and `run_app.sh` for Streamlit.

## Expected output

- HF Space or container serves Streamlit at `http://localhost:8501` (or Space URL).
- Landing UI shows the four primary tabs above and sidebar mode controls.
- Smoke scripts return exit code `0`.
- Log output should include app boot and optional backend fallback metadata (`backend_id`, `backend_chain`, `contract_flags`, `profiler_flags`) when enabled.

## Troubleshooting checklist

- **App starts but UI blank**: confirm `app/streamlit_app.py` exists and branch matches `main` or configured Space branch.
- **`ModuleNotFoundError` on launch**: ensure dependencies from `app/requirements.lock` installed and `python` points to same env as dependency install.
- **Port 8501 already in use**: change port via `PORT=<port> ./run_app.sh` or `-p <port>:8501` in Docker.
- **Docker build hangs**: network/proxy issues or stale pip cache; retry with a clean build.
- **Qhybrid/cudaq/cuquantum warnings**: expected when optional accelerators are unavailable; app should continue with Stim fallback.
- **Streamlit smoke timeout**: check startup logs; large models may need more startup time or stricter memory limits.

## Contributing and support

- For development and PR workflows, use root `Readme.md` plus `CONTRIBUTING.md`.
- Report security issues via `SECURITY.md`.
- Use `CODE_OF_CONDUCT.md` for behavior expectations.
