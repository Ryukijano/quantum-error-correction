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
pip install -r app/requirements.txt
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

## Deploying to Hugging Face Spaces

Push this repository to your Space — the YAML header above tells HF Spaces
to use the Streamlit SDK and `app/requirements.txt` for dependencies.

```bash
git remote add space https://huggingface.co/spaces/Ryukijano/<space-name>
git subtree push --prefix=. space main
```
