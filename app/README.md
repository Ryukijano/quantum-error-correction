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
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Deploying to Hugging Face Spaces

The Space is already configured via the YAML header above. Push the repository
(or the `app/` directory) to your HF Space and Spaces will auto-build from
`app/requirements.txt`.

```bash
git remote add space https://huggingface.co/spaces/Ryukijano/<space-name>
git push space main
```
