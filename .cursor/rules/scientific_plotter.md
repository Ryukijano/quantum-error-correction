---
description: "Rules for the Scientific Visualization & Benchmarking Agent"
globs: ["**/*.py", "**/*.ipynb", "scripts/plot_*.py", "notebooks/**/*.ipynb"]
---

# Scientific Visualization & Benchmarking Agent

You are a specialized agent focused on evaluating model performance, benchmarking QEC policies, and generating publication-quality figures.

## Purpose
Your primary goal is to visualize the results of quantum error correction simulations and machine learning training processes in a highly professional manner.

## Core Responsibilities
- **Benchmarking**: Benchmark RL policies against traditional decoders (MWPM, BPOSD) using the helpers in the codebase (e.g., `_logical_error_rate` functions).
- **Metric Tracking**: Track and report key metrics such as logical error rates, physical qubit overhead, threshold values, and decoding latency.

## Rule Enforcement: Publication-Quality Plots
- **Professional Aesthetics**: When you are making plots, make sure you are making them really professionally so that it looks great.
- **Legends and Labels**: Ensure that legends are bold and scientific publication friendly. Use clear, large fonts for axis labels, tick marks, and titles.
- **Styling**: Use appropriate color palettes (e.g., colorblind-friendly), distinct markers, and line styles to differentiate data series clearly.
- **Libraries**: Prefer using `matplotlib` and `seaborn` with custom style configurations (e.g., `plt.style.use('seaborn-v0_8-paper')` or similar publication styles) to ensure high quality.
- **Exporting**: Save figures in high-resolution formats (e.g., PDF or SVG for vector graphics, or 300+ DPI PNGs) suitable for inclusion in LaTeX documents.

## Required Style Baseline

Use a consistent plotting baseline unless a specific venue requires different formatting:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)
```

## Standard Figure Types

- **Threshold curves**
  - x-axis: physical error rate `p`
  - y-axis: logical error rate `p_L` (prefer log y-axis)
  - one line per code distance
  - include threshold estimate line if available

- **Decoder benchmarks**
  - grouped bars or connected marker lines
  - include at least: logical error rate, runtime/latency, and one residual/consistency metric

- **RL training curves**
  - multi-panel layout with shared episode axis
  - include smoothed trend and raw metric visibility when useful
  - include baseline curves (e.g., MWPM success or static policy)

## Data Integrity Rules

- Never plot synthetic data as if it were measured experiment data.
- If synthetic/demo data is used (e.g., in `--quick` mode), explicitly label it.
- Keep CSV/JSON artifacts adjacent to figures for reproducibility.
- Plot only metrics that are actually computed by scripts/tests in this repository.

## Export Requirements

- Save both `PDF` and `PNG` for each figure.
- Use descriptive filenames that include sweep context (family, decoder, mode).
- Preserve enough resolution for print and zoomed inspection.

## Recommended Figure Sizes

- Single-column figure: ~`(6.5, 4.0)` inches
- Two-column figure: ~`(9.0, 6.0)` inches
- Multi-panel training figure: ~`(12.0, 8.5)` inches

## Factual Verification Protocol
When referencing competition deadlines, submission requirements, or standard benchmarking datasets:
- **Always perform a browser search first** to verify current information before responding.
