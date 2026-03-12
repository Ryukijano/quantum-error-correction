# Scientific Visualization & Benchmarking

Generates publication-quality figures and benchmarks QEC policies for quantum error correction research.

## Benchmarking

- Compare RL/ML policies against traditional decoders (MWPM, BPOSD) using codebase helpers (e.g., `_logical_error_rate`).
- Track metrics: logical error rates, physical qubit overhead, thresholds, decoding latency.
- **ML/RL Visualization**:
    - Plot training loss curves (actor/critic loss).
    - Track logical error rate evolution over training steps.
    - Visualize policy entropy and exploration rates.

## Publication-Quality Plots (MANDATORY)

- **Professional aesthetics**: Produce polished, visually strong figures.
- **Legends & labels**: Bold legends, large clear fonts for axes, ticks, titles—suitable for scientific publications.
- **Styling**: Colorblind-friendly palettes; distinct markers and line styles to differentiate series.
- **Libraries**: Prefer `matplotlib` and `seaborn` with publication styles (e.g., `plt.style.use('seaborn-v0_8-paper')`).
- **Export**: Save as PDF or SVG for vector graphics, or 300+ DPI PNG for raster; compatible with LaTeX.

## Specific Plot Types

- **Vulnerability Heatmap**: Plot spacetime coordinates (X=Time, Y=Location) colored by failure frequency to identify decoder blind spots.
- **Entanglement Spectrum**: Bar chart of eigenvalues grouped by cut location to diagnose topological order.
- **QEC Gain Plot**: Plot Logical Lifetime vs. Physical Lifetime (or Error Rate) to show break-even points (Gain > 1.0).

## Factual Verification

For competition deadlines, submission requirements, or standard benchmarking datasets, perform a web search first before responding.
