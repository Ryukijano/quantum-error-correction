"""Core visualization helpers for QEC circuit and RL loop rendering.

Provides:
- Stim circuit SVG via circuit.diagram()
- Stim matchgraph HTML via diagram('matchgraph-3d-html') fallback
- Plotly syndrome heatmaps
- Detector graph network plots
- RL metrics time-series panels
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False


# ---------------------------------------------------------------------------
# Colour palette (colorblind-friendly, publication-grade)
# ---------------------------------------------------------------------------
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(15,17,26,1)",
    plot_bgcolor="rgba(20,22,35,1)",
    font=dict(family="JetBrains Mono, monospace", color="#e0e6f0", size=13),
    title_font=dict(size=16, color="#a8c8ff"),
    legend=dict(
        bgcolor="rgba(30,33,50,0.8)",
        bordercolor="#3a3f5c",
        borderwidth=1,
    ),
    margin=dict(l=50, r=30, t=55, b=45),
    xaxis=dict(gridcolor="#2a2f4a", zerolinecolor="#3a3f5c"),
    yaxis=dict(gridcolor="#2a2f4a", zerolinecolor="#3a3f5c"),
)


# ---------------------------------------------------------------------------
# Stim circuit diagrams
# ---------------------------------------------------------------------------

def circuit_svg(circuit: "stim.Circuit", diagram_type: str = "timeline-svg") -> str:
    """Return an SVG string for the given Stim circuit.

    diagram_type options (subset):
      - 'timeline-svg'          : gate-level time-axis view
      - 'timeslice-svg'         : one TICK slice
      - 'detslice-svg'          : stabilizer/detector 2-D view
      - 'detslice-with-ops-svg' : detectors + operations overlay
    """
    if not HAS_STIM:
        return "<svg><text x='10' y='20'>stim not installed</text></svg>"
    diag = circuit.diagram(diagram_type)
    return str(diag)


def circuit_interactive_html(circuit: "stim.Circuit") -> str:
    """Return Crumble interactive HTML for embedding in st.components.v1.html."""
    if not HAS_STIM:
        return "<p>stim not installed</p>"
    return str(circuit.diagram("interactive-html"))


def circuit_matchgraph_html(
    circuit: "stim.Circuit",
    diagram_type: str = "matchgraph-3d-html",
) -> str:
    """Return a Stim matchgraph visualization HTML for backend diagnostics.

    `diagram_type` can be switched to fallback options if the renderer doesn't
    support the preferred output.
    """
    if not HAS_STIM:
        return "<p>stim not installed</p>"

    candidates = (diagram_type, "matchgraph-3d-html", "matchgraph-3d-svg", "matchgraph-svg")
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return str(circuit.diagram(candidate))
        except Exception as exc:
            last_error = exc
            continue
    if last_error is None:
        return "<p>matchgraph unavailable</p>"
    return f"<p>matchgraph unavailable: {last_error}</p>"


# ---------------------------------------------------------------------------
# Syndrome heatmap
# ---------------------------------------------------------------------------

def syndrome_heatmap(
    syndrome: np.ndarray,
    distance: int,
    title: str = "Syndrome (detector events)",
    highlight_errors: Optional[np.ndarray] = None,
) -> go.Figure:
    """Plotly heatmap of the syndrome bit vector reshaped onto the 2-D code patch.

    Args:
        syndrome: 1-D bool/int array of length num_detectors.
        distance: code distance d  →  grid is roughly (d-1) x d.
        title:    figure title.
        highlight_errors: optional mask of known error positions (same length).
    """
    n = len(syndrome)
    ncols = distance
    nrows = math.ceil(n / ncols)

    grid = np.zeros((nrows, ncols), dtype=float)
    for i, v in enumerate(syndrome):
        grid[i // ncols, i % ncols] = float(v)

    hover = np.full((nrows, ncols), "", dtype=object)
    for i, v in enumerate(syndrome):
        r, c = i // ncols, i % ncols
        hover[r][c] = f"Det {i}: {'⚡ FIRE' if v else '·'}"

    fig = go.Figure(
        go.Heatmap(
            z=grid,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[
                [0.0, "rgba(20,22,35,1)"],
                [0.5, PALETTE["sky"]],
                [1.0, PALETTE["red"]],
            ],
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )

    if highlight_errors is not None:
        egrid = np.zeros((nrows, ncols), dtype=float)
        for i, v in enumerate(highlight_errors):
            if i < n:
                egrid[i // ncols, i % ncols] = float(v)
        # overlay errored cells with an X marker
        xs, ys = [], []
        for r in range(nrows):
            for c in range(ncols):
                if egrid[r, c] > 0:
                    xs.append(c)
                    ys.append(r)
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="x", size=14, color=PALETTE["orange"],
                            line=dict(width=2, color=PALETTE["orange"])),
                name="Error location",
                hoverinfo="skip",
            ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["title"] = title
    layout["xaxis"] = dict(**PLOTLY_LAYOUT["xaxis"], title="Column", showticklabels=False)
    layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], title="Row", showticklabels=False, autorange="reversed")
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Detector graph
# ---------------------------------------------------------------------------

def detector_graph(
    circuit: "stim.Circuit",
    syndrome: Optional[np.ndarray] = None,
    title: str = "Detector Error Graph",
) -> go.Figure:
    """Visualise the detector error model as a 2-D network.

    Fired detectors are coloured bright red. Edges are DEM hyperedge connections.
    """
    if not HAS_STIM:
        return go.Figure()

    try:
        dem = circuit.detector_error_model(decompose_errors=True)
    except Exception:
        return go.Figure(layout=dict(**PLOTLY_LAYOUT, title="DEM unavailable"))

    # Collect detector coordinates (from circuit annotations)
    coords: dict[int, tuple[float, float]] = {}
    for inst in circuit:
        if inst.name == "DETECTOR":
            args = inst.gate_args_copy()
            targets = inst.targets_copy()
            # targets[0] is the relative detector index inside the instruction
            # We use the coordinate annotations if present (x, y, t)
            if args and len(args) >= 2:
                det_idx = len(coords)
                coords[det_idx] = (float(args[0]), float(args[1]))

    num_dets = circuit.num_detectors
    # Fill missing coords on a grid
    for i in range(num_dets):
        if i not in coords:
            coords[i] = (float(i % 10), float(i // 10))

    # Build edge list from DEM
    edges: list[tuple[int, int, float]] = []
    for inst in dem:
        if inst.type == "error":
            prob = inst.args_copy()[0]
            targets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            for i in range(len(targets)):
                for j in range(i + 1, len(targets)):
                    edges.append((targets[i], targets[j], prob))

    fired = set()
    if syndrome is not None:
        for i, v in enumerate(syndrome):
            if v:
                fired.add(i)

    # Build traces
    edge_x, edge_y, edge_w = [], [], []
    for (u, v, p) in edges[:500]:  # cap for performance
        xu, yu = coords.get(u, (0, 0))
        xv, yv = coords.get(v, (0, 0))
        edge_x += [xu, xv, None]
        edge_y += [yu, yv, None]
        edge_w.append(p)

    node_x = [coords[i][0] for i in range(num_dets)]
    node_y = [coords[i][1] for i in range(num_dets)]
    node_colors = [
        PALETTE["red"] if i in fired else PALETTE["blue"]
        for i in range(num_dets)
    ]
    node_sizes = [14 if i in fired else 7 for i in range(num_dets)]
    node_text = [f"D{i}{'  ⚡' if i in fired else ''}" for i in range(num_dets)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#3a3f5c"),
        hoverinfo="none",
        name="DEM edges",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color="#aabbdd"),
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color="#1a1d2e"),
        ),
        hovertemplate="%{text}<extra></extra>",
        name="Detectors",
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["title"] = title
    layout["showlegend"] = False
    layout["xaxis"] = dict(**PLOTLY_LAYOUT["xaxis"], showticklabels=False, title="")
    layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], showticklabels=False, title="", autorange="reversed")
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# RL training metrics panel
# ---------------------------------------------------------------------------

def rl_metrics_panel(
    history: list[dict],
    window: int = 20,
    title: str = "RL Training — Live",
) -> go.Figure:
    """Multi-row Plotly panel: reward, success rate vs MWPM, logical error rate."""
    if not history:
        fig = go.Figure(layout=dict(**PLOTLY_LAYOUT, title=title))
        return fig

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]

    def smooth(arr, w):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid").tolist()

    rewards   = [h.get("reward", 0.0) for h in history]
    rl_succ   = [h.get("rl_success", 0.0) for h in history]
    rl_iqm    = [h.get("rl_success_iqm", v) for v, h in zip(rl_succ, history)]
    mw_succ   = [h.get("mwpm_success", 0.0) for h in history]
    ler       = [h.get("logical_error_rate", 0.0) for h in history]

    n_rows = 2 if any(v > 0 for v in ler) else 2
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=["Reward", "Decoding Success Rate"],
    )

    # Row 1 — reward (raw faint + smoothed bold)
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards,
        mode="lines", name="Reward (raw)",
        line=dict(color=PALETTE["sky"], width=1, dash="dot"),
        opacity=0.4,
    ), row=1, col=1)
    if len(rewards) >= window:
        ep_s = episodes[window - 1:]
        fig.add_trace(go.Scatter(
            x=ep_s, y=smooth(rewards, window),
            mode="lines", name=f"Reward (MA-{window})",
            line=dict(color=PALETTE["sky"], width=2.5),
        ), row=1, col=1)

    # Row 2 — RL vs MWPM success
    fig.add_trace(go.Scatter(
        x=episodes, y=rl_succ,
        mode="lines", name="RL agent",
        line=dict(color=PALETTE["green"], width=2.2),
        opacity=0.5,
    ), row=2, col=1)
    if len(rl_succ) >= window:
        fig.add_trace(go.Scatter(
            x=episodes[window - 1:], y=smooth(rl_succ, window),
            mode="lines", name=f"RL (MA-{window})",
            line=dict(color=PALETTE["green"], width=3),
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=episodes, y=mw_succ,
        mode="lines", name="MWPM baseline",
        line=dict(color=PALETTE["orange"], width=2, dash="dash"),
    ), row=2, col=1)
    if any(v for v in rl_iqm):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=rl_iqm,
            mode="lines",
            name="RL (IQM)",
            line=dict(color=PALETTE["purple"], width=2.2, dash="dot"),
        ), row=2, col=1)

    # Styling
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#a8c8ff")),
        paper_bgcolor=PLOTLY_LAYOUT["paper_bgcolor"],
        plot_bgcolor=PLOTLY_LAYOUT["plot_bgcolor"],
        font=PLOTLY_LAYOUT["font"],
        legend=PLOTLY_LAYOUT["legend"],
        margin=PLOTLY_LAYOUT["margin"],
        height=480,
    )
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{axis: dict(gridcolor="#2a2f4a", zerolinecolor="#3a3f5c")})
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Success rate", range=[0, 1.05], row=2, col=1)
    return fig


def policy_diagnostics_panel(
    history: list[dict],
    window: int = 10,
    title: str = "Policy diagnostics",
) -> go.Figure:
    """Panel for PPO/SAC optimization diagnostics (loss curves + entropy/alpha traces)."""
    if not history:
        return go.Figure(layout=dict(**PLOTLY_LAYOUT, title=title))

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    policy_loss = [float(h.get("policy_loss", 0.0) or 0.0) for h in history]
    value_loss = [float(h.get("value_loss", 0.0) or 0.0) for h in history]
    alpha_loss = [float(h.get("alpha_loss", 0.0) or 0.0) for h in history]
    alpha = [float(h.get("alpha", 0.0) or 0.0) for h in history]
    updates = [float(h.get("policy_updates", 0.0) or 0.0) for h in history]
    sigma_mean = [float(h.get("sigma_mean", 0.0) or 0.0) for h in history]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["Policy/value/alpha-loss", "Entropy alpha + updates"],
    )

    def smooth(arr, w):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid").tolist()

    if any(v != 0.0 for v in policy_loss):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=policy_loss,
            mode="lines",
            name="policy_loss",
            line=dict(color=PALETTE["sky"], width=2),
        ), row=1, col=1)
        if len(policy_loss) >= window:
            fig.add_trace(go.Scatter(
                x=episodes[window - 1:],
                y=smooth(policy_loss, window),
                mode="lines",
                name=f"policy_loss MA-{window}",
                line=dict(color=PALETTE["sky"], width=2.8),
            ), row=1, col=1)

    if any(v != 0.0 for v in value_loss):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=value_loss,
            mode="lines",
            name="value_loss",
            line=dict(color=PALETTE["orange"], width=2),
        ), row=1, col=1)

    if any(v != 0.0 for v in alpha_loss):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=alpha_loss,
            mode="lines",
            name="alpha_loss",
            line=dict(color=PALETTE["purple"], width=2),
        ), row=2, col=1)
    if any(v != 0.0 for v in alpha):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=alpha,
            mode="lines",
            name="alpha",
            line=dict(color=PALETTE["green"], width=2, dash="dot"),
        ), row=2, col=1)
    if any(v != 0.0 for v in updates):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=updates,
            mode="lines",
            name="policy_updates",
            line=dict(color=PALETTE["red"], width=2),
            opacity=0.6,
        ), row=2, col=1)
    if any(v != 0.0 for v in sigma_mean):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=sigma_mean,
            mode="lines",
            name="sigma_mean",
            line=dict(color=PALETTE["yellow"], width=2, dash="dash"),
            opacity=0.8,
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#a8d8ff")),
        paper_bgcolor=PLOTLY_LAYOUT["paper_bgcolor"],
        plot_bgcolor=PLOTLY_LAYOUT["plot_bgcolor"],
        font=PLOTLY_LAYOUT["font"],
        legend=PLOTLY_LAYOUT["legend"],
        margin=PLOTLY_LAYOUT["margin"],
        height=440,
    )
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{axis: dict(gridcolor="#2a2f4a", zerolinecolor="#3a3f5c")})
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Loss / alpha", row=1, col=1)
    fig.update_yaxes(title_text="Aux diagnostics", row=2, col=1)
    return fig


def logical_error_rate_panel(
    history: list[dict],
    window: int = 10,
    title: str = "Logical Error Rate — SAC Calibration",
) -> go.Figure:
    """Log-scale LER vs episode for continuous calibration runs."""
    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    ler = [max(h.get("logical_error_rate", 1e-4), 1e-6) for h in history]
    ep = [h.get("effective_p", 0.0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=ler,
        mode="lines+markers",
        name="Logical error rate",
        line=dict(color=PALETTE["red"], width=2.2),
        marker=dict(size=4),
    ))
    if ep and any(v > 0 for v in ep):
        fig.add_trace(go.Scatter(
            x=episodes, y=ep,
            mode="lines",
            name="Effective p",
            line=dict(color=PALETTE["orange"], width=1.8, dash="dash"),
        ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["title"] = title
    layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], type="log", title="Error rate")
    layout["xaxis"] = dict(**PLOTLY_LAYOUT["xaxis"], title="Episode")
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Threshold curve
# ---------------------------------------------------------------------------

def threshold_figure(
    data: dict[int, list[tuple[float, float]]],
    threshold_p: Optional[float] = None,
    title: str = "Error Threshold",
) -> go.Figure:
    """Plotly version of threshold curve.

    Args:
        data: {distance: [(p, p_L), ...]}
        threshold_p: estimated crossing point
    """
    fig = go.Figure()
    colors = list(PALETTE.values())
    for idx, (dist, curve) in enumerate(sorted(data.items())):
        xs = [pt[0] for pt in curve]
        ys = [pt[1] for pt in curve]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=f"d={dist}",
            line=dict(color=colors[idx % len(colors)], width=2.2),
            marker=dict(size=6),
        ))

    if threshold_p is not None:
        fig.add_vline(
            x=threshold_p,
            line=dict(color=PALETTE["yellow"], width=2, dash="dash"),
            annotation_text=f"p_th ≈ {threshold_p:.4f}",
            annotation_font=dict(color=PALETTE["yellow"], size=12),
        )

    layout = dict(**PLOTLY_LAYOUT)
    layout["title"] = title
    layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], type="log", title="Logical error rate p_L")
    layout["xaxis"] = dict(**PLOTLY_LAYOUT["xaxis"], title="Physical error rate p")
    fig.update_layout(**layout)
    return fig
