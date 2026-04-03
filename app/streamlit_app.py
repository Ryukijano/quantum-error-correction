"""Syndrome-Net Interactive Visualizer — Streamlit App.

Tabs:
  1. 🔬 Circuit Viewer  — Stim SVG circuit diagram + Crumble interactive editor
  2. ⚡ RL Live Training — real-time metrics + syndrome heatmap during training
  3. 📈 Threshold Explorer — interactive threshold sweep with Plotly
  4. 🛰  Teraquop Footprint — physical qubit overhead estimator
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Repo path injection
# ------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.live_viz import (
    create_surface_code_lattice,
    generate_live_rl_visualization,
)
from app.qec_viz import (
    PALETTE,
    PLOTLY_LAYOUT,
    circuit_interactive_html,
    circuit_svg,
    detector_graph,
    logical_error_rate_panel,
    rl_metrics_panel,
    syndrome_heatmap,
    threshold_figure,
)
from app.rl_runner import DoneEvent, ErrorEvent, MetricEvent, RLRunner, SyndromeEvent

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Syndrome-Net QEC Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Global CSS — dark quantum aesthetic
# ------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Dark base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f1a;
    color: #dce8ff;
}
[data-testid="stSidebar"] {
    background: #111424;
    border-right: 1px solid #1e2340;
}
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111424;
    border-bottom: 2px solid #1e2340;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #181c30;
    color: #8899cc;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    padding: 8px 20px;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: #1e2a50 !important;
    color: #a8d0ff !important;
    border-bottom: 3px solid #4a9eff;
}
/* ── Metrics / KPI cards ── */
[data-testid="metric-container"] {
    background: #151929;
    border: 1px solid #2a3060;
    border-radius: 12px;
    padding: 12px 18px;
}
[data-testid="metric-container"] label {
    color: #6688cc !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #a8d8ff;
    font-size: 1.6rem;
    font-weight: 700;
}
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a2a6c 0%, #4a1a8c 100%);
    color: #dce8ff;
    border: 1px solid #3a4a8c;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a3a8c 0%, #6a2aac 100%);
    border-color: #5a7acf;
    box-shadow: 0 0 12px rgba(100,150,255,0.3);
}
/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #4a9eff;
}
/* ── Status badges ── */
.badge-running { color:#4aff9e; font-weight:700; }
.badge-done    { color:#a8d8ff; font-weight:700; }
.badge-error   { color:#ff6b6b; font-weight:700; }
/* ── SVG circuit panel ── */
.circuit-panel {
    background: #0a0c18;
    border: 1px solid #1e2a50;
    border-radius: 12px;
    overflow: auto;
    max-height: 560px;
    padding: 16px;
}
/* ── Glow pulse for live indicator ── */
@keyframes glow {
    0%,100% { box-shadow: 0 0 6px rgba(74,255,158,0.4); }
    50%      { box-shadow: 0 0 18px rgba(74,255,158,0.9); }
}
.live-indicator {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #4aff9e;
    animation: glow 1.2s ease-in-out infinite;
    margin-right: 6px;
}
/* ── Live Environment Window ── */
.live-env-container {
    background: linear-gradient(135deg, #0a0c18 0%, #151929 100%);
    border: 2px solid #2a3060;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 0 20px rgba(74, 158, 255, 0.1);
}
.live-env-title {
    color: #ff6b6b;
    font-weight: 700;
    font-size: 1.1rem;
    text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
}
/* ── Error pulse animation ── */
@keyframes error-pulse {
    0%, 100% { box-shadow: 0 0 5px rgba(255, 68, 68, 0.5); }
    50% { box-shadow: 0 0 20px rgba(255, 68, 68, 1); }
}
/* ── Syndrome fire animation ── */
@keyframes syndrome-fire {
    0%, 100% { box-shadow: 0 0 5px rgba(255, 102, 0, 0.5); }
    50% { box-shadow: 0 0 25px rgba(255, 102, 0, 1); }
}
/* ── Correction glow ── */
@keyframes correction-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(68, 255, 68, 0.5); }
    50% { box-shadow: 0 0 15px rgba(68, 255, 68, 0.9); }
}
/* ── Legend styling ── */
.viz-legend {
    background: #151929;
    border: 1px solid #2a3060;
    border-radius: 8px;
    padding: 12px;
    margin-top: 8px;
}
.viz-legend-item {
    display: inline-flex;
    align-items: center;
    margin-right: 16px;
    font-size: 0.85rem;
    color: #a8d8ff;
}
.viz-legend-symbol {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 6px;
}

/* ── Enhanced professional polish for snappy UX ── */
.stPlotlyChart {
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease;
}
.stPlotlyChart:hover {
    transform: translateY(-2px);
}
.export-btn {
    background: linear-gradient(135deg, #1a2a6c 0%, #4a1a8c 100%) !important;
    color: white !important;
    border-radius: 6px !important;
}
.error-state {
    animation: error-pulse 1s ease-in-out infinite;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Sidebar — global controls
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚛️ Syndrome-Net")
    st.markdown("*Quantum Error Correction × Deep RL*")
    st.divider()

    st.markdown("### Code parameters")
    distance = st.slider("Code distance d", 3, 9, 3, step=2)
    rounds    = st.slider("Syndrome rounds", 1, 6, 2)
    p_phys    = st.slider("Physical error rate p", 0.001, 0.02, 0.01, step=0.001, format="%.3f")

    st.divider()
    st.markdown("### RL training")
    rl_mode     = st.selectbox("Agent", ["ppo", "sac"])
    episodes    = st.slider("Episodes", 50, 2000, 300, step=50)
    batch_size  = st.slider("Batch size", 16, 128, 32, step=16)
    use_diff    = st.checkbox("Use flow-matching (SAC)", value=False)

    st.divider()
    diagram_type = st.selectbox(
        "Circuit diagram style",
        ["timeline-svg", "timeslice-svg", "detslice-svg", "detslice-with-ops-svg"],
    )

    st.divider()
    st.caption("Syndrome-Net · MIT licence")


# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------
def _init_state() -> None:
    defaults = dict(
        runner=None,
        history=[],
        latest_syndrome=None,
        latest_action=None,
        latest_correct=False,
        training_done=False,
        training_error=None,
        training_episode=0,
        error_locations=None,  # for interactive error injection
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ------------------------------------------------------------------
# Helper: build Stim circuit
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_circuit(distance: int, rounds: int, p: float):
    try:
        import stim
        from surface_code_in_stem.surface_code import surface_code_circuit_string
        return stim.Circuit(surface_code_circuit_string(distance=distance, rounds=rounds, p=p))
    except Exception as exc:
        return None, str(exc)


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_circuit, tab_rl, tab_threshold, tab_footprint = st.tabs([
    "🔬 Circuit Viewer",
    "⚡ RL Live Training",
    "📈 Threshold Explorer",
    "🛰 Teraquop Footprint",
])


# ==================================================================
# TAB 1 — Circuit Viewer
# ==================================================================
with tab_circuit:
    st.markdown("## Stim Circuit Viewer")
    st.markdown(
        "Visual inspection of the surface-code Stim circuit. "
        "Switch between timeline and detector-slice views in the sidebar."
    )

    circuit = _get_circuit(distance, rounds, p_phys)
    if circuit is None:
        st.error("Could not build circuit — check that `stim` is installed.")
    else:
        col_svg, col_graph = st.columns([3, 2])

        with col_svg:
            st.markdown("#### Circuit diagram")
            view_mode = st.radio(
                "View",
                ["Static SVG", "Interactive (Crumble)"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if view_mode == "Static SVG":
                svg_str = circuit_svg(circuit, diagram_type)
                st.markdown(
                    f'<div class="circuit-panel">{svg_str}</div>',
                    unsafe_allow_html=True,
                )
            else:
                html_str = circuit_interactive_html(circuit)
                st.components.v1.html(html_str, height=540, scrolling=True)

        with col_graph:
            st.markdown("#### Detector error graph")
            st.caption("Nodes = detectors · Edges = DEM connections")

            # Sample one syndrome to highlight fired detectors
            try:
                sampler = circuit.compile_detector_sampler(seed=42)
                det, _ = sampler.sample(1, separate_observables=True)
                syn = det[0].astype(np.int8)
            except Exception:
                syn = None

            dem_fig = detector_graph(circuit, syndrome=syn, title="DEM — sample syndrome")
            st.plotly_chart(dem_fig, width="stretch")

        st.divider()
        st.markdown("#### Syndrome heatmap (single shot)")
        if syn is not None:
            hm_fig = syndrome_heatmap(syn, distance=distance)
            st.plotly_chart(hm_fig, width="stretch")
            fired = int(np.sum(syn))
            st.caption(f"🔴 {fired} / {len(syn)} detectors fired in this shot.")

        with st.expander("Circuit statistics"):
            try:
                st.json({
                    "num_qubits": circuit.num_qubits,
                    "num_detectors": circuit.num_detectors,
                    "num_observables": circuit.num_observables,
                    "num_instructions": len(list(circuit)),
                    "distance": distance,
                    "rounds": rounds,
                    "physical_error_rate": p_phys,
                })
            except Exception as e:
                st.warning(str(e))


# ==================================================================
# TAB 2 — RL Live Training
# ==================================================================
with tab_rl:
    st.markdown("## ⚡ Live RL Training")
    st.markdown(
        "Train a Transformer-PPO (discrete decoding) or Continuous SAC (calibration) agent "
        "and watch metrics + syndrome snapshots update in real time."
    )

    runner: RLRunner | None = st.session_state.runner

    ctrl_col, status_col = st.columns([2, 3])
    with ctrl_col:
        start_btn = st.button("▶ Start Training", type="primary", width="stretch")
        stop_btn  = st.button("⏹ Stop", width="stretch")

    with status_col:
        status_box = st.empty()

    # Handle start
    if start_btn:
        if runner and runner.is_running():
            st.warning("Training is already running.")
        else:
            st.session_state.history = []
            st.session_state.latest_syndrome = None
            st.session_state.training_done = False
            st.session_state.training_error = None
            st.session_state.training_episode = 0
            st.session_state.latest_error_locations = None
            st.session_state.error_history = []
            new_runner = RLRunner(
                mode=rl_mode,
                distance=distance,
                rounds=rounds,
                physical_error_rate=p_phys,
                episodes=episodes,
                batch_size=batch_size,
                use_diffusion=use_diff,
                syndrome_emit_every=max(1, episodes // 40),
            )
            new_runner.start()
            st.session_state.runner = new_runner

    if stop_btn and runner:
        runner.stop()

    # ------------------------------------------------------------------
    # Metrics layout
    # ------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi_episode  = kpi1.empty()
    kpi_reward   = kpi2.empty()
    kpi_success  = kpi3.empty()
    kpi_mwpm     = kpi4.empty()

    chart_metrics   = st.empty()
    chart_syndrome  = st.empty()

    st.divider()
    col_heatmap, col_ler = st.columns(2)
    chart_heatmap   = col_heatmap.empty()
    chart_ler       = col_ler.empty()

    # ------------------------------------------------------------------
    # Live Environment Visualization — Real-time error propagation
    # ------------------------------------------------------------------
    st.divider()
    st.markdown(
        '<div class="live-env-container">'
        '<span class="live-env-title">🔴 Live Environment — Error Propagation</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.caption("Watch errors appear (🔴 X-error, 🔵 Z-error, 🟣 Y-error), syndrome measurements fire (⭐), and decoder corrections apply (🟢) in real-time.")
    
    # Legend
    st.markdown(
        """
        <div class="viz-legend">
            <span class="viz-legend-item"><span class="viz-legend-symbol" style="background:#ff4444"></span> X Error</span>
            <span class="viz-legend-item"><span class="viz-legend-symbol" style="background:#4444ff"></span> Z Error</span>
            <span class="viz-legend-item"><span class="viz-legend-symbol" style="background:#ff00ff"></span> Y Error (X+Z)</span>
            <span class="viz-legend-item"><span class="viz-legend-symbol" style="background:#ff6600"></span> X-Syndrome Fired</span>
            <span class="viz-legend-item"><span class="viz-legend-symbol" style="background:#00aaff"></span> Z-Syndrome Fired</span>
            <span class="viz-legend-item"><span class="viz-legend-symbol" style="background:#44ff44"></span> Decoder Correction</span>
            <span class="viz-legend-item"><span class="viz-legend-symbol" style="background:#666666"></span> Clean Qubit</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    rl_left, rl_right = st.columns([3, 2])
    
    with rl_left:
        st.markdown("**Live Lattice**")
        live_lattice = st.empty()
        if st.button("🔴 Inject Random X Error", key="inject_error", use_container_width=True):
            if 'error_locations' not in st.session_state or not isinstance(st.session_state.error_locations, dict):
                st.session_state.error_locations = {'x': [], 'z': []}
            import random
            d = distance
            n_data = len(create_surface_code_lattice(d)['data'])
            idx = random.randint(0, n_data - 1)
            if len(st.session_state.error_locations.get('x', [])) == 0:
                st.session_state.error_locations['x'] = [False] * n_data
                st.session_state.error_locations['z'] = [False] * n_data
            st.session_state.error_locations['x'][idx] = not st.session_state.error_locations['x'][idx]
            st.rerun()
    
    with rl_right:
        st.markdown("**Circuit Viewer (during training)**")
        circuit = _get_circuit(distance, rounds, p_phys)
        if circuit is None:
            st.error("Could not build circuit")
        else:
            view_mode = st.radio(
                "View",
                ["Static SVG", "Interactive (Crumble)"],
                horizontal=True,
                key="rl_circuit_view"
            )
            if view_mode == "Static SVG":
                svg_str = circuit_svg(circuit, diagram_type)
                st.markdown(f'<div class="circuit-panel">{svg_str}</div>', unsafe_allow_html=True)
            else:
                html_str = circuit_interactive_html(circuit)
                st.components.v1.html(html_str, height=480, scrolling=True)
            
            st.caption("Detector graph and heatmap below")
            try:
                sampler = circuit.compile_detector_sampler(seed=42)
                det, _ = sampler.sample(1, separate_observables=True)
                syn = det[0].astype(np.int8)
                dem_fig = detector_graph(circuit, syndrome=syn, title="DEM — sample")
                st.plotly_chart(dem_fig, use_container_width=True)
                hm_fig = syndrome_heatmap(syn, distance=distance, title="Syndrome Heatmap")
                st.plotly_chart(hm_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Circuit viz error: {e}")
        
    # Generate live visualization (left column only)
    history = st.session_state.history
    syn = st.session_state.latest_syndrome
    action = st.session_state.latest_action
    reward = history[-1].get('reward', 0.0) if history else 0.0
    correct = st.session_state.latest_correct
    ep = st.session_state.training_episode
    
    with rl_left:
        if syn is not None:
            try:
                viz_result = generate_live_rl_visualization(
                    distance=distance,
                    syndrome=syn,
                    action=action,
                    correct=correct,
                    episode=ep,
                    reward=reward,
                    error_locations=st.session_state.get('error_locations'),
                )
                live_lattice.plotly_chart(viz_result['lattice'], use_container_width=True, key=f"live_lattice_{ep}")
                
                stats = viz_result.get('stats', {})
                st.metric("Active Syndromes", f"{int(stats.get('syndrome_weight', 0))}/{stats.get('n_x_ancilla', 0) + stats.get('n_z_ancilla', 0)}")
                st.metric("Code Distance", f"d={distance}")
            except Exception as e:
                st.error(f"Live viz error: {e}")
        else:
            st.info("Start training to see live lattice")

    # ------------------------------------------------------------------
    # Live fragment — auto-refreshes every 150 ms while training is live
    # ------------------------------------------------------------------
    @st.fragment(run_every=0.15)
    def _live_rl_fragment(
        _status_box, _kpi_episode, _kpi_reward, _kpi_success, _kpi_mwpm,
        _chart_metrics, _chart_heatmap, _chart_ler, _live_lattice, _distance,
    ):
        """Drains training events and re-renders live metrics without full page reload."""
        _runner = st.session_state.runner

        # Drain queued events
        if _runner and _runner.is_running():
            events = _runner.drain()
            for ev in events:
                if isinstance(ev, MetricEvent):
                    st.session_state.history.append(ev.data)
                    st.session_state.training_episode = int(ev.data.get("episode", 0))
                elif isinstance(ev, SyndromeEvent):
                    st.session_state.latest_syndrome = ev.syndrome
                    st.session_state.latest_action   = ev.action
                    st.session_state.latest_correct  = ev.correct
                elif isinstance(ev, DoneEvent):
                    st.session_state.training_done = True
                elif isinstance(ev, ErrorEvent):
                    st.session_state.training_error = ev.message

        # Status badge
        if _runner and _runner.is_running():
            _status_box.markdown(
                '<span class="live-indicator"></span> <span class="badge-running">TRAINING LIVE</span>',
                unsafe_allow_html=True,
            )
        elif st.session_state.training_done:
            _status_box.markdown('<span class="badge-done">✅ Training complete</span>', unsafe_allow_html=True)
        elif st.session_state.training_error:
            _status_box.markdown(
                f'<span class="badge-error">❌ Error: {st.session_state.training_error}</span>',
                unsafe_allow_html=True,
            )
        else:
            _status_box.markdown("*Press ▶ Start Training to begin.*")

        # KPIs + charts
        history = st.session_state.history
        ep = st.session_state.training_episode or len(history)

        _kpi_episode.metric("Episode", f"{ep:,}")
        if history:
            latest = history[-1]
            _kpi_reward.metric("Last reward", f"{latest.get('reward', 0):.3f}")
            _kpi_success.metric("RL success", f"{latest.get('rl_success', 0):.1%}")
            _kpi_mwpm.metric("MWPM success", f"{latest.get('mwpm_success', 0):.1%}")

            _chart_metrics.plotly_chart(rl_metrics_panel(history, window=20), width="stretch")

            ler_vals = [h.get("logical_error_rate", 0.0) for h in history]
            if any(v > 0 for v in ler_vals):
                _chart_ler.plotly_chart(logical_error_rate_panel(history, window=10), width="stretch")

        syn = st.session_state.latest_syndrome
        if syn is not None and len(syn) > 0:
            _chart_heatmap.plotly_chart(
                syndrome_heatmap(
                    syn, distance=_distance,
                    title=f"Syndrome snapshot — ep {ep}  "
                          f"{'✅ correct' if st.session_state.latest_correct else '❌ error'}",
                ),
                width="stretch",
            )
            # Live lattice visualization
            try:
                action = st.session_state.latest_action
                reward = history[-1].get("reward", 0.0) if history else 0.0
                correct = st.session_state.latest_correct
                viz_result = generate_live_rl_visualization(
                    distance=_distance,
                    syndrome=syn,
                    action=action,
                    correct=correct,
                    episode=ep,
                    reward=reward,
                    error_locations=st.session_state.get("error_locations"),
                )
                _live_lattice.plotly_chart(
                    viz_result["lattice"],
                    use_container_width=True,
                    key=f"frag_lattice_{ep}",
                )
            except Exception:
                pass

    _live_rl_fragment(
        status_box, kpi_episode, kpi_reward, kpi_success, kpi_mwpm,
        chart_metrics, chart_heatmap, chart_ler, live_lattice, distance,
    )


# ==================================================================
# TAB 3 — Threshold Explorer
# ==================================================================
with tab_threshold:
    st.markdown("## 📈 Error Threshold Explorer")
    st.markdown(
        "Sweep physical error rates across code distances to locate the logical error threshold. "
        "Use **Quick sweep** for a fast preview or **Full sweep** for publication-quality results."
    )

    th_col1, th_col2 = st.columns([1, 2])
    with th_col1:
        th_decoder  = st.selectbox("Decoder", ["mwpm", "union_find"])
        th_builder  = st.selectbox("Code family", ["surface", "hexagonal", "walking", "iswap", "xyz2"])
        th_quick    = st.checkbox("Quick sweep (fast, fewer shots)", value=True)
        th_distances = st.multiselect("Distances", [3, 5, 7, 9], default=[3, 5])
        th_run_btn  = st.button("Run threshold sweep", type="primary", width="stretch")

    with th_col2:
        th_chart = st.empty()
        th_info  = st.empty()

    if th_run_btn:
        try:
            import stim  # noqa: F401
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from surface_code_in_stem.surface_code import surface_code_circuit_string
            from surface_code_in_stem.dynamic import (
                hexagonal_surface_code,
                walking_surface_code,
                iswap_surface_code,
                xyz2_hexagonal_code,
            )
            from surface_code_in_stem.decoders import MWPMDecoder, UnionFindDecoder
            from surface_code_in_stem.rl_nested_learning import _logical_error_rate

            builders = {
                "surface":   surface_code_circuit_string,
                "hexagonal": hexagonal_surface_code,
                "walking":   walking_surface_code,
                "iswap":     iswap_surface_code,
                "xyz2":      xyz2_hexagonal_code,
            }
            decoder_map = {"mwpm": MWPMDecoder(), "union_find": UnionFindDecoder()}
            builder_fn = builders.get(th_builder, surface_code_circuit_string)
            decoder    = decoder_map[th_decoder]

            p_values = np.linspace(0.003, 0.018, 7) if th_quick else np.linspace(0.001, 0.020, 13)
            shots    = 256 if th_quick else 2048
            n_workers = min(8, len(th_distances) * len(p_values))

            def _run_one(d: int, p_val: float) -> tuple[int, float, float]:
                circ_artifact = builder_fn(distance=d, rounds=d, p=float(p_val))
                circ_str = circ_artifact if isinstance(circ_artifact, str) else str(circ_artifact)
                ler = _logical_error_rate(circ_str, shots=shots, seed=7, decoder=decoder)
                return d, float(p_val), float(ler)

            data: dict[int, list[tuple[float, float]]] = {d: [] for d in th_distances}
            progress = st.progress(0.0)
            status_text = st.empty()
            total = len(th_distances) * len(p_values)
            done  = 0

            all_jobs = [(d, p_val) for d in sorted(th_distances) for p_val in p_values]

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_one, d, p_val): (d, p_val) for d, p_val in all_jobs}
                for future in as_completed(futures):
                    d_key, p_val_done, ler_val = future.result()
                    data[d_key].append((p_val_done, ler_val))
                    done += 1
                    progress.progress(done / total)
                    status_text.caption(
                        f"⚡ Parallel sweep: {done}/{total} complete "
                        f"(d={d_key}, p={p_val_done:.4f} → p_L={ler_val:.4f})"
                    )
                    # Live chart update every 4 results
                    if done % 4 == 0 or done == total:
                        partial_data = {k: sorted(v) for k, v in data.items() if v}
                        th_chart.plotly_chart(
                            threshold_figure(
                                partial_data,
                                title=f"{th_builder.title()} — {th_decoder.upper()} (sweeping…)",
                            ),
                            width="stretch",
                        )

            progress.empty()
            status_text.empty()

            # Sort each distance curve by p
            data = {k: sorted(v) for k, v in data.items()}

            # Estimate threshold (first crossing between d[0] and d[1] curves)
            thresh = None
            dists = sorted(data.keys())
            if len(dists) >= 2:
                curve1 = data[dists[0]]
                curve2 = data[dists[1]]
                for i in range(len(curve1) - 1):
                    p1,  l1  = curve1[i];    p1n, l1n = curve1[i + 1]
                    _p2, l2  = curve2[i];    _,   l2n = curve2[i + 1]
                    if (l1 - l2) * (l1n - l2n) < 0:
                        thresh = (p1 + p1n) / 2
                        break

            fig = threshold_figure(
                data,
                threshold_p=thresh,
                title=f"{th_builder.title()} code — {th_decoder.upper()} threshold",
            )
            th_chart.plotly_chart(fig, width="stretch")
            if thresh:
                th_info.success(f"✅ Estimated threshold: **p_th ≈ {thresh:.4f}**")
            else:
                th_info.info("No crossing found — try more distances or a wider p range.")

        except ImportError as exc:
            th_info.error(f"Missing dependency: {exc}")
        except Exception as exc:
            th_info.error(f"Sweep failed: {exc}")


# ==================================================================
# TAB 4 — Teraquop Footprint
# ==================================================================
with tab_footprint:
    st.markdown("## 🛰 Teraquop Physical Qubit Footprint")
    st.markdown(
        "Estimate the number of physical qubits required to achieve a target logical error rate "
        "using simplified threshold-based scaling models."
    )

    fp_col1, fp_col2 = st.columns([1, 2])

    with fp_col1:
        target_ler = st.number_input(
            "Target logical error rate (p_L)", value=1e-12, format="%.2e",
            min_value=1e-15, max_value=1e-3,
        )
        phys_p = st.number_input(
            "Physical error rate (p)", value=0.001, format="%.4f",
            min_value=1e-5, max_value=0.05,
        )
        families = st.multiselect(
            "Code families",
            ["surface", "hexagonal", "walking", "iswap", "xyz2", "toric", "hypergraph_product"],
            default=["surface", "hexagonal", "toric"],
        )
        fp_btn = st.button("Compute footprint", type="primary", width="stretch")

    with fp_col2:
        fp_chart = st.empty()
        fp_table = st.empty()

    _FOOTPRINT_PARAMS: dict[str, dict] = {
        "surface":            {"threshold": 1.0e-2, "prefactor": 0.12, "qubit_factor": 2.0},
        "hexagonal":          {"threshold": 1.5e-2, "prefactor": 0.12, "qubit_factor": 1.8},
        "walking":            {"threshold": 1.4e-2, "prefactor": 0.14, "qubit_factor": 1.8},
        "iswap":              {"threshold": 1.3e-2, "prefactor": 0.14, "qubit_factor": 1.8},
        "xyz2":               {"threshold": 1.2e-2, "prefactor": 0.15, "qubit_factor": 2.1},
        "toric":              {"threshold": 1.1e-2, "prefactor": 0.18, "qubit_factor": 2.0},
        "hypergraph_product": {"threshold": 2.5e-2, "prefactor": 0.20, "qubit_factor": 2.5},
    }

    import math as _math

    def _estimate(family, p_phys, target_ler):
        params = _FOOTPRINT_PARAMS.get(family, {})
        pth = params.get("threshold", 0.01)
        a   = params.get("prefactor", 0.15)
        qf  = params.get("qubit_factor", 2.0)
        if p_phys >= pth:
            return None
        try:
            ratio = p_phys / pth
            d = max(3, _math.ceil(_math.log(target_ler / a) / (2 * _math.log(ratio))))
            if d % 2 == 0:
                d += 1
            return {"family": family, "d": d, "physical_qubits": int(qf * d * d), "threshold": pth}
        except Exception:
            return None

    if fp_btn:
        rows = []
        for fam in families:
            est = _estimate(fam, phys_p, target_ler)
            if est:
                rows.append(est)

        if rows:
            fig = go.Figure()
            palette = list(PALETTE.values())
            for idx, r in enumerate(rows):
                fig.add_trace(go.Bar(
                    x=[r["family"]],
                    y=[r["physical_qubits"]],
                    name=r["family"],
                    marker_color=palette[idx % len(palette)],
                    text=[f"d={r['d']}  ({r['physical_qubits']:,} qubits)"],
                    textposition="outside",
                ))
            layout = dict(**PLOTLY_LAYOUT)
            layout["title"] = f"Estimated footprint for p_L ≤ {target_ler:.1e}"
            layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], title="Physical qubits")
            layout["showlegend"] = False
            layout["barmode"] = "group"
            fig.update_layout(**layout)
            fp_chart.plotly_chart(fig, width="stretch")

            import pandas as pd
            df = pd.DataFrame(rows)
            df.columns = ["Code family", "Required d", "Physical qubits", "Threshold"]
            fp_table.dataframe(
                df.style.background_gradient(subset=["Physical qubits"], cmap="Blues_r"),
                width="stretch",
            )
        else:
            fp_chart.warning(
                "Physical error rate is above threshold for all selected families — "
                "error correction is not possible in this regime."
            )
