"""Syndrome-Net Interactive Visualizer — Streamlit App.

Tabs:
  1. 🔬 Circuit Viewer  — Stim SVG circuit diagram + Crumble interactive editor
  2. ⚡ RL Live Training — real-time metrics + syndrome heatmap during training
  3. 📈 Threshold Explorer — interactive threshold sweep with Plotly
  4. 🛰  Teraquop Footprint — physical qubit overhead estimator
"""

from __future__ import annotations

import sys
import time
import threading
import queue
import csv
import json
from collections import defaultdict
from io import StringIO
from typing import Any
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    circuit_matchgraph_html,
    circuit_svg,
    detector_graph,
    policy_diagnostics_panel,
    logical_error_rate_panel,
    rl_metrics_panel,
    syndrome_heatmap,
    threshold_figure,
)
from app.services.circuit_services import (
    build_circuit as service_build_circuit,
    get_builder_names,
    get_decoder_names,
    get_visualizer,
    get_visualizer_names,
)
from app.services.rl_services import (
    RLTrainingConfig,
    RLTrainingService,
    history_to_download_json,
    metric_payload_for_history,
    detect_threshold_crossing,
    run_threshold_sweep,
)
from surface_code_in_stem.protocols import DEFAULT_PROTOCOL_REGISTRY

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Syndrome-Net QEC Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RL_HISTORY_MAX = 500
RL_METRIC_WINDOW = 20
RL_LERP_WINDOW = 10
BACKEND_TRACE_MATRIX_MAX_COLUMNS = 500

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
    builder_names = get_builder_names()
    if not builder_names:
        builder_names = ["surface"]
    builder_default = "surface" if "surface" in builder_names else builder_names[0]
    code_family = st.selectbox("Code family", builder_names, index=builder_names.index(builder_default))
    distance = st.slider("Code distance d", 3, 9, 3, step=2)
    rounds    = st.slider("Syndrome rounds", 1, 6, 2)
    p_phys    = st.slider("Physical error rate p", 0.001, 0.02, 0.01, step=0.001, format="%.3f")

    visualizer_names = get_visualizer_names()
    visualizer_names = sorted(set(visualizer_names + ["matchgraph"]))
    if not visualizer_names:
        visualizer_names = ["plotly", "svg", "crumble"]
    visualizer_default = "svg" if "svg" in visualizer_names else visualizer_names[0]
    visualizer_name = st.selectbox(
        "Circuit visualizer",
        visualizer_names,
        index=visualizer_names.index(visualizer_default),
    )

    st.divider()
    st.markdown("### RL training")
    rl_mode     = st.selectbox("Agent", ["ppo", "sac", "pepg"])
    available_protocols = DEFAULT_PROTOCOL_REGISTRY.list()
    protocol_name = st.selectbox("Execution protocol", available_protocols, index=available_protocols.index("surface") if "surface" in available_protocols else 0)
    episodes    = st.slider("Episodes", 50, 2000, 300, step=50)
    batch_size  = st.slider("Batch size", 16, 128, 32, step=16)
    use_diff    = st.checkbox("Use flow-matching (SAC)", value=False)

    with st.expander("Advanced backend controls", expanded=False):
        available_backends = ["auto", "stim", "qhybrid", "cuquantum", "qujax", "cudaq"]
        sampling_backend = st.selectbox(
            "Sampling backend",
            available_backends,
            index=0,
            help="Choose explicit sampling backend (auto enables dynamic fallback by capability).",
        )
        available_rl_decoders, available_rl_decoders_fallback = _get_decoder_options()
        if available_rl_decoders_fallback:
            st.caption("Using fallback decoder list (registry not ready).")
        default_decoder = "mwpm" if "mwpm" in available_rl_decoders else available_rl_decoders[0]
        rl_decoder = st.selectbox(
            "Decoder backend (for diagnostics)",
            available_rl_decoders,
            index=available_rl_decoders.index(default_decoder),
            help="Propagates through training metadata for backend experiments.",
        )
        predecoder_backend = st.selectbox(
            "Ising pre-decoder backend",
            ["identity", "disabled", "torch", "safetensors", "numpy"],
            index=0,
            help="Choose how the pre-decoder should be instantiated.",
        )
        predecoder_artifact = st.text_input(
            "Ising pre-decoder artifact path",
            value="",
            help="Optional path to a torch/safetensor/numpy artifact. Empty means identity path.",
        )
        if predecoder_artifact.strip() == "":
            predecoder_artifact = None
        predecoder_seed = st.number_input(
            "Ising pre-decoder seed",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=1,
        )
        predecoder_seed = int(predecoder_seed) if rl_decoder == "ising" else None
        enable_profile_traces = st.checkbox("Enable backend trace payload", value=False)
        benchmark_probe_token = st.text_input("Trace token (optional)", value="", help="Optional identifier for trace grouping.")
        if benchmark_probe_token == "":
            benchmark_probe_token = None
        backend_matrix_scope = st.selectbox(
            "Backend matrix scope",
            ["all episodes", "recent episodes"],
            index=0,
            help="Limit the matrix view to a recent window for faster rendering.",
        )
        backend_matrix_window = st.slider(
            "Recent backend matrix window",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Used only when 'recent episodes' is selected.",
        )
        show_profiler_panel = st.checkbox("Show profiler diagnostics panel", value=True)

    with st.expander("Advanced RL controls", expanded=False):
        seed = st.number_input("Training seed", min_value=0, max_value=1_000_000, value=42, step=1)
        curriculum_enabled = st.checkbox("Enable curriculum learning", value=True, help="Progressively adapt task difficulty during training.")
        curriculum_distance_start = st.slider(
            "Curriculum start distance",
            min_value=3,
            max_value=13,
            value=distance,
            step=2,
        )
        curriculum_distance_end = st.slider(
            "Curriculum end distance",
            min_value=3,
            max_value=13,
            value=distance,
            step=2,
        )
        curriculum_p_start = st.slider(
            "Curriculum start p",
            min_value=0.0005,
            max_value=0.05,
            value=min(max(0.001, float(p_phys * 1.25)), 0.05),
            step=0.0005,
            format="%.4f",
        )
        curriculum_p_end = st.slider(
            "Curriculum end p",
            min_value=0.0005,
            max_value=0.05,
            value=max(0.001, min(float(p_phys), 0.05)),
            step=0.0005,
            format="%.4f",
            help="Lower target error rate for easier final tasks.",
        )
        curriculum_ramp_episodes = st.slider(
            "Curriculum ramp episodes",
            min_value=0,
            max_value=max(episodes, 0),
            value=min(episodes, 2000),
            step=50,
            help="Linear curriculum only; set to 0 for a single-shot schedule.",
        )
        max_gradient_norm = st.slider(
            "Gradient clip norm",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Set to 0 to disable gradient clipping in PPO/SAC updates.",
        )
        early_stopping_patience = st.slider(
            "Early stopping patience",
            min_value=0,
            max_value=200,
            value=0,
            step=5,
            help="Set >0 to stop when validation window doesn't improve.",
        )
        early_stopping_min_delta = st.slider(
            "Early stop minimum delta",
            min_value=0.0,
            max_value=0.2,
            value=0.0,
            step=0.005,
            format="%.4f",
        )
        pepg_population_size = st.number_input(
            "PEPG population size",
            min_value=2,
            max_value=256,
            value=32,
            step=2,
            help="Even population size used by PEPG ask/tell updates.",
        )
        pepg_learning_rate = st.slider(
            "PEPG mean learning rate",
            min_value=0.001,
            max_value=0.2,
            value=0.05,
            step=0.001,
            format="%.3f",
        )
        pepg_sigma_learning_rate = st.slider(
            "PEPG sigma learning rate",
            min_value=0.0,
            max_value=0.1,
            value=0.02,
            step=0.001,
            format="%.3f",
        )

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
        history_version=0,
        history_dirty=False,
        latest_syndrome=None,
        latest_action=None,
        latest_correct=False,
        training_done=False,
        training_error=None,
        training_episode=0,
        error_locations=None,
        latest_error_locations=None,
        queue_drop_count=0,
        threshold_sweep_job=None,
        threshold_sweep_queue=None,
        threshold_sweep_progress=0.0,
        threshold_sweep_done=False,
        threshold_sweep_error=None,
        threshold_sweep_data=None,
        backend_matrix_signature=None,
        backend_profiler_signature=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Remove legacy entries carried over from older app versions.
    if "error_history" in st.session_state:
        st.session_state.pop("error_history")

    # Defensive sanitization for stale state left by prior UI versions.
    history = st.session_state.get("history")
    if isinstance(history, list) and len(history) > RL_HISTORY_MAX:
        del history[: len(history) - RL_HISTORY_MAX]
        st.session_state.history = history

    if "error_locations" in st.session_state and not isinstance(st.session_state.error_locations, dict):
        st.session_state.error_locations = None

    threshold_job = st.session_state.get("threshold_sweep_job")
    if threshold_job is not None and not getattr(threshold_job, "is_alive", lambda: False)():
        st.session_state.threshold_sweep_job = None
        st.session_state.threshold_sweep_queue = None
        if (
            st.session_state.get("threshold_sweep_error") is None
            and st.session_state.get("threshold_sweep_data") is None
        ):
            st.session_state.threshold_sweep_error = "Threshold sweep ended unexpectedly."

_init_state()


# ------------------------------------------------------------------
# Helper: build Stim circuit
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_circuit(distance: int, rounds: int, p: float, builder: str | None = None):
    try:
        return service_build_circuit(distance=distance, rounds=rounds, p=p, builder_name=builder)
    except Exception as exc:
        st.warning(f"Circuit build failed: {exc}")
        return None


@st.cache_data(show_spinner=False)
def _sample_detector_syndrome(distance: int, rounds: int, p: float, builder: str | None = None) -> np.ndarray | None:
    circuit = _get_circuit(distance=distance, rounds=rounds, p=p, builder=builder)
    if circuit is None:
        return None

    try:
        sampler = circuit.compile_detector_sampler(seed=42)
        det, _ = sampler.sample(1, separate_observables=True)
        return det[0].astype(np.int8)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _get_cached_circuit_diagnostics(
    distance: int,
    rounds: int,
    p: float,
    builder: str | None = None,
) -> dict[str, int | float | str]:
    circuit = _get_circuit(distance=distance, rounds=rounds, p=p, builder=builder)
    if circuit is None:
        return {}
    try:
        return {
            "num_qubits": int(circuit.num_qubits),
            "num_detectors": int(circuit.num_detectors),
            "num_observables": int(circuit.num_observables),
            "num_instructions": int(len(list(circuit))),
            "distance": distance,
            "rounds": rounds,
            "physical_error_rate": float(p),
        }
    except Exception:
        return {}


def _append_history(entry: dict[str, Any], history: list[dict[str, Any]], max_len: int = RL_HISTORY_MAX) -> None:
    history.append(entry)
    if len(history) > max_len:
        del history[: len(history) - max_len]


def _export_history_json(history: list[dict[str, Any]]) -> str:
    return history_to_download_json(history)


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [segment.strip() for segment in value.split(",") if segment.strip()]
    if value is None:
        return []
    return [str(value).strip()]


def _get_decoder_options() -> tuple[list[str], bool]:
    decoder_names = get_decoder_names()
    if decoder_names:
        return decoder_names, False
    return ["mwpm", "union_find"], True


def _coerce_flag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [segment.strip() for segment in value.split(",") if segment.strip()]
    return [str(value)]


def _chain_tokens(raw_chain: Any) -> list[str]:
    if isinstance(raw_chain, list):
        return [str(item).strip() for item in raw_chain if str(item).strip()]
    if isinstance(raw_chain, tuple):
        return [str(item).strip() for item in raw_chain if str(item).strip()]
    if isinstance(raw_chain, str):
        tokens = [segment.strip() for segment in raw_chain.split("->") if segment.strip()]
        if tokens:
            return tokens
        return [segment.strip() for segment in raw_chain.split(",") if segment.strip()]
    return []


def _flag_text(value: Any) -> str:
    flags = _coerce_flag_list(value)
    if not flags:
        return "none"
    return ", ".join(dict.fromkeys(flags))


def _backend_matrix_signature(
    matrix_rows: list[dict[str, Any]],
    scope: str,
    window: int,
) -> tuple[int, int, int, str, int]:
    if not matrix_rows:
        return (0, 0, 0, scope, int(window))
    first_episode = int(matrix_rows[0].get("episode", 0))
    last_episode = int(matrix_rows[-1].get("episode", first_episode))
    return (len(matrix_rows), first_episode, last_episode, str(scope), int(window))


def _backend_trace_records(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        backend_id = entry.get("backend_id")
        if not backend_id:
            continue
        trace_tokens = _coerce_str_list(entry.get("trace_tokens"))
        backend_chain_tokens = entry.get("backend_chain_tokens")
        if isinstance(backend_chain_tokens, (list, tuple)):
            backend_chain_tokens = [str(item).strip() for item in backend_chain_tokens if str(item).strip()]
        elif isinstance(backend_chain_tokens, str):
            backend_chain_tokens = _chain_tokens(backend_chain_tokens)
        else:
            backend_chain_tokens = _chain_tokens(entry.get("backend_chain")) or trace_tokens
        rows.append(
            {
                "episode": int(entry.get("episode", idx + 1)),
                "backend_id": str(backend_id),
                "backend_enabled": bool(entry.get("backend_enabled", False)),
                "sample_us": float(entry.get("sample_us") or 0.0),
                "sample_rate": float(entry.get("sample_rate") or 0.0),
                "backend_version": entry.get("backend_version"),
                "trace_tokens": trace_tokens,
                "backend_chain": entry.get("backend_chain"),
                "backend_chain_tokens": backend_chain_tokens,
                "contract_flags": entry.get("contract_flags"),
                "profiler_flags": entry.get("profiler_flags"),
                "fallback_reason": entry.get("fallback_reason"),
                "sample_trace_id": entry.get("sample_trace_id"),
                "trace_id": entry.get("trace_id"),
                "details": entry.get("details"),
            }
        )
    return rows


def _backend_trace_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = [
        "episode",
        "backend_id",
        "backend_enabled",
        "sample_us",
        "sample_rate",
        "backend_version",
        "backend_chain_tokens",
        "backend_chain",
        "contract_flags",
        "profiler_flags",
        "fallback_reason",
        "sample_trace_id",
        "trace_id",
        "trace_tokens",
        "details",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        row_copy = dict(row)
        row_copy["backend_enabled"] = int(bool(row_copy.get("backend_enabled")))
        tokens = row_copy.get("trace_tokens")
        if isinstance(tokens, list):
            row_copy["trace_tokens"] = ",".join(tokens)
        chain_tokens = row_copy.get("backend_chain_tokens")
        if isinstance(chain_tokens, list):
            row_copy["backend_chain_tokens"] = json.dumps(chain_tokens)
        else:
            row_copy["backend_chain_tokens"] = json.dumps(_coerce_str_list(chain_tokens))
        details = row_copy.get("details")
        if isinstance(details, (dict, list)):
            try:
                row_copy["details"] = json.dumps(details, sort_keys=True)
            except TypeError:
                row_copy["details"] = str(details)
        elif details is None:
            row_copy["details"] = ""
        else:
            row_copy["details"] = str(details)
        writer.writerow(row_copy)
    return buffer.getvalue()


def _backend_profiler_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    aggregates = defaultdict(
        lambda: {
            "episodes": 0,
            "fallback_episodes": 0,
            "sample_us_total": 0.0,
            "sample_rate_total": 0.0,
            "trace_id_rows": 0,
            "contract_disabled_rows": 0,
            "profiler_missing_rows": 0,
            "chain_values": set(),
            "backend_chain_tokens": set(),
            "sample_ids": set(),
        }
    )
    for row in rows:
        backend_id = str(row.get("backend_id") or "unknown")
        state = aggregates[backend_id]
        state["episodes"] += 1
        if not bool(row.get("backend_enabled", False)):
            state["fallback_episodes"] += 1
        state["sample_us_total"] += float(row.get("sample_us") or 0.0)
        state["sample_rate_total"] += float(row.get("sample_rate") or 0.0)
        if row.get("sample_trace_id") is not None:
            state["trace_id_rows"] += 1
            if row.get("sample_trace_id") not in (None, ""):
                state["sample_ids"].add(str(row.get("sample_trace_id")))
        flags = _coerce_flag_list(row.get("contract_flags"))
        if any("disabled" in flag for flag in flags):
            state["contract_disabled_rows"] += 1
        profiler_flags = _coerce_flag_list(row.get("profiler_flags"))
        if not any("sample_trace_present" in flag for flag in profiler_flags):
            state["profiler_missing_rows"] += 1
        chain = row.get("backend_chain")
        if isinstance(chain, str) and chain:
            state["chain_values"].add(chain)
        chain_tokens = row.get("backend_chain_tokens")
        if isinstance(chain_tokens, (list, tuple)):
            for token in chain_tokens:
                token_text = str(token).strip()
                if token_text:
                    state["backend_chain_tokens"].add(token_text)
        else:
            for token in _chain_tokens(chain_tokens):
                if token:
                    state["backend_chain_tokens"].add(token)

    summary_rows: list[dict[str, Any]] = []
    for backend_id in sorted(aggregates):
        state = aggregates[backend_id]
        episodes = int(state["episodes"])
        fallback_episodes = int(state["fallback_episodes"])
        trace_id_rows = int(state["trace_id_rows"])
        profiler_missing_rows = int(state["profiler_missing_rows"])
        summary_rows.append(
            {
                "backend_id": backend_id,
                "episodes": episodes,
                "fallback_episodes": fallback_episodes,
                "fallback_ratio": round(fallback_episodes / episodes, 4) if episodes else 0.0,
                "avg_sample_us": round(state["sample_us_total"] / episodes, 6) if episodes else 0.0,
                "avg_sample_rate": round(state["sample_rate_total"] / episodes, 6) if episodes else 0.0,
                "trace_id_rows": trace_id_rows,
                "trace_id_coverage": round(trace_id_rows / episodes, 4) if episodes else 0.0,
                "unique_chain_paths": len(state["chain_values"]),
                "unique_trace_id_count": len(state["sample_ids"]),
                "contract_disabled_rows": state["contract_disabled_rows"],
                "profiler_missing_rows": profiler_missing_rows,
                "profiler_trace_coverage": round(
                    1.0 - (profiler_missing_rows / episodes), 4
                ) if episodes else 0.0,
                "backend_chain_tokens": "; ".join(sorted(state["backend_chain_tokens"])) if state["backend_chain_tokens"] else "",
            }
        )
    return summary_rows


def _backend_profiler_summary_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = [
        "backend_id",
        "episodes",
        "fallback_episodes",
        "fallback_ratio",
        "avg_sample_us",
        "avg_sample_rate",
        "trace_id_rows",
        "trace_id_coverage",
        "unique_chain_paths",
        "unique_trace_id_count",
        "contract_disabled_rows",
        "profiler_missing_rows",
        "profiler_trace_coverage",
        "backend_chain_tokens",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _backend_trace_matrix_figure(history: list[dict[str, Any]]) -> go.Figure:
    records = _backend_trace_records(history)
    if not records:
        return go.Figure(layout_title_text="Backend trace matrix")
    if len(records) > BACKEND_TRACE_MATRIX_MAX_COLUMNS:
        records = records[-BACKEND_TRACE_MATRIX_MAX_COLUMNS:]
    episodes = [int(entry.get("episode", idx + 1)) for idx, entry in enumerate(records)]
    backends = sorted({str(entry.get("backend_id", "unknown")) for entry in records})
    if "None" in backends:
        backends.remove("None")
    if not backends:
        return go.Figure(layout_title_text="Backend trace matrix")

    backend_index = {backend: row for row, backend in enumerate(backends)}
    sample_matrix: list[list[float | None]] = [[float("nan") for _ in episodes] for _ in backends]
    rate_matrix: list[list[float | None]] = [[float("nan") for _ in episodes] for _ in backends]
    fallback_matrix: list[list[int]] = [[0 for _ in episodes] for _ in backends]
    fallback_text: list[list[str]] = [["" for _ in episodes] for _ in backends]
    for col, entry in enumerate(records):
        backend = str(entry.get("backend_id", "unknown"))
        if backend in backend_index:
            sample_matrix[backend_index[backend]][col] = float(entry.get("sample_us", 0.0) or 0.0)
            rate_matrix[backend_index[backend]][col] = float(entry.get("sample_rate", 0.0) or 0.0)
            if not bool(entry.get("backend_enabled", True)):
                fallback_matrix[backend_index[backend]][col] = 1
                fallback_text[backend_index[backend]][col] = "F"

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[
        "Sampling latency by backend (µs)",
        "Sampling throughput by backend (samples/s)",
        "Fallback status by backend (1 = fallback path)",
    ])
    fig.add_trace(
        go.Heatmap(
            z=sample_matrix,
            x=episodes,
            y=backends,
            colorscale="Agsunset",
            hovertemplate="backend=%{y}<br>episode=%{x}<br>sample time=%{z:.3f} µs<extra></extra>",
            colorbar=dict(title_text="Sampling time (µs)"),
            showscale=True,
            name="Sampling time (µs)",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=rate_matrix,
            x=episodes,
            y=backends,
            colorscale="Blues",
            hovertemplate="backend=%{y}<br>episode=%{x}<br>sampling rate=%{z:.2f} /s<extra></extra>",
            colorbar=dict(title_text="Samples per second"),
            showscale=True,
            name="Sampling rate (1/s)",
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=fallback_matrix,
            x=episodes,
            y=backends,
            zmin=0,
            zmax=1,
            text=fallback_text,
            texttemplate="%{text}",
            textfont=dict(color="#ffffff", size=10),
            colorscale=[(0.0, "rgba(32, 48, 80, 0.08)"), (1.0, "rgba(255, 102, 102, 0.95)")],
            hovertemplate="backend=%{y}<br>episode=%{x}<br>fallback path=%{z:.0f}<extra></extra>",
            colorbar=dict(
                title_text="Fallback event",
                tickvals=[0, 1],
                ticktext=["No", "Yes"],
            ),
            showscale=True,
            name="Fallback path",
            showlegend=True,
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        title="Backend × Episode trace matrix",
        xaxis_title="Episode",
        yaxis_title="Backend (sampling backend)",
        template="plotly_white",
        height=max(360, 90 * len(backends)),
        margin=dict(t=90, l=90, r=20, b=40),
        legend=dict(orientation="h", y=1.03, x=0.0, xanchor="left"),
    )
    fig.update_xaxes(
        tickmode="linear",
        dtick=max(1, len(episodes) // 8),
        title_text="Episode",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickmode="linear",
        dtick=max(1, len(episodes) // 8),
        title_text="Episode",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        tickmode="linear",
        dtick=max(1, len(episodes) // 8),
        title_text="Episode",
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Backend", row=1, col=1)
    fig.update_yaxes(title_text="Backend", row=2, col=1)
    fig.update_yaxes(title_text="Backend", row=3, col=1)
    return fig


def _start_threshold_sweep_job(
    *,
    distances: list[int],
    p_values: list[float],
    shots: int,
    builder: str,
    decoder: str,
    seed: int = 7,
) -> None:
    running = st.session_state.threshold_sweep_job
    if running is not None and getattr(running, "is_alive", lambda: False)():
        return

    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1024)
    st.session_state.threshold_sweep_queue = event_queue
    st.session_state.threshold_sweep_progress = 0.0
    st.session_state.threshold_sweep_done = False
    st.session_state.threshold_sweep_error = None
    st.session_state.threshold_sweep_data = None

    def worker() -> None:
        try:
            data = run_threshold_sweep(
                distances=sorted(distances),
                p_values=p_values,
                shots=shots,
                builder_name=builder,
                decoder_name=decoder,
                seed=seed,
                on_progress=lambda done, total: event_queue.put(("progress", done / total if total else 0.0)),
            )
            event_queue.put(("result", data))
        except Exception as exc:
            event_queue.put(("error", str(exc)))

    job = threading.Thread(target=worker, daemon=True)
    st.session_state.threshold_sweep_job = job
    job.start()


def _poll_threshold_sweep_job() -> None:
    job = st.session_state.threshold_sweep_job
    event_queue = st.session_state.threshold_sweep_queue
    if job is None or event_queue is None:
        return

    while True:
        try:
            kind, payload = event_queue.get_nowait()
        except queue.Empty:
            break
        if kind == "progress":
            st.session_state.threshold_sweep_progress = float(payload)
        elif kind == "result":
            st.session_state.threshold_sweep_data = payload
            st.session_state.threshold_sweep_done = True
            st.session_state.threshold_sweep_progress = 1.0
            st.session_state.threshold_sweep_job = None
            st.session_state.threshold_sweep_queue = None
        elif kind == "error":
            st.session_state.threshold_sweep_error = str(payload)
            st.session_state.threshold_sweep_done = True
            st.session_state.threshold_sweep_job = None
            st.session_state.threshold_sweep_queue = None

    if job is not None and not job.is_alive() and st.session_state.threshold_sweep_queue is not None:
        st.session_state.threshold_sweep_job = None
        if st.session_state.threshold_sweep_data is None and st.session_state.threshold_sweep_error is None:
            st.session_state.threshold_sweep_error = "Threshold sweep ended unexpectedly."


def _reset_rl_runtime_state() -> None:
    st.session_state.history = []
    st.session_state.history_version = 0
    st.session_state.history_dirty = False
    st.session_state.latest_syndrome = None
    st.session_state.latest_action = None
    st.session_state.latest_correct = False
    st.session_state.training_done = False
    st.session_state.training_error = None
    st.session_state.training_episode = 0
    st.session_state.latest_error_locations = None
    st.session_state.error_locations = None
    st.session_state.queue_drop_count = 0
    st.session_state.pop("error_history", None)
    st.session_state.backend_matrix_signature = None
    st.session_state.backend_profiler_signature = None


def _build_rl_training_config(
    *,
    sampling_backend: str,
    benchmark_probe_token: str | None,
    rl_mode: str,
    distance: int,
    rounds: int,
    p_phys: float,
    episodes: int,
    batch_size: int,
    use_diff: bool,
    rl_decoder: str | None,
    enable_profile_traces: bool,
    protocol_name: str,
    predecoder_backend: str,
    predecoder_artifact: str | None,
    predecoder_seed: int | None,
    seed: int,
    curriculum_enabled: bool,
    curriculum_distance_start: int,
    curriculum_distance_end: int,
    curriculum_p_start: float,
    curriculum_p_end: float,
    curriculum_ramp_episodes: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    max_gradient_norm: float,
    pepg_population_size: int,
    pepg_learning_rate: float,
    pepg_sigma_learning_rate: float,
) -> RLTrainingConfig:
    from surface_code_in_stem.accelerators import qhybrid_backend

    capability = qhybrid_backend.probe_capability()
    auto_use_accelerated = bool(capability.get("enabled", False))
    resolved_sampling_backend = None if sampling_backend == "auto" else sampling_backend
    resolved_use_accelerated = auto_use_accelerated if sampling_backend == "auto" else sampling_backend != "stim"
    token = benchmark_probe_token.strip() if isinstance(benchmark_probe_token, str) else None
    trace_token = token if token else None

    return RLTrainingConfig(
        mode=rl_mode,
        distance=distance,
        rounds=rounds,
        physical_error_rate=p_phys,
        episodes=episodes,
        batch_size=batch_size,
        use_diffusion=use_diff,
        use_accelerated=resolved_use_accelerated,
        sampling_backend=resolved_sampling_backend,
        decoder_name=rl_decoder,
        enable_profile_traces=enable_profile_traces,
        benchmark_probe_token=trace_token,
        protocol=protocol_name,
        syndrome_emit_every=max(1, episodes // 40),
        seed=seed,
        predecoder_backend=predecoder_backend,
        predecoder_artifact=predecoder_artifact,
        predecoder_seed=predecoder_seed,
        curriculum_enabled=curriculum_enabled,
        curriculum_distance_start=curriculum_distance_start,
        curriculum_distance_end=curriculum_distance_end,
        curriculum_p_start=curriculum_p_start,
        curriculum_p_end=curriculum_p_end,
        curriculum_ramp_episodes=curriculum_ramp_episodes,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        max_gradient_norm=max_gradient_norm,
        pepg_population_size=int(pepg_population_size),
        pepg_learning_rate=float(pepg_learning_rate),
        pepg_sigma_learning_rate=float(pepg_sigma_learning_rate),
    )


def _drain_rl_events(runner: RLTrainingService) -> bool:
    events = runner.drain_events(max_events=80, coalesce=True)
    dropped_events = runner.dropped_events()
    if dropped_events != st.session_state.queue_drop_count:
        st.session_state.queue_drop_count = dropped_events

    metrics_updated = False
    for ev in events:
        if ev.kind == "metric":
            data = metric_payload_for_history(ev.payload)
            _append_history(data, st.session_state.history)
            st.session_state.history_version += 1
            st.session_state.history_dirty = True
            metrics_updated = True
            st.session_state.training_episode = int(data.get("episode", 0))
        elif ev.kind == "syndrome":
            st.session_state.latest_syndrome = np.asarray(ev.payload.get("syndrome", []), dtype=np.int8)
            st.session_state.latest_action = np.asarray(ev.payload.get("action", []), dtype=np.int8)
            st.session_state.latest_correct = bool(ev.payload.get("correct", False))
            st.session_state.history_dirty = st.session_state.history_dirty or metrics_updated
        elif ev.kind == "done":
            st.session_state.training_done = True
        elif ev.kind == "error":
            st.session_state.training_error = ev.payload.get("message", "")

    return metrics_updated


def _render_circuit_panel(
    circuit,
    visualizer: str,
    diagram_style: str,
    *,
    height: int = 540,
    chart_key: str | None = None,
):
    if visualizer == "matchgraph":
        html_str = circuit_matchgraph_html(circuit)
        st.components.v1.html(html_str, height=height, scrolling=True)
        return

    if visualizer == "plotly":
        fig = get_visualizer("plotly").render_circuit(circuit)
        st.plotly_chart(
            fig,
            width="stretch",
            key=chart_key or f"circuit_plotly_{visualizer}_{height}",
        )
        return

    if visualizer == "crumble":
        html_str = circuit_interactive_html(circuit)
        st.components.v1.html(html_str, height=height, scrolling=True)
        return

    if visualizer == "svg":
        svg_str = circuit_svg(circuit, diagram_style)
    else:
        try:
            svg_str = str(get_visualizer(visualizer).render_circuit(circuit))
        except Exception:
            svg_str = circuit_svg(circuit, diagram_style)

    st.markdown(f'<div class="circuit-panel">{svg_str}</div>', unsafe_allow_html=True)


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

    circuit = _get_circuit(distance, rounds, p_phys, code_family)
    if circuit is None:
        st.error("Could not build circuit — check that `stim` is installed.")
    else:
        col_svg, col_graph = st.columns([3, 2])

        with col_svg:
            st.markdown("#### Circuit diagram")
            _render_circuit_panel(
                circuit,
                visualizer_name,
                diagram_type,
                height=540,
                chart_key=f"circuit_plotly_main_{distance}_{rounds}_{p_phys}_{code_family}_{visualizer_name}_{diagram_type}",
            )

        with col_graph:
            st.markdown("#### Detector error graph")
            st.caption("Nodes = detectors · Edges = DEM connections")

            # Sample one syndrome to highlight fired detectors
            syn = _sample_detector_syndrome(distance=distance, rounds=rounds, p=p_phys, builder=code_family)

            dem_fig = detector_graph(circuit, syndrome=syn, title="DEM — sample syndrome")
            st.plotly_chart(
                dem_fig,
                width="stretch",
                key=f"dem_graph_main_{distance}_{rounds}_{p_phys}_{code_family}",
            )

        st.divider()
        st.markdown("#### Syndrome heatmap (single shot)")
        if syn is not None:
            hm_fig = syndrome_heatmap(syn, distance=distance)
            st.plotly_chart(
                hm_fig,
                width="stretch",
                key=f"syndrome_heatmap_main_{distance}_{rounds}_{p_phys}_{code_family}",
            )
            fired = int(np.sum(syn))
            st.caption(f"🔴 {fired} / {len(syn)} detectors fired in this shot.")

        with st.expander("Circuit statistics"):
            diagnostics = _get_cached_circuit_diagnostics(
                distance=distance,
                rounds=rounds,
                p=p_phys,
                builder=code_family,
            )
            if diagnostics:
                st.json(diagnostics)
            else:
                st.warning("Circuit diagnostics unavailable")


# ==================================================================
# TAB 2 — RL Live Training
# ==================================================================
with tab_rl:
    st.markdown("## ⚡ Live RL Training")
    st.markdown(
        "Train Transformer-PPO (discrete decoding), Continuous SAC (calibration), or PEPG (control) "
        "and watch metrics + syndrome snapshots update in real time."
    )

    runner: RLTrainingService | None = st.session_state.runner

    ctrl_col, status_col = st.columns([2, 3])
    with ctrl_col:
        start_btn = st.button("▶ Start Training", type="primary", width="stretch")
        stop_btn  = st.button("⏹ Stop", width="stretch")

    with status_col:
        status_box = st.empty()

    if start_btn:
        if runner and runner.is_running():
            st.warning("Training is already running.")
        elif rl_decoder not in available_rl_decoders:
            st.error(f"Selected decoder '{rl_decoder}' is not available.")
        else:
            _reset_rl_runtime_state()
            service_config = _build_rl_training_config(
                sampling_backend=sampling_backend,
                benchmark_probe_token=benchmark_probe_token,
                rl_mode=rl_mode,
                distance=distance,
                rounds=rounds,
                p_phys=p_phys,
                episodes=episodes,
                batch_size=batch_size,
                use_diff=use_diff,
                rl_decoder=rl_decoder,
                enable_profile_traces=enable_profile_traces,
                protocol_name=protocol_name,
                predecoder_backend=predecoder_backend,
                predecoder_artifact=predecoder_artifact,
                predecoder_seed=predecoder_seed,
                seed=seed,
                curriculum_enabled=curriculum_enabled,
                curriculum_distance_start=curriculum_distance_start,
                curriculum_distance_end=curriculum_distance_end,
                curriculum_p_start=curriculum_p_start,
                curriculum_p_end=curriculum_p_end,
                curriculum_ramp_episodes=curriculum_ramp_episodes,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                max_gradient_norm=max_gradient_norm,
                pepg_population_size=pepg_population_size,
                pepg_learning_rate=pepg_learning_rate,
                pepg_sigma_learning_rate=pepg_sigma_learning_rate,
            )
            st.session_state.runner = RLTrainingService(service_config)
            st.session_state.runner.start()

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
    chart_policy    = st.empty()
    chart_backend_matrix = st.empty()

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
        circuit = _get_circuit(distance, rounds, p_phys, code_family)
        if circuit is None:
            st.error("Could not build circuit")
        else:
            _render_circuit_panel(
                circuit,
                visualizer_name,
                diagram_type,
                height=480,
                chart_key=f"circuit_plotly_rl_{distance}_{rounds}_{p_phys}_{code_family}_{visualizer_name}_{diagram_type}",
            )
            
            st.caption("Detector graph and heatmap below")
            syn = _sample_detector_syndrome(distance=distance, rounds=rounds, p=p_phys, builder=code_family)
            if syn is not None:
                try:
                    dem_fig = detector_graph(circuit, syndrome=syn, title="DEM — sample")
                    st.plotly_chart(
                        dem_fig,
                        use_container_width=True,
                        key=f"dem_graph_rl_{distance}_{rounds}_{p_phys}_{code_family}",
                    )
                    hm_fig = syndrome_heatmap(syn, distance=distance, title="Syndrome Heatmap")
                    st.plotly_chart(
                        hm_fig,
                        use_container_width=True,
                        key=f"syndrome_heatmap_rl_{distance}_{rounds}_{p_phys}_{code_family}",
                    )
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
    # Poll loop — runs on each Streamlit rerun cycle
    # ------------------------------------------------------------------
    runner = st.session_state.runner

    if runner and runner.is_running():
        status_box.markdown(
            '<span class="live-indicator"></span> <span class="badge-running">TRAINING LIVE</span>',
            unsafe_allow_html=True,
        )
        _drain_rl_events(runner)

        # Schedule rerun to keep polling
        time.sleep(0.05)
        st.rerun()

    elif st.session_state.training_done:
        status_box.markdown(
            '<span class="badge-done">✅ Training complete</span>', unsafe_allow_html=True
        )
    elif st.session_state.training_error:
        status_box.markdown(
            f'<span class="badge-error">❌ Error: {st.session_state.training_error}</span>',
            unsafe_allow_html=True,
        )
    else:
        status_box.markdown("*Press ▶ Start Training to begin.*")

    # ------------------------------------------------------------------
    # Render charts from session history
    # ------------------------------------------------------------------
    history = st.session_state.history
    ep = st.session_state.training_episode or (len(history))

    kpi_episode.metric("Episode", f"{ep:,}")
    if history:
        latest = history[-1]
        backend_rows = _backend_trace_records(history)
        if backend_matrix_scope == "recent episodes":
            window_size = max(1, int(backend_matrix_window))
            matrix_rows = backend_rows[-window_size:]
        else:
            matrix_rows = backend_rows
        profiler_summary = (
            _backend_profiler_summary(matrix_rows) if show_profiler_panel else []
        )
        display_window = backend_matrix_window if backend_matrix_scope == "recent episodes" else 0
        matrix_signature = _backend_matrix_signature(matrix_rows, backend_matrix_scope, display_window)
        should_render_matrix = (
            st.session_state.history_dirty
            or st.session_state.get("backend_matrix_signature") != matrix_signature
        )
        profiler_signature = (
            matrix_signature,
            bool(show_profiler_panel),
            len(matrix_rows),
        )
        should_render_profiler = (
            show_profiler_panel
            and (
                st.session_state.history_dirty
                or st.session_state.get("backend_profiler_signature") != profiler_signature
            )
        )
        latest_backend_enabled = bool(latest.get("backend_enabled", False))
        latest_fallback_mode = "Primary backend" if latest_backend_enabled else "Fallback path"
        latest_fallback_reason = latest.get("fallback_reason")
        if latest_backend_enabled and not latest_fallback_reason:
            latest_fallback_reason = "No fallback used"

        kpi_reward.metric("Last reward", f"{latest.get('reward', 0):.3f}")
        kpi_success.metric("RL success", f"{latest.get('rl_success', 0):.1%}")
        kpi_mwpm.metric("MWPM success", f"{latest.get('mwpm_success', 0):.1%}")
        backend_panel = st.expander("Backend trace (latest metric)", expanded=False)
        backend_panel.markdown(f"**Backend mode:** {latest_fallback_mode}")
        if not latest_backend_enabled and latest_fallback_reason:
            backend_panel.caption(f"Fallback reason: {latest_fallback_reason}")
        backend_panel.json(
            {
                "backend_id": latest.get("backend_id"),
                "backend_mode": latest_fallback_mode,
                "backend_enabled": latest_backend_enabled,
                "backend_version": latest.get("backend_version"),
                "sample_us": latest.get("sample_us"),
                "sample_rate": latest.get("sample_rate"),
                "trace_tokens": latest.get("trace_tokens"),
                "backend_chain": latest.get("backend_chain"),
                "contract_flags": _flag_text(latest.get("contract_flags")),
                "profiler_flags": _flag_text(latest.get("profiler_flags")),
                "fallback_reason": latest_fallback_reason,
                "trace_id": latest.get("trace_id"),
                "sample_trace_id": latest.get("sample_trace_id"),
                "details": latest.get("details"),
                "ler_ci": latest.get("ler_ci"),
            }
        )
        backend_panel.download_button(
            "Download backend trace (JSON)",
            data=_export_history_json(matrix_rows),
            file_name="rl_backend_trace.json",
            mime="application/json",
            use_container_width=False,
            key="rl-backend-trace-json",
        )
        backend_panel.download_button(
            "Download backend trace (CSV)",
            data=_backend_trace_csv(matrix_rows),
            file_name="rl_backend_trace.csv",
            mime="text/csv",
            use_container_width=False,
            key="rl-backend-trace-csv",
        )
        if should_render_profiler:
            backend_panel.markdown("#### Profiler diagnostics")
            if profiler_summary:
                profiler_cols = backend_panel.columns(3)
                profiler_cols[0].metric("Backends", len(profiler_summary))
                profiler_cols[1].metric(
                    "Total backend rows",
                    sum(row.get("episodes", 0) for row in profiler_summary),
                )
                profiler_cols[2].metric(
                    "Max fallback ratio",
                    f"{max((row.get('fallback_ratio', 0.0) or 0.0) for row in profiler_summary):.1%}",
                )
                backend_panel.dataframe(
                    [
                        {
                            "Backend": row.get("backend_id"),
                            "Episodes": row.get("episodes"),
                            "Fallback ratio": f"{row.get('fallback_ratio', 0.0):.1%}",
                            "Avg sample (µs)": f"{row.get('avg_sample_us', 0.0):.4f}",
                            "Avg rate (/s)": f"{row.get('avg_sample_rate', 0.0):.3f}",
                            "Trace coverage": f"{row.get('profiler_trace_coverage', 0.0):.1%}",
                            "Trace ID coverage": f"{row.get('trace_id_coverage', 0.0):.1%}",
                            "Fallback rows": row.get("fallback_episodes", 0),
                            "Contract disabled rows": row.get("contract_disabled_rows", 0),
                            "Profiler missing rows": row.get("profiler_missing_rows", 0),
                            "Unique trace IDs": row.get("unique_trace_id_count", 0),
                        }
                        for row in profiler_summary
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                backend_panel.download_button(
                    "Download profiler summary (JSON)",
                    data=_export_history_json(profiler_summary),
                    file_name="rl_backend_profiler_summary.json",
                    mime="application/json",
                    use_container_width=False,
                    key="rl-backend-profiler-summary-json",
                )
                backend_panel.download_button(
                    "Download profiler summary (CSV)",
                    data=_backend_profiler_summary_csv(profiler_summary),
                    file_name="rl_backend_profiler_summary.csv",
                    mime="text/csv",
                    use_container_width=False,
                    key="rl-backend-profiler-summary-csv",
                )
                st.session_state.backend_profiler_signature = profiler_signature
            else:
                backend_panel.info("Profiler diagnostics are not available for current dataset.")
                st.session_state.backend_profiler_signature = profiler_signature
        if show_profiler_panel:
            if not should_render_profiler:
                backend_panel.caption("Profiler diagnostics are up to date.")
        refresh_charts = bool(st.session_state.history_dirty)
        should_render_policy = refresh_charts or st.session_state.get("history_version", 0) == 0
        if st.session_state.history_dirty:
            chart_metrics.plotly_chart(
                rl_metrics_panel(
                    history,
                    window=min(RL_METRIC_WINDOW, max(1, len(history))),
                ),
                width="stretch",
            )

        if should_render_matrix:
            chart_backend_matrix.plotly_chart(
                _backend_trace_matrix_figure(matrix_rows),
                width="stretch",
            )
            st.session_state.backend_matrix_signature = matrix_signature

        if should_render_policy and any(
            h.get("policy_loss", 0)
            or h.get("value_loss", 0)
            or h.get("alpha_loss", 0)
            or h.get("policy_updates", 0)
            or h.get("sigma_mean", 0)
            for h in history
        ):
            chart_policy.plotly_chart(
                policy_diagnostics_panel(
                    history,
                    window=min(RL_METRIC_WINDOW, max(1, len(history))),
                ),
                width="stretch",
            )

        ler_vals = [h.get("logical_error_rate", 0.0) for h in history]
        if any(v > 0 for v in ler_vals):
            if refresh_charts or st.session_state.get("history_version", 0) == 0:
                chart_ler.plotly_chart(
                    logical_error_rate_panel(
                        history,
                        window=min(RL_LERP_WINDOW, max(1, len(history))),
                    ),
                    width="stretch",
                )
        if refresh_charts:
            st.session_state.history_dirty = False

    syn = st.session_state.latest_syndrome
    if syn is not None and len(syn) > 0:
        hm = syndrome_heatmap(
            syn, distance=distance,
            title=f"Syndrome snapshot — ep {ep}  "
                  f"{'✅ correct' if st.session_state.latest_correct else '❌ error'}",
        )
        chart_heatmap.plotly_chart(hm, width="stretch")


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
        available_threshold_decoders, available_threshold_decoders_fallback = _get_decoder_options()
        if available_threshold_decoders_fallback:
            st.caption("Using fallback decoder list (registry not ready).")
        th_decoder = st.selectbox("Decoder", available_threshold_decoders)
        th_builder = st.selectbox("Code family", get_builder_names())
        th_quick    = st.checkbox("Quick sweep (fast, fewer shots)", value=True)
        th_distances = st.multiselect("Distances", [3, 5, 7, 9], default=[3, 5])
        th_run_btn  = st.button("Run threshold sweep", type="primary", width="stretch")

    with th_col2:
        th_chart = st.empty()
        th_info  = st.empty()
        th_progress = st.progress(st.session_state.threshold_sweep_progress)
        th_status = st.empty()

    _poll_threshold_sweep_job()

    if th_run_btn:
        if st.session_state.threshold_sweep_job is not None:
            th_info.warning("A threshold sweep is already running.")
        elif th_decoder not in available_threshold_decoders:
            th_info.error(f"Selected decoder '{th_decoder}' is not available.")
        else:
            p_values = np.linspace(0.003, 0.018, 7) if th_quick else np.linspace(0.001, 0.020, 13)
            shots = 256 if th_quick else 2048
            _start_threshold_sweep_job(
                distances=sorted(th_distances),
                p_values=[float(x) for x in list(p_values)],
                shots=shots,
                builder=th_builder,
                decoder=th_decoder,
                seed=7,
            )

    # While the background job is alive, keep the UI responsive by polling and rerunning.
    if st.session_state.threshold_sweep_job is not None:
        th_progress.progress(st.session_state.threshold_sweep_progress)
        pct = float(st.session_state.threshold_sweep_progress) * 100
        th_status.info(f"Sweeping threshold space: {pct:.1f}% complete")
        st.rerun()

    if st.session_state.threshold_sweep_error is not None:
        th_progress.empty()
        th_status.empty()
        th_info.error(f"Sweep failed: {st.session_state.threshold_sweep_error}")
        st.session_state.threshold_sweep_error = None
        st.session_state.threshold_sweep_done = False
        st.session_state.threshold_sweep_data = None

    data = st.session_state.threshold_sweep_data
    if data:
        # Estimate threshold (first crossing between d[0] and d[1] curves)
        thresh = detect_threshold_crossing(data)

        fig = threshold_figure(
            data,
            threshold_p=thresh,
            title=f"{th_builder.title()} code — {th_decoder.upper()} threshold",
        )
        th_chart.plotly_chart(fig, width="stretch")
        if thresh:
            th_info.success(f"Estimated threshold: **p_th ≈ {thresh:.4f}**")
        else:
            th_status.info("Sweep finished; unable to estimate threshold from sampled points.")

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
