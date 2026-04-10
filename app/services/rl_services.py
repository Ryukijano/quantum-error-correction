"""Service adapters for threshold sweeps and RL training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import json

from app.rl_runner import RLRunner
from app.services.circuit_services import estimate_logical_error_rate, service_build_circuit


ProgressCallback = Callable[[int, int], None]

_METADATA_FIELDS = (
    "backend_id",
    "backend_enabled",
    "sample_us",
    "backend_version",
    "sample_rate",
    "trace_tokens",
    "backend_chain",
    "contract_flags",
    "profiler_flags",
    "sample_trace_id",
    "details",
    "ler_ci",
    "trace_id",
    "fallback_reason",
)


@dataclass(frozen=True)
class RLPanelEvent:
    """UI-friendly transport object for training events."""

    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class RLTrainingConfig:
    mode: str
    distance: int
    rounds: int
    physical_error_rate: float
    episodes: int
    batch_size: int
    use_diffusion: bool
    use_accelerated: bool = False
    sampling_backend: str | None = None
    decoder_name: str | None = None
    enable_profile_traces: bool = False
    benchmark_probe_token: str | None = None
    protocol: str = "surface"
    syndrome_emit_every: int = 5
    seed: int = 0
    curriculum_enabled: bool = False
    curriculum_distance_start: int | None = None
    curriculum_distance_end: int | None = None
    curriculum_p_start: float | None = None
    curriculum_p_end: float | None = None
    curriculum_ramp_episodes: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    max_gradient_norm: float = 1.0
    pepg_population_size: int = 32
    pepg_learning_rate: float = 0.05
    pepg_sigma_learning_rate: float = 0.02


class RLTrainingService:
    """Adapter that hides direct RLRunner usage from the Streamlit UI."""

    def __init__(self, config: RLTrainingConfig):
        self._config = config
        self._runner = RLRunner(
            mode=config.mode,
            distance=config.distance,
            rounds=config.rounds,
            physical_error_rate=config.physical_error_rate,
            episodes=config.episodes,
            batch_size=config.batch_size,
            use_diffusion=config.use_diffusion,
            use_accelerated=config.use_accelerated,
            sampling_backend=config.sampling_backend,
            decoder_name=config.decoder_name,
            enable_profile_traces=config.enable_profile_traces,
            benchmark_probe_token=config.benchmark_probe_token,
            syndrome_emit_every=config.syndrome_emit_every,
            protocol=config.protocol,
            seed=config.seed,
            curriculum_enabled=config.curriculum_enabled,
            curriculum_distance_start=config.curriculum_distance_start,
            curriculum_distance_end=config.curriculum_distance_end,
            curriculum_p_start=config.curriculum_p_start,
            curriculum_p_end=config.curriculum_p_end,
            curriculum_ramp_episodes=config.curriculum_ramp_episodes,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            max_gradient_norm=config.max_gradient_norm,
            pepg_population_size=config.pepg_population_size,
            pepg_learning_rate=config.pepg_learning_rate,
            pepg_sigma_learning_rate=config.pepg_sigma_learning_rate,
        )

    def start(self) -> None:
        self._runner.start()

    def stop(self) -> None:
        self._runner.stop()

    def is_running(self) -> bool:
        return self._runner.is_running()

    def join(self, timeout: float = 5.0) -> None:
        self._runner.join(timeout=timeout)

    def drain_events(
        self,
        max_events: int | None = None,
        coalesce: bool = False,
    ) -> list[RLPanelEvent]:
        if coalesce:
            events = self._runner.drain_latest(max_events or 0, coalesce=True)
        else:
            events = self._runner.drain(max_events)
        return [_normalize_event(event) for event in events]

    def dropped_events(self) -> int:
        return self._runner.dropped_events()


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _coerce_metric_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _merge_metadata_fields(
    payload: Mapping[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    for field in _METADATA_FIELDS:
        if field not in metric and field in payload:
            metric[field] = payload[field]
    return metric


def metric_payload_for_history(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a metric event payload into a UI-ready history row.

    Some producers may emit metadata at top level while older payloads only place
    metadata inside ``data``. This helper reconciles both shapes.
    """

    metric = _coerce_metric_payload(payload.get("data", {}))
    return _merge_metadata_fields(payload, metric)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def history_to_download_json(history: list[dict[str, Any]]) -> str:
    """Serialize metric history for a JSON download in a Streamlit-friendly form."""

    return json.dumps([_to_jsonable(row) for row in history], indent=2)

def _normalize_event(event: Any) -> RLPanelEvent:
    kind = getattr(event, "kind", "")

    if kind == "metric":
        data = _coerce_metric_payload(getattr(event, "data", {}))
        normalized = {
            "data": data,
            "backend_id": data.get("backend_id"),
            "backend_enabled": data.get("backend_enabled"),
            "sample_us": data.get("sample_us"),
            "backend_version": data.get("backend_version"),
            "sample_rate": data.get("sample_rate"),
            "trace_tokens": data.get("trace_tokens"),
            "backend_chain": data.get("backend_chain"),
            "contract_flags": data.get("contract_flags"),
            "profiler_flags": data.get("profiler_flags"),
            "sample_trace_id": data.get("sample_trace_id"),
            "details": data.get("details"),
            "ler_ci": data.get("ler_ci"),
            "trace_id": data.get("trace_id"),
            "fallback_reason": data.get("fallback_reason"),
        }
        if isinstance(getattr(event, "data", None), Mapping):
            normalized["data"] = _merge_metadata_fields(normalized, normalized["data"])
        return RLPanelEvent("metric", normalized)

    if kind == "syndrome":
        return RLPanelEvent(
            "syndrome",
            {
                "syndrome": _to_list(getattr(event, "syndrome", [])),
                "action": _to_list(getattr(event, "action", [])),
                "correct": bool(getattr(event, "correct", False)),
                "episode": int(getattr(event, "episode", 0)),
            },
        )

    if kind == "done":
        return RLPanelEvent("done", {"total_episodes": int(getattr(event, "total_episodes", 0))})

    if kind == "error":
        return RLPanelEvent("error", {"message": str(getattr(event, "message", ""))})

    return RLPanelEvent("unknown", {"raw": repr(event)})


def run_threshold_sweep(
    *,
    distances: list[int],
    p_values: list[float],
    shots: int,
    builder_name: str,
    decoder_name: str,
    seed: int = 7,
    on_progress: ProgressCallback | None = None,
) -> dict[int, list[tuple[float, float]]]:
    """Run the circuit/decoder threshold sweep using circuit services."""

    data: dict[int, list[tuple[float, float]]] = {}
    total = len(distances) * len(p_values)
    done = 0

    for d in sorted(distances):
        samples: list[tuple[float, float]] = []
        for p in p_values:
            circuit = service_build_circuit(
                distance=d,
                rounds=d,
                p=float(p),
                builder_name=builder_name,
            )
            ler = estimate_logical_error_rate(
                circuit,
                shots=shots,
                seed=seed,
                decoder_name=decoder_name,
            )
            samples.append((float(p), float(ler)))
            done += 1
            if on_progress is not None:
                on_progress(done, total)
        data[d] = samples

    return data


def detect_threshold_crossing(data: dict[int, list[tuple[float, float]]]) -> float | None:
    """Find the first intersection between the two smallest-distance curves."""

    dists = sorted(data.keys())
    if len(dists) < 2:
        return None

    c1 = sorted(data[dists[0]], key=lambda x: x[0])
    c2 = sorted(data[dists[1]], key=lambda x: x[0])
    for i in range(min(len(c1), len(c2)) - 1):
        p1, l1 = c1[i]
        p1n, l1n = c1[i + 1]
        _, l2 = c2[i]
        _, l2n = c2[i + 1]

        if (l1 - l2) * (l1n - l2n) < 0:
            return (p1 + p1n) / 2

    return None
