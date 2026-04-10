"""Benchmark and emit simple runtime contract metrics for CI regression checks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import app.rl_runner as rl_runner
from app.rl_strategies import _BaseTrainingStrategy, TrainingStrategyRegistry
from surface_code_in_stem.protocols import DEFAULT_PROTOCOL_REGISTRY
from surface_code_in_stem.rl_control.envs import EnvBuilderRegistry


class BurstStrategy(_BaseTrainingStrategy):
    name = "bench"
    environment_name = "bench"

    def run(self, runtime) -> None:
        metric_profile_fields = {
            "backend_id": "stim",
            "backend_enabled": True,
            "sample_us": 12.5,
            "backend_version": "runtime-contract",
            "fallback_reason": None,
            "trace_id": "runtime-contract",
            "sample_trace_id": "runtime-contract",
            "ler_ci": {"lower": 0.0, "upper": 0.3},
            "sample_rate": 8200.0,
            "trace_tokens": ["runtime-contract", "stim"],
            "backend_chain": "runtime-contract->stim",
            "contract_flags": "backend_enabled,contract_met",
            "profiler_flags": "sample_trace_present,trace_chain_recorded",
            "details": {"source": "runtime-contract", "phase": "ci"},
        }
        for i in range(1, 121):
            metric_payload = dict(
                episode=i,
                reward=float(i % 7),
                rl_success=0.5,
                mwpm_success=0.4,
                **metric_profile_fields,
            )
            runtime.emit_metric(metric_payload)
            runtime.emit_syndrome(
                np.array([i % 2], dtype=np.int8),
                np.array([i % 4], dtype=np.int8),
                i % 2 == 0,
                i,
            )
        runtime.emit_done(120)


class BenchBuilder:
    name = "bench"

    def build(self, context):
        return context


def _build_runner() -> rl_runner.RLRunner:
    strategy_registry = TrainingStrategyRegistry()
    strategy_registry.register(BurstStrategy())

    env_builder_registry = EnvBuilderRegistry()
    env_builder_registry.register(BenchBuilder())

    return rl_runner.RLRunner(
        mode="bench",
        episodes=120,
        distance=3,
        rounds=2,
        physical_error_rate=0.01,
        batch_size=32,
        syndrome_emit_every=20,
        strategy_registry=strategy_registry,
        env_builder_registry=env_builder_registry,
        protocol_registry=DEFAULT_PROTOCOL_REGISTRY,
        protocol="surface",
    )


def _extract_kind_counts(events: list[Any]) -> dict[str, int]:
    kind_counts: dict[str, int] = {}
    for event in events:
        kind = getattr(event, "kind", "")
        if not kind:
            continue
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    return kind_counts


def _extract_metric_field_hits(events: list[Any], field: str) -> int:
    count = 0
    for event in events:
        if getattr(event, "kind", "") != "metric":
            continue
        data = getattr(event, "data", {})
        if isinstance(data, dict) and field in data:
            count += 1
    return count


def _extract_metric_profile_hits(events: list[Any]) -> int:
    required = {"backend_id", "backend_enabled", "sample_us", "fallback_reason", "trace_id", "ler_ci"}
    required.update(
        {
            "backend_chain",
            "contract_flags",
            "profiler_flags",
            "details",
            "backend_version",
            "sample_rate",
            "sample_trace_id",
            "trace_tokens",
        }
    )
    hits = 0
    for event in events:
        if getattr(event, "kind", "") != "metric":
            continue
        data = getattr(event, "data", {})
        if isinstance(data, dict) and required.issubset(data.keys()):
            hits += 1
    return hits


def _coalesce_events(events: list[Any], max_events: int = 0) -> list[Any]:
    if max_events:
        events = events[-max_events:]
    latest: dict[str, Any] = {}
    ordered_kinds: list[str] = []
    for event in events:
        kind = getattr(event, "kind", "")
        if kind not in latest:
            ordered_kinds.append(kind)
        latest[kind] = event
    return [latest[kind] for kind in ordered_kinds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect lightweight runtime contract metrics.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/runtime.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    runner = _build_runner()
    start = time.perf_counter()
    runner.start()
    runner.join(timeout=10.0)
    elapsed = time.perf_counter() - start

    # Drain once to force event processing and validate queue behavior.
    all_events = runner.drain()
    coalesced = _coalesce_events(all_events, max_events=30)

    metrics = {
        "runner_elapsed_s": elapsed,
        "event_count": len(all_events),
        "coalesced_count": len(coalesced),
        "event_kind_counts": _extract_kind_counts(all_events),
        "coalesced_kind_counts": _extract_kind_counts(coalesced),
        "metric_count": sum(1 for event in all_events if getattr(event, "kind", "") == "metric"),
        "metric_profile_field_hits": {
            "backend_id": _extract_metric_field_hits(all_events, "backend_id"),
            "backend_enabled": _extract_metric_field_hits(all_events, "backend_enabled"),
            "backend_version": _extract_metric_field_hits(all_events, "backend_version"),
            "backend_chain": _extract_metric_field_hits(all_events, "backend_chain"),
            "contract_flags": _extract_metric_field_hits(all_events, "contract_flags"),
            "profiler_flags": _extract_metric_field_hits(all_events, "profiler_flags"),
            "sample_us": _extract_metric_field_hits(all_events, "sample_us"),
            "sample_rate": _extract_metric_field_hits(all_events, "sample_rate"),
            "fallback_reason": _extract_metric_field_hits(all_events, "fallback_reason"),
            "trace_tokens": _extract_metric_field_hits(all_events, "trace_tokens"),
            "trace_id": _extract_metric_field_hits(all_events, "trace_id"),
            "sample_trace_id": _extract_metric_field_hits(all_events, "sample_trace_id"),
            "ler_ci": _extract_metric_field_hits(all_events, "ler_ci"),
        },
        "metric_profile_event_count": _extract_metric_profile_hits(all_events),
        "queue_drops": runner.dropped_events(),
        "done_events": _extract_kind_counts(all_events).get("done", 0),
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

