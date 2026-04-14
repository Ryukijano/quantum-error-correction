"""Tests for RL event normalization and backend metadata propagation."""

from __future__ import annotations

import json

from app.services.rl_services import (
    _normalize_event,
    history_to_download_json,
    metric_payload_for_history,
)


class _Event:
    kind = "metric"

    def __init__(self, data):
        self.data = data


def test_normalize_event_exposes_backend_metadata_fields() -> None:
    event = _Event(
        data={
            "backend_id": "qujax",
            "backend_enabled": False,
            "sample_us": 12.5,
            "backend_version": "sim",
            "sample_rate": 1.2,
            "trace_tokens": ["qujax", "runtime"],
            "sample_trace_id": "trace-007",
            "backend_chain": "requested:qujax->selected:qujax",
            "contract_flags": "backend_disabled,contract_fallback",
            "profiler_flags": "sample_trace_present,trace_chain_recorded",
            "ler_ci": {"lower": 0.01, "upper": 0.3},
            "trace_id": "trace-007",
            "fallback_reason": "qujax unavailable",
            "details": {"backend": "qujax", "seed": 7},
        }
    )

    normalized = _normalize_event(event)
    assert normalized.kind == "metric"
    assert normalized.payload["backend_id"] == "qujax"
    assert normalized.payload["backend_enabled"] is False
    assert normalized.payload["sample_us"] == 12.5
    assert normalized.payload["backend_version"] == "sim"
    assert normalized.payload["sample_rate"] == 1.2
    assert normalized.payload["trace_tokens"] == ["qujax", "runtime"]
    assert normalized.payload["backend_chain"] == "requested:qujax->selected:qujax"
    assert normalized.payload["contract_flags"] == "backend_disabled,contract_fallback"
    assert normalized.payload["profiler_flags"] == "sample_trace_present,trace_chain_recorded"
    assert normalized.payload["sample_trace_id"] == "trace-007"
    assert normalized.payload["details"] == {"backend": "qujax", "seed": 7}
    assert normalized.payload["ler_ci"]["upper"] == 0.3
    assert normalized.payload["trace_id"] == "trace-007"
    assert normalized.payload["fallback_reason"] == "qujax unavailable"
    assert normalized.payload["data"]["backend_id"] == "qujax"


def test_normalize_event_exposes_baseline_decoder_metadata_fields() -> None:
    event = _Event(
        data={
            "backend_id": "mwpm",
            "backend_enabled": True,
            "sample_us": 10.0,
            "baseline_decoder": "ising",
            "baseline_decoder_requested": "ising",
            "baseline_decoder_fallback_reason": "predecode_disabled",
            "baseline_predecode_backend": "cnn",
            "baseline_predecode_latency_ms": 2.5,
            "baseline_predecode_fallback_reason": "none",
            "baseline_contract_flags": "backend_enabled,contract_met",
            "baseline_decoder_diagnostics": {
                "predecoder_backend": "cnn",
                "predecoder_latency_ms": 2.5,
                "predecoder_fallback_reason": "none",
            },
        }
    )

    normalized = _normalize_event(event)
    assert normalized.payload["baseline_decoder"] == "ising"
    assert normalized.payload["baseline_decoder_requested"] == "ising"
    assert normalized.payload["baseline_decoder_fallback_reason"] == "predecode_disabled"
    assert normalized.payload["baseline_predecode_backend"] == "cnn"
    assert normalized.payload["baseline_predecode_latency_ms"] == 2.5
    assert normalized.payload["baseline_predecode_fallback_reason"] == "none"
    assert normalized.payload["baseline_contract_flags"] == "backend_enabled,contract_met"
    assert normalized.payload["baseline_decoder_diagnostics"]["predecoder_backend"] == "cnn"
    assert normalized.payload.get("predecoder_seed") is None


def test_normalize_event_handles_non_dict_metric_payload() -> None:
    event = _Event(data="not-a-dict")
    normalized = _normalize_event(event)
    assert normalized.payload["data"] == {}
    assert normalized.payload["backend_id"] is None
    assert normalized.payload["sample_us"] is None


def test_normalize_event_includes_predecoder_config_fields() -> None:
    event = _Event(
        data={
            "backend_id": "mwpm",
            "backend_enabled": True,
            "predecoder_backend": "torch",
            "predecoder_artifact": "/tmp/ising.pt",
            "predecoder_seed": 123,
            "predecoder_fallback_reason": "none",
        }
    )

    normalized = _normalize_event(event)
    assert normalized.payload["predecoder_backend"] == "torch"
    assert normalized.payload["predecoder_artifact"] == "/tmp/ising.pt"
    assert normalized.payload["predecoder_seed"] == 123
    assert normalized.payload["predecoder_fallback_reason"] == "none"


def test_metric_payload_for_history_merges_top_level_metadata_fields() -> None:
    payload = {
        "data": {"episode": 3},
        "backend_id": "qhybrid",
        "backend_enabled": False,
        "sample_us": 11.5,
        "backend_version": "qhybrid",
        "sample_rate": 7.5,
        "trace_tokens": ["qhybrid", "fallback"],
        "backend_chain": "requested:qhybrid->selected:qhybrid",
        "contract_flags": "backend_disabled,contract_fallback",
        "profiler_flags": "sample_trace_present,trace_chain_recorded",
        "sample_trace_id": "trace-top-level",
        "details": {"selected": "qhybrid"},
        "fallback_reason": "fallback triggered",
        "trace_id": "trace-top-level",
        "ler_ci": {"lower": 0.05, "upper": 0.15},
    }
    history_row = metric_payload_for_history(payload)

    assert history_row["backend_id"] == "qhybrid"
    assert history_row["backend_enabled"] is False
    assert history_row["sample_us"] == 11.5
    assert history_row["backend_version"] == "qhybrid"
    assert history_row["sample_rate"] == 7.5
    assert history_row["trace_tokens"] == ["qhybrid", "fallback"]
    assert history_row["backend_chain"] == "requested:qhybrid->selected:qhybrid"
    assert history_row["contract_flags"] == "backend_disabled,contract_fallback"
    assert history_row["profiler_flags"] == "sample_trace_present,trace_chain_recorded"
    assert history_row["sample_trace_id"] == "trace-top-level"
    assert history_row["details"] == {"selected": "qhybrid"}
    assert history_row["fallback_reason"] == "fallback triggered"
    assert history_row["trace_id"] == "trace-top-level"
    assert history_row["ler_ci"]["upper"] == 0.15


def test_history_download_json_keeps_backend_metadata_fields() -> None:
    payload = metric_payload_for_history(
        {
            "data": {
                "episode": 4,
                "backend_id": "qujax",
                "backend_enabled": True,
                "sample_us": 9.2,
                "backend_version": "qujax",
                "sample_rate": 11.0,
                "trace_tokens": ["qujax", "fallback"],
                "backend_chain": "requested:qujax->selected:qujax",
                "contract_flags": "backend_enabled,contract_met",
                "profiler_flags": "sample_trace_present,trace_chain_recorded",
                "sample_trace_id": "trace-json",
                "details": {"attempt": 1},
                "fallback_reason": None,
                "ler_ci": {"lower": 0.01, "upper": 0.2},
                "trace_id": "trace-json",
            }
        }
    )
    dumped = history_to_download_json([payload, {"episode": 5}])
    parsed = json.loads(dumped)

    assert parsed[0]["backend_id"] == "qujax"
    assert parsed[0]["backend_enabled"] is True
    assert parsed[0]["sample_us"] == 9.2
    assert parsed[0]["backend_version"] == "qujax"
    assert parsed[0]["sample_rate"] == 11.0
    assert parsed[0]["trace_tokens"] == ["qujax", "fallback"]
    assert parsed[0]["backend_chain"] == "requested:qujax->selected:qujax"
    assert parsed[0]["contract_flags"] == "backend_enabled,contract_met"
    assert parsed[0]["profiler_flags"] == "sample_trace_present,trace_chain_recorded"
    assert parsed[0]["sample_trace_id"] == "trace-json"
    assert parsed[0]["details"] == {"attempt": 1}
    assert parsed[0]["trace_id"] == "trace-json"
    assert parsed[0]["ler_ci"] == {"lower": 0.01, "upper": 0.2}
