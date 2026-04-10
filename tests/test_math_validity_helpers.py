"""Tests for confidence-interval and profile helper utilities."""

from __future__ import annotations

from app.rl_strategies import (
    _append_ler_ci,
    _append_sampling_fields,
    _has_iqm_convergence,
    _interquartile_mean,
    _safe_float,
    _wilson_interval,
)


def test_append_ler_ci_empty_history() -> None:
    metric: dict[str, float | dict[str, float] | None] = {}
    metric = _append_ler_ci(metric, [])
    assert metric["ler_ci"] == {"lower": 0.0, "upper": 0.0}


def test_append_ler_ci_bounds_and_order() -> None:
    metric: dict[str, float | dict[str, float] | None] = {}
    metric = _append_ler_ci(metric, [1.0, 1.0, 0.0, 1.0, 0.0])
    ci = metric["ler_ci"]
    assert isinstance(ci, dict)
    assert "lower" in ci and "upper" in ci
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0


def test_append_sampling_fields_stable_payload() -> None:
    metric: dict[str, object] = {}
    _append_sampling_fields(
        metric,
        {
            "sampling_backend": {
                "backend_id": "cuqnn",
                "backend_enabled": True,
                "fallback_reason": None,
                "sample_us": 12.5,
                "backend_version": "test",
                "sample_rate": 0.0,
                "trace_tokens": ["sample", "backend"],
                "details": {"origin": "test"},
            },
            "sample_trace_id": "trace-01",
            "sample_us": 12.5,
        },
    )
    assert metric["backend_id"] == "cuqnn"
    assert metric["backend_enabled"] is True
    assert metric["fallback_reason"] is None
    assert metric["backend_version"] == "test"
    assert metric["sample_rate"] == 0.0
    assert metric["trace_tokens"] == ["sample", "backend"]
    assert metric["details"] == {"origin": "test"}
    assert metric["sample_us"] == 12.5
    assert metric["trace_id"] == "trace-01"


def test_append_sampling_fields_transports_chain_and_flags() -> None:
    metric: dict[str, object] = {}
    _append_sampling_fields(
        metric,
        {
            "sampling_backend": {
                "backend_id": "qujax",
                "backend_enabled": False,
                "fallback_reason": "qujax unavailable",
                "sample_rate": 0.0,
                "trace_tokens": ["requested:qujax", "selected:qujax", "disabled"],
                "sample_trace_id": "trace-chain",
                "backend_chain": "requested:qujax->selected:qujax->disabled",
                "contract_flags": "backend_disabled,contract_fallback",
                "profiler_flags": "sample_trace_present,trace_chain_recorded",
            },
            "sample_us": 18.2,
        },
    )
    assert metric["backend_chain"] == "requested:qujax->selected:qujax->disabled"
    assert metric["contract_flags"] == "backend_disabled,contract_fallback"
    assert metric["profiler_flags"] == "sample_trace_present,trace_chain_recorded"
    assert metric["sample_trace_id"] == "trace-chain"
    assert metric["trace_id"] == "trace-chain"


def test_safe_float_casting() -> None:
    assert _safe_float(3) == 3.0
    assert _safe_float(2.5) == 2.5
    assert _safe_float(True) == 1.0
    assert _safe_float("not-a-number") is None


def test_safe_float_rejects_non_finite() -> None:
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None


def test_interquartile_mean_trims_extremes() -> None:
    values = [0.0, 0.0, 0.0, 0.9, 1.0, 1.0, 1.0, 100.0]
    assert 0.0 < _interquartile_mean(values) < 1.0
    assert _interquartile_mean([]) == 0.0


def test_wilson_interval_extremes_are_bounded() -> None:
    lower, upper = _wilson_interval(0, 10)
    assert 0.0 <= lower <= upper <= 1.0
    assert lower == 0.0

    lower, upper = _wilson_interval(10, 10)
    assert 0.0 <= lower <= upper <= 1.0
    assert upper == 1.0


def test_wilson_interval_zero_trial_history_stays_neutral() -> None:
    assert _wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_handles_fractional_counts() -> None:
    lower, upper = _wilson_interval(2.5, 4.0)
    assert 0.0 <= lower <= upper <= 1.0
    assert lower < 0.5 < upper


def test_iqm_convergence_checks_recent_windows() -> None:
    stable = [0.78] * 20 + [0.781, 0.779, 0.780, 0.782, 0.781, 0.779, 0.780, 0.781, 0.782, 0.780]
    assert _has_iqm_convergence(stable, window=10, relative_tolerance=0.01)


def test_iqm_convergence_detects_small_noise_stable_history() -> None:
    stable = [0.5 + (0.005 if i % 2 == 0 else -0.005) for i in range(40)]
    assert _has_iqm_convergence(stable, window=10, relative_tolerance=0.05)


def test_iqm_convergence_rejects_unstable_windows_with_equal_iqm() -> None:
    unstable = [0.0, 1.0] * 20
    assert not _has_iqm_convergence(unstable, window=8, relative_tolerance=0.2)


def test_iqm_convergence_zero_scale_history_stays_stable() -> None:
    assert _has_iqm_convergence([0.0] * 20, window=5, relative_tolerance=0.01)


def test_iqm_convergence_rejects_noisy_history() -> None:
    unstable = [0.1, 0.2, 0.9, 0.2, 0.8, 0.1, 0.4, 0.5, 0.7, 0.2] * 2
    assert not _has_iqm_convergence(unstable, window=5, relative_tolerance=0.05)


def test_append_ler_ci_skips_non_finite_entries_and_clamps_bounds() -> None:
    metric: dict[str, float | dict[str, float] | None] = {}
    metric = _append_ler_ci(metric, [1.0, 0.0, 2.0, -1.0, float("nan"), float("inf")])
    ci = metric["ler_ci"]
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci["lower"] < ci["upper"]


def test_interquartile_mean_ignores_non_finite_and_extremes() -> None:
    values = [1.0, 1.0, 0.0, 0.0, float("inf"), float("nan")]
    assert _interquartile_mean(values) == 0.5
