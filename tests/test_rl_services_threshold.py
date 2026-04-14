from __future__ import annotations

from typing import Any

from app.services import rl_services


def test_run_threshold_sweep_passes_decoder_name_to_estimator(monkeypatch) -> None:
    estimate_calls: list[tuple[str, int]] = []

    def fake_build_threshold_decoder(decoder_name: str):
        if decoder_name == "ising":
            return object()
        raise KeyError(decoder_name)

    def fake_service_build_circuit(distance: int, rounds: int, p: float, builder_name: str | None = None):
        return f"circuit-{distance}-{rounds}-{p}-{builder_name}"

    def fake_estimate_logical_error_rate(
        circuit: Any,
        *,
        shots: int,
        seed: int,
        decoder_name: str,
    ) -> float:
        estimate_calls.append((str(decoder_name), int(seed)))
        return 0.0

    monkeypatch.setattr(
        "app.services.rl_services.build_threshold_decoder",
        fake_build_threshold_decoder,
    )
    monkeypatch.setattr(
        "app.services.rl_services.service_build_circuit",
        fake_service_build_circuit,
    )
    monkeypatch.setattr(
        "app.services.rl_services.estimate_logical_error_rate",
        fake_estimate_logical_error_rate,
    )

    result = rl_services.run_threshold_sweep(
        distances=[3],
        p_values=[0.001, 0.002],
        shots=16,
        builder_name="surface",
        decoder_name="ising",
        seed=7,
        on_progress=None,
    )

    assert result[3] == [(0.001, 0.0), (0.002, 0.0)]
    assert all(decoder == "ising" for decoder, _ in estimate_calls)


def test_run_threshold_sweep_bubbles_missing_decoder_error(monkeypatch) -> None:
    def fake_build_threshold_decoder(decoder_name: str):
        raise KeyError("unknown decoder")

    monkeypatch.setattr(
        "app.services.rl_services.build_threshold_decoder",
        fake_build_threshold_decoder,
    )

    try:
        rl_services.run_threshold_sweep(
            distances=[3],
            p_values=[0.001, 0.002],
            shots=16,
            builder_name="surface",
            decoder_name="missing",
            seed=7,
            on_progress=None,
        )
    except KeyError as exc:
        assert "unknown decoder" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing decoder")
