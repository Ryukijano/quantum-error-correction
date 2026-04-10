"""Sampling-backend factory and fallback contract tests."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from surface_code_in_stem.accelerators import sampling_backends as sb


class DummySampler:
    def sample(self, shots: int, separate_observables: bool = True):
        return np.array([[1]], dtype=np.uint8), np.array([[0]], dtype=np.uint8)


class DummyCircuit:
    def compile_detector_sampler(self, seed: int):
        return DummySampler()


def _make_backend_factory(backend_id: str):
    def _backend(*args: Any, **kwargs: Any):
        details = kwargs.get("details", {})
        if len(args) >= 3 and not isinstance(details, dict):
            # Accept legacy positional call signatures.
            details = args[2] if isinstance(args[2], dict) else {}

        trace_tokens = details.get("backend_chain")
        if not isinstance(trace_tokens, list):
            trace_tokens = [backend_id]

        return type(
            "_BackendProxy",
            (),
            {
                "last_sample_us": 0.0,
                "metadata": sb.SamplingBackendMetadata(
                    backend_id=backend_id,
                    backend_version="test",
                    trace_tokens=trace_tokens,
                    sample_rate=0.0,
                    backend_enabled=not bool(details.get("disabled", False)),
                    fallback_reason=details.get("fallback_reason"),
                    sample_trace_id=details.get("sample_trace_id"),
                    details=details,
                ),
                "sample": lambda self: (
                    np.array([[1]], dtype=np.int8),
                    np.array([[0]], dtype=np.int8),
                ),
            },
        )()

    return _backend


def _always_probe() -> dict[str, dict[str, Any]]:
    return {
        "stim": {"enabled": True, "version": "x", "details": {}},
        "qhybrid": {"enabled": False, "version": "x", "details": {}},
        "cuquantum": {"enabled": False, "version": "x", "details": {}},
        "qujax": {"enabled": False, "version": "x", "details": {}},
        "cudaq": {"enabled": False, "version": "x", "details": {}},
    }


def test_sampling_backend_factory_prefers_stim_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(sb, "_probe_backends", _always_probe)
    monkeypatch.setattr(sb, "_StimSamplingBackend", _make_backend_factory("stim"))
    backend = sb.build_sampling_backend(
        DummyCircuit(),
        seed=7,
        use_accelerated=True,
        backend_override="stim",
        protocol_metadata={"tag": "unit"},
        sample_trace_id="trace-1",
    )

    assert backend.metadata.backend_id == "stim"
    assert backend.metadata.sample_trace_id == "trace-1"
    samples = backend.sample()
    assert isinstance(samples[0], np.ndarray)


def test_sampling_backend_factory_auto_uses_accelerated_candidates_when_enabled(monkeypatch) -> None:
    probe = _always_probe()
    probe["qhybrid"]["enabled"] = True
    monkeypatch.setattr(sb, "_probe_backends", lambda: probe)
    monkeypatch.setattr(sb, "_QhybridSamplingBackend", _make_backend_factory("qhybrid"))
    backend = sb.build_sampling_backend(
        DummyCircuit(),
        seed=13,
        use_accelerated=True,
        protocol_metadata={"tag": "unit"},
        sample_trace_id="trace-auto",
    )

    assert backend.metadata.backend_id == "qhybrid"
    assert backend.metadata.sample_trace_id == "trace-auto"


def test_sampling_backend_factory_falls_back_to_stim_when_candidate_fails(monkeypatch) -> None:
    probe = _always_probe()
    probe["qhybrid"]["enabled"] = True
    monkeypatch.setattr(sb, "_probe_backends", lambda: probe)
    monkeypatch.setattr(sb, "_QhybridSamplingBackend", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("qhybrid disabled")))
    monkeypatch.setattr(sb, "_CuQuantumSamplingBackend", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cuquantum unavailable")))
    monkeypatch.setattr(sb, "_QuJaxSamplingBackend", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("qujax unavailable")))
    monkeypatch.setattr(sb, "_CudaQSamplingBackend", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cudaq unavailable")))
    monkeypatch.setattr(sb, "_StimSamplingBackend", _make_backend_factory("stim"))

    backend = sb.build_sampling_backend(
        DummyCircuit(),
        seed=7,
        use_accelerated=True,
        backend_override="qhybrid",
        protocol_metadata={"tag": "unit"},
    )

    assert backend.metadata.backend_id == "stim"
    assert backend.metadata.fallback_reason is not None
    assert "qhybrid sample path failed" in backend.metadata.fallback_reason
    assert backend.metadata.trace_tokens[-1] == "stim"
    assert "qhybrid_fallback" in backend.metadata.trace_tokens


def test_sampling_backend_factory_falls_back_to_stim_when_accelerated_override_fails(monkeypatch) -> None:
    probe = _always_probe()
    probe["qhybrid"]["enabled"] = True
    monkeypatch.setattr(sb, "_probe_backends", lambda: probe)
    monkeypatch.setattr(
        sb,
        "_QhybridSamplingBackend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("qhybrid unavailable")),
    )
    monkeypatch.setattr(sb, "_StimSamplingBackend", _make_backend_factory("stim"))

    backend = sb.build_sampling_backend(
        DummyCircuit(),
        seed=17,
        use_accelerated=True,
        backend_override="qhybrid",
        protocol_metadata={"tag": "accel"},
        sample_trace_id="trace-accel",
    )

    assert backend.metadata.backend_id == "stim"
    assert backend.metadata.sample_trace_id == "trace-accel"
    assert backend.metadata.fallback_reason is not None
    assert "qhybrid sample path failed" in backend.metadata.fallback_reason
    assert "qhybrid_fallback" in backend.metadata.trace_tokens


def test_sampling_backend_factory_records_backend_chain_when_all_candidates_disabled(monkeypatch) -> None:
    probe = _always_probe()
    probe["qhybrid"]["enabled"] = False
    probe["cuquantum"]["enabled"] = False
    probe["qujax"]["enabled"] = False
    probe["cudaq"]["enabled"] = False
    monkeypatch.setattr(sb, "_probe_backends", lambda: probe)

    backend = sb.build_sampling_backend(
        DummyCircuit(),
        seed=11,
        use_accelerated=False,
        backend_override="qhybrid",
        protocol_metadata={"tag": "chain-test"},
        sample_trace_id="trace-chain",
    )

    assert backend.metadata.backend_id == "stim"
    assert backend.metadata.fallback_reason == "all candidates unavailable"
    assert backend.metadata.sample_trace_id == "trace-chain"
    assert backend.metadata.trace_tokens == [
        "stim",
        "qhybrid_unavailable",
        "cuquantum_unavailable",
        "qujax_unavailable",
        "cudaq_unavailable",
    ]


def test_sampling_backend_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown sampling backend"):
        sb.build_sampling_backend(
            DummyCircuit(),
            seed=7,
            use_accelerated=False,
            backend_override="missing-backend",
            protocol_metadata={},
            sample_trace_id="trace",
        )


def test_sampling_backend_backend_sample_rate_populates_after_sample(monkeypatch) -> None:
    def _rate_backend(*_args: Any, **_kwargs: Any):
        details = _kwargs.get("details", {})
        if not isinstance(details, dict):
            details = {}
        return type(
            "_BackendProxy",
            (),
            {
                "last_sample_us": 0.0,
                "metadata": sb.SamplingBackendMetadata(
                    backend_id="qhybrid",
                    backend_version="test",
                    trace_tokens=details.get("backend_chain", ["qhybrid", "probe"]),
                    sample_rate=0.0,
                    backend_enabled=True,
                    sample_trace_id=details.get("sample_trace_id"),
                    details=details,
                ),
                "sample": lambda self: (
                    np.array([[1]], dtype=np.int8),
                    np.array([[0]], dtype=np.int8),
                ),
            },
        )()

    monkeypatch.setattr(sb, "_probe_backends", _always_probe)
    monkeypatch.setattr(sb, "_QhybridSamplingBackend", _rate_backend)
    backend = sb.build_sampling_backend(
        DummyCircuit(),
        seed=9,
        use_accelerated=True,
        backend_override="qhybrid",
        protocol_metadata={},
    )

    backend.sample()
    assert backend.metadata.backend_id == "qhybrid"
    assert backend.metadata.sample_rate >= 0.0


def test_sampling_backend_probe_contract_contracts_match_runtime_dependencies(monkeypatch) -> None:
    class _DummyModule:
        __version__ = "unit-test"

        @staticmethod
        def probe_capability() -> dict[str, object]:
            return {"enabled": False, "details": {"module": "qhybrid", "tag": "probe-test"}}

    class _NoBackend:
        __version__ = "unit-test"

    monkeypatch.setattr(sb, "stim", _DummyModule())
    monkeypatch.setattr(sb, "qhybrid_backend", _DummyModule())
    monkeypatch.setattr(sb, "cuquantum_tensornet", _NoBackend())
    monkeypatch.setattr(sb, "jax", _NoBackend())
    monkeypatch.setattr(sb, "cudaq", _NoBackend())

    probe = sb.probe_sampling_backends()

    assert {"stim", "qhybrid", "cuquantum", "qujax", "cudaq"} <= set(probe.keys())
    for details in probe.values():
        assert isinstance(details.get("enabled"), bool)
        assert isinstance(details.get("version"), str)
        assert isinstance(details.get("details"), dict)
    assert probe["qhybrid"]["enabled"] is False
    assert probe["qhybrid"]["details"] == {"module": "qhybrid", "tag": "probe-test"}
