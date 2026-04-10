"""Boundary checks for fallback-mode accelerator behavior."""

import numpy as np

from surface_code_in_stem.accelerators import qhybrid_backend


def test_fallback_backend_marks_execution_as_degraded(monkeypatch):
    monkeypatch.setattr(qhybrid_backend, "HAS_QHYBRID", False)
    monkeypatch.setattr(qhybrid_backend, "_BACKEND", qhybrid_backend._NoopQhybridAccelerator())

    initial = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    result = qhybrid_backend.apply_pauli_channel_with_metadata(
        initial,
        n_qubits=1,
        target_qubit=0,
        probs=[1.0, 0.0, 0.0, 0.0],
    )

    metadata = result.metadata
    assert metadata.enabled is False
    assert metadata.degraded is True
    assert metadata.reason == "qhybrid_kernels unavailable"
    assert qhybrid_backend.is_accelerated() is False
    np.testing.assert_array_equal(result.state, initial)


def test_fallback_backend_keeps_backward_compatible_state_return(monkeypatch):
    monkeypatch.setattr(qhybrid_backend, "HAS_QHYBRID", False)
    monkeypatch.setattr(qhybrid_backend, "_BACKEND", qhybrid_backend._NoopQhybridAccelerator())

    initial = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    output = qhybrid_backend.apply_pauli_channel(
        initial,
        n_qubits=1,
        target_qubit=0,
        probs=[1.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_array_equal(output, initial)


def test_probe_capability_reports_accelerated_state(monkeypatch):
    monkeypatch.setattr(qhybrid_backend, "HAS_QHYBRID", False)
    monkeypatch.setattr(qhybrid_backend, "_BACKEND", qhybrid_backend._NoopQhybridAccelerator())

    capability = qhybrid_backend.probe_capability()
    assert capability["name"] == "qhybrid"
    assert capability["enabled"] is False
    assert capability["degraded"] is True
    assert capability["available"] is False
    assert capability["reason"] == "qhybrid_kernels unavailable"
