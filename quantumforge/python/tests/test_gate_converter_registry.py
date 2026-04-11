"""Boundary tests for qhybrid circuit converter registry contracts."""

import json

import pytest

from qhybrid_kernels.circuit import (
    QHYBRID_PAYLOAD_VERSION,
    QhybridGatePayload,
    get_supported_qiskit_gates,
    qiskit_to_qhybrid_json,
    register_gate_converter,
)


class _DummyOperation:
    def __init__(self, params: list[float]):
        self.params = params


def test_register_gate_converter_rejects_invalid_arity():
    with pytest.raises(ValueError):
        register_gate_converter("bad", "X", arity=0)

    with pytest.raises(ValueError):
        register_gate_converter("bad", "X", arity=4)


def test_register_gate_converter_registers_and_converts_payload():
    converter = register_gate_converter("custom_rot", "RX", arity=1, expected_params=1)
    payload = converter(_DummyOperation([0.25]), [2])
    assert payload == QhybridGatePayload(
        gate_type={"RX": 0.25},
        qubits=[2],
        parameters=[0.25],
    )

    assert "custom_rot" in get_supported_qiskit_gates()

    with pytest.raises(ValueError):
        converter(_DummyOperation([]), [2])


def test_supported_gate_registry_exposes_deterministic_order():
    supported = get_supported_qiskit_gates()
    assert supported == sorted(supported)


def test_qiskit_to_qhybrid_json_includes_schema_version():
    pytest.importorskip("qiskit")

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    payload = json.loads(qiskit_to_qhybrid_json(qc))

    assert payload["schema_version"] == QHYBRID_PAYLOAD_VERSION
    assert payload["n_qubits"] == 2
