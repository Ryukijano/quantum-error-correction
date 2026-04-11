"""Contract tests for qhybrid JSON schema version handling."""

import json

import pytest

from qhybrid_kernels import execute_quantum_circuit
from qhybrid_kernels.circuit import QHYBRID_PAYLOAD_VERSION


def test_execute_quantum_circuit_rejects_missing_schema_version():
    payload = {
        "n_qubits": 1,
        "gates": [],
        "name": "legacy_payload",
    }

    with pytest.raises(ValueError, match="Missing circuit schema_version"):
        execute_quantum_circuit(json.dumps(payload))


def test_execute_quantum_circuit_rejects_unsupported_schema_version():
    payload = {
        "schema_version": QHYBRID_PAYLOAD_VERSION + 1,
        "n_qubits": 1,
        "gates": [],
        "name": "future_payload",
    }

    with pytest.raises(ValueError, match="Unsupported circuit schema_version"):
        execute_quantum_circuit(json.dumps(payload))


def test_execute_quantum_circuit_accepts_current_schema_version():
    payload = {
        "schema_version": QHYBRID_PAYLOAD_VERSION,
        "n_qubits": 1,
        "gates": [],
        "name": "valid_payload",
    }

    result = execute_quantum_circuit(json.dumps(payload))
    assert result.shape == (2, 2)
