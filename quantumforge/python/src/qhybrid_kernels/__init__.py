"""Public exports and compatibility shims for qhybrid_kernels."""

import json
import math

import numpy as np

from .conversions import (
    complex_matrix_to_ri,
    complex_statevector_to_ri,
    kraus_1q_to_ri,
    ri_to_complex_matrix,
    ri_to_complex_statevector,
)
from .noise import (
    apply_kraus_1q_density_matrix,
    apply_pauli_channel_statevector,
    apply_correlated_pauli_noise_statevector,
    apply_cnot_error_statevector,
    expectation_value_pauli_string_py,
)
from .qiskit_adapter import (
    kraus_1q_from_qiskit,
    apply_qiskit_kraus_1q_to_density_matrix,
)
from .circuit import (
    GateConverter,
    QHYBRID_PAYLOAD_VERSION,
    QhybridCircuitPayload,
    QhybridGatePayload,
    get_supported_qiskit_gates,
    qiskit_to_qhybrid_json,
    register_gate_converter,
)

try:
    from .rust_kernels import (
        execute_quantum_circuit,
        apply_correlated_pauli_noise_statevector as _apply_correlated_pauli_noise_statevector,
        apply_cnot_error_statevector as _apply_cnot_error_statevector,
        expectation_value_pauli_string_py as _expectation_value_pauli_string_py,
        QuantumCircuit,
    )
    apply_correlated_pauli_noise_statevector = _apply_correlated_pauli_noise_statevector
    apply_cnot_error_statevector = _apply_cnot_error_statevector
    expectation_value_pauli_string_py = _expectation_value_pauli_string_py
except ImportError:
    QuantumCircuit = None

    def _normalise_gate_name(gate_type) -> str:
        if isinstance(gate_type, str):
            return gate_type
        if not isinstance(gate_type, dict) or len(gate_type) != 1:
            raise TypeError("Invalid qhybrid gate payload")
        return next(iter(gate_type.keys()))

    def _normalise_gate_params(gate_type) -> list[float]:
        if isinstance(gate_type, str):
            return []
        if not isinstance(gate_type, dict) or len(gate_type) != 1:
            raise TypeError("Invalid qhybrid gate payload")
        params = next(iter(gate_type.values()))
        if isinstance(params, (list, tuple)):
            return [float(v) for v in params]
        return [float(params)]

    def _coerce_gate_matrix(gate_name: str, params: list[float]) -> np.ndarray:
        if gate_name == "I":
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        if gate_name == "X":
            return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        if gate_name == "Y":
            return np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
        if gate_name == "Z":
            return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        if gate_name == "H":
            inv_root2 = 1.0 / math.sqrt(2.0)
            return np.array(
                [[inv_root2, inv_root2], [inv_root2, -inv_root2]],
                dtype=np.complex128,
            )
        if gate_name == "S":
            return np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128)
        if gate_name == "Sdg":
            return np.array([[1.0, 0.0], [0.0, -1j]], dtype=np.complex128)
        if gate_name == "T":
            return np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        if gate_name == "Tdg":
            return np.array(
                [[1.0, 0.0], [0.0, np.exp(-1j * np.pi / 4)]],
                dtype=np.complex128,
            )
        if gate_name == "RX":
            theta = float(params[0])
            c = math.cos(theta / 2.0)
            s = math.sin(theta / 2.0)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        if gate_name == "RY":
            theta = float(params[0])
            c = math.cos(theta / 2.0)
            s = math.sin(theta / 2.0)
            return np.array([[c, -s], [s, c]], dtype=np.complex128)
        if gate_name in {"RZ", "P"}:
            theta = float(params[0])
            return np.array(
                [[np.exp(-1j * theta / 2.0), 0.0], [0.0, np.exp(1j * theta / 2.0)]],
                dtype=np.complex128,
            )
        if gate_name == "U3":
            if len(params) != 3:
                raise ValueError("U3 gate expects 3 parameters")
            theta, phi, lam = params
            c = math.cos(theta / 2.0)
            s = math.sin(theta / 2.0)
            return np.array(
                [
                    [c, -np.exp(1j * lam) * s],
                    [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
                ],
                dtype=np.complex128,
            )
        raise ValueError(f"Unsupported qhybrid gate: {gate_name}")

    def _apply_single_qubit(state: np.ndarray, matrix: np.ndarray, qubit: int) -> np.ndarray:
        dim = state.shape[0]
        for idx in range(dim):
            if (idx >> qubit) & 1:
                continue
            pair_idx = idx | (1 << qubit)
            a = state[idx]
            b = state[pair_idx]
            state[idx] = matrix[0, 0] * a + matrix[0, 1] * b
            state[pair_idx] = matrix[1, 0] * a + matrix[1, 1] * b
        return state

    def _apply_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
        for idx in range(state.shape[0]):
            if ((idx >> control) & 1) == 1 and ((idx >> target) & 1) == 0:
                partner = idx | (1 << target)
                state[idx], state[partner] = state[partner], state[idx]
        return state

    def execute_quantum_circuit(circuit_json: str) -> np.ndarray:
        payload = json.loads(circuit_json)
        schema_version = payload.get("schema_version")
        if schema_version is None:
            raise ValueError("Missing circuit schema_version")
        if schema_version != QHYBRID_PAYLOAD_VERSION:
            raise ValueError(f"Unsupported circuit schema_version: {schema_version}")

        n_qubits = int(payload["n_qubits"])
        dim = 1 << n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0.0j

        for gate in payload.get("gates", []):
            gate_name = _normalise_gate_name(gate["gate_type"])
            gate_params = _normalise_gate_params(gate["gate_type"])
            qubits = list(gate["qubits"])

            if gate_name in {"CX", "CNOT"}:
                state = _apply_cx(state, qubits[0], qubits[1])
            elif gate_name == "CY":
                for idx in range(state.shape[0]):
                    if ((idx >> qubits[0]) & 1) == 1 and ((idx >> qubits[1]) & 1) == 0:
                        partner = idx | (1 << qubits[1])
                        y_matrix = _coerce_gate_matrix("Y", [])
                        a = state[idx]
                        b = state[partner]
                        state[idx] = y_matrix[0, 0] * a + y_matrix[0, 1] * b
                        state[partner] = y_matrix[1, 0] * a + y_matrix[1, 1] * b
            elif gate_name == "CZ":
                for idx in range(state.shape[0]):
                    if ((idx >> qubits[0]) & 1) == 1 and ((idx >> qubits[1]) & 1) == 1:
                        state[idx] = -state[idx]
                # no-op for all other basis states
            else:
                matrix = _coerce_gate_matrix(gate_name, gate_params)
                state = _apply_single_qubit(state, matrix, qubits[0])

        return complex_statevector_to_ri(state)

__all__ = [
    "complex_statevector_to_ri",
    "ri_to_complex_statevector",
    "complex_matrix_to_ri",
    "ri_to_complex_matrix",
    "kraus_1q_to_ri",
    "apply_pauli_channel_statevector",
    "apply_kraus_1q_density_matrix",
    "apply_correlated_pauli_noise_statevector",
    "apply_cnot_error_statevector",
    "expectation_value_pauli_string_py",
    "kraus_1q_from_qiskit",
    "apply_qiskit_kraus_1q_to_density_matrix",
    "GateConverter",
    "QHYBRID_PAYLOAD_VERSION",
    "QhybridCircuitPayload",
    "QhybridGatePayload",
    "get_supported_qiskit_gates",
    "execute_quantum_circuit",
    "register_gate_converter",
]


def _patch_qiskit_kraus_compat() -> None:
    """Patch qiskit Kraus constructor to accept NumPy Kraus stacks.

    Newer Qiskit versions are stricter about Kraus input shape and may reject
    a 3-D NumPy array directly. The project tests still use this legacy input
    form in some places, so we add a small compatibility shim at import time.
    """

    try:
        from qiskit.quantum_info import Kraus
    except Exception:
        return

    if getattr(Kraus, "_qhybrid_kernels_numpy_compat", False):
        return

    _original_init = Kraus.__init__

    def _compatible_init(self, data, *args, **kwargs):  # type: ignore[no-redef]
        if (
            isinstance(data, np.ndarray)
            and data.ndim == 3
            and data.shape[1:] == (2, 2)
        ):
            data = [np.asarray(data[idx], dtype=np.complex128) for idx in range(data.shape[0])]
        return _original_init(self, data, *args, **kwargs)

    Kraus.__init__ = _compatible_init  # type: ignore[assignment]
    Kraus._qhybrid_kernels_numpy_compat = True  # type: ignore[attr-defined]


_patch_qiskit_kraus_compat()

