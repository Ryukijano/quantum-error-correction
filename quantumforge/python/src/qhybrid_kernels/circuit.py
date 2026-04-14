"""Circuit conversion contracts for the qhybrid simulator payload."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable


QHYBRID_PAYLOAD_VERSION = 1

if TYPE_CHECKING:
    from qiskit import QuantumCircuit as QiskitCircuit


@dataclass(frozen=True)
class QhybridGatePayload:
    """Typed payload for one gate in qhybrid JSON."""

    gate_type: str | dict[str, list[float] | float]
    qubits: list[int]
    parameters: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_type": self.gate_type,
            "qubits": self.qubits,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class QhybridCircuitPayload:
    """Typed payload for a full qhybrid circuit."""

    n_qubits: int
    gates: list[QhybridGatePayload]
    schema_version: int
    name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_qubits": self.n_qubits,
            "gates": [gate.as_dict() for gate in self.gates],
            "name": self.name,
        }


GateConverter = Callable[[Any, list[int]], QhybridGatePayload]


def _default_gate_type(serde_gate: str, params: list[float]) -> str | dict[str, list[float] | float]:
    if serde_gate in {"RX", "RY", "RZ", "P"}:
        if len(params) != 1:
            raise ValueError(f"{serde_gate} requires exactly one parameter")
        return {serde_gate: params[0]}
    if serde_gate == "U3":
        if len(params) != 3:
            raise ValueError("U3 requires exactly three parameters")
        return {"U3": params}
    if params:
        raise ValueError(f"{serde_gate} does not accept parameters")
    return serde_gate


def register_gate_converter(
    qiskit_name: str,
    serde_gate: str,
    arity: int,
    *,
    expected_params: int | None = None,
) -> GateConverter:
    """Register a qiskit-name converter for a qhybrid gate type."""
    qiskit_name = qiskit_name.lower()

    if arity < 1 or arity > 3:
        raise ValueError(f"Unsupported arity for {qiskit_name}: {arity}")

    def _converter(operation: Any, op_qubits: list[int]) -> QhybridGatePayload:
        if len(op_qubits) != arity:
            raise ValueError(
                f"{qiskit_name} expects {arity} qubits, got {len(op_qubits)}"
            )
        params = [float(value) for value in operation.params]
        if expected_params is not None and len(params) != expected_params:
            raise ValueError(
                f"{qiskit_name} expects {expected_params} params, got {len(params)}"
            )
        gate_type = _default_gate_type(serde_gate, params)
        return QhybridGatePayload(
            gate_type=gate_type,
            qubits=op_qubits,
            parameters=params,
        )

    _GATE_CONVERTERS[qiskit_name] = _converter
    return _converter


def _build_gate_converter_registry() -> dict[str, GateConverter]:
    registry: dict[str, GateConverter] = {}
    global _GATE_CONVERTERS
    _GATE_CONVERTERS = registry

    register_gate_converter("id", "I", arity=1)
    register_gate_converter("x", "X", arity=1)
    register_gate_converter("y", "Y", arity=1)
    register_gate_converter("z", "Z", arity=1)
    register_gate_converter("h", "H", arity=1)
    register_gate_converter("s", "S", arity=1)
    register_gate_converter("sdg", "Sdg", arity=1)
    register_gate_converter("t", "T", arity=1)
    register_gate_converter("tdg", "Tdg", arity=1)
    register_gate_converter("rx", "RX", arity=1, expected_params=1)
    register_gate_converter("ry", "RY", arity=1, expected_params=1)
    register_gate_converter("rz", "RZ", arity=1, expected_params=1)
    register_gate_converter("p", "P", arity=1, expected_params=1)
    register_gate_converter("u", "U3", arity=1, expected_params=3)
    register_gate_converter("u3", "U3", arity=1, expected_params=3)
    register_gate_converter("cx", "CX", arity=2)
    register_gate_converter("cy", "CY", arity=2)
    register_gate_converter("cz", "CZ", arity=2)
    register_gate_converter("ccx", "CCX", arity=3)

    return registry


_GATE_CONVERTERS: dict[str, GateConverter] = _build_gate_converter_registry()


def get_supported_qiskit_gates() -> list[str]:
    """Return supported qiskit gate names (lower-case canonical)."""
    return sorted(_GATE_CONVERTERS.keys())


def qiskit_to_qhybrid_json(qc: QiskitCircuit) -> str:
    """Convert a Qiskit circuit to a versioned qhybrid JSON payload."""
    try:
        from qiskit import transpile  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("qiskit is required for qiskit_to_qhybrid_json") from e

    qc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    gates: list[QhybridGatePayload] = []
    for instruction in qc.data:
        operation = instruction.operation
        operation_name = str(operation.name).lower()
        if operation_name in {"barrier", "measure"}:
            continue

        converter = _GATE_CONVERTERS.get(operation_name)
        if converter is None:
            supported = ", ".join(get_supported_qiskit_gates())
            raise NotImplementedError(
                f"Unsupported gate for qhybrid: {operation.name}. "
                f"Supported gates: {supported}"
            )

        op_qubits = [qc.find_bit(q).index for q in instruction.qubits]
        gates.append(converter(operation, op_qubits))

    payload = QhybridCircuitPayload(
        schema_version=QHYBRID_PAYLOAD_VERSION,
        n_qubits=qc.num_qubits,
        gates=gates,
        name=qc.name,
    )
    return json.dumps(payload.as_dict())


__all__ = [
    "QHYBRID_PAYLOAD_VERSION",
    "QhybridCircuitPayload",
    "QhybridGatePayload",
    "GateConverter",
    "get_supported_qiskit_gates",
    "qiskit_to_qhybrid_json",
    "register_gate_converter",
]

