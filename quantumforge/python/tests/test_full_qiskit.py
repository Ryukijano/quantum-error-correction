import numpy as np
import pytest
from qhybrid_kernels import qiskit_to_qhybrid_json, execute_quantum_circuit
from qhybrid_kernels.circuit import QHYBRID_PAYLOAD_VERSION

pytest.importorskip("qiskit")

def test_qiskit_to_rust_execution():
    from qiskit import QuantumCircuit as QiskitCircuit

    # Create Bell state in Qiskit
    qc = QiskitCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Convert to qhybrid JSON
    circuit_json = qiskit_to_qhybrid_json(qc)

    # Execute in Rust
    result_ri = execute_quantum_circuit(circuit_json)

    # Convert back to complex for verification
    result = result_ri[:, 0] + 1j * result_ri[:, 1]

    # Expected amplitudes: [1/√2, 0, 0, 1/√2] for |00⟩ + |11⟩
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

    assert np.allclose(np.abs(result), np.abs(expected), atol=1e-10)

def test_simple_hadamard():
    from qiskit import QuantumCircuit as QiskitCircuit

    # Just H gate on single qubit
    qc = QiskitCircuit(1)
    qc.h(0)

    circuit_json = qiskit_to_qhybrid_json(qc)
    result_ri = execute_quantum_circuit(circuit_json)
    result = result_ri[:, 0] + 1j * result_ri[:, 1]

    # H|0⟩ = 1/√2(|0⟩ + |1⟩)
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    print(f"H gate result: {result}")
    print(f"H gate expected: {expected}")

    assert np.allclose(np.abs(result), np.abs(expected), atol=1e-10)

def test_manual_cnot():
    """Test CNOT manually to debug the issue"""
    from qhybrid_kernels import execute_quantum_circuit
    import json

    # Manual circuit: H(0) then CNOT(0,1)
    circuit_data = {
        "n_qubits": 2,
        "schema_version": QHYBRID_PAYLOAD_VERSION,
        "gates": [
            {"gate_type": "H", "qubits": [0], "parameters": []},
            {"gate_type": "CX", "qubits": [0, 1], "parameters": []}
        ],
        "name": "bell_test"
    }

    circuit_json = json.dumps(circuit_data)
    print(f"Manual circuit JSON: {circuit_json}")

    result_ri = execute_quantum_circuit(circuit_json)
    result = result_ri[:, 0] + 1j * result_ri[:, 1]

    print(f"Manual CNOT result: {result}")
    print(f"Manual CNOT abs: {np.abs(result)}")

    # Should be [1/√2, 0, 0, 1/√2]
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    print(f"Expected: {expected}")

    # Check which indices have non-zero amplitudes
    nonzero_indices = [i for i, amp in enumerate(result) if abs(amp) > 1e-10]
    print(f"Non-zero indices: {nonzero_indices}")

    assert np.allclose(np.abs(result), np.abs(expected), atol=1e-10)
    
def test_qiskit_rotations():
    from qiskit import QuantumCircuit as QiskitCircuit
    
    qc = QiskitCircuit(1)
    qc.rx(np.pi/2, 0)
    
    circuit_json = qiskit_to_qhybrid_json(qc)
    result_ri = execute_quantum_circuit(circuit_json)
    result = result_ri[:, 0] + 1j * result_ri[:, 1]
    
    # RX(π/2)|0⟩ = cos(π/4)|0⟩ - i sin(π/4)|1⟩ = 1/√2|0⟩ - i/√2|1⟩
    expected = np.array([1/np.sqrt(2), -1j/np.sqrt(2)])
    
    assert np.allclose(result, expected, atol=1e-10)

