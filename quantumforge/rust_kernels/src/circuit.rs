use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Quantum gate types supported in our circuit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateType {
    /// Identity gate
    I,
    /// Pauli-X (NOT) gate
    X,
    /// Pauli-Y gate
    Y,
    /// Pauli-Z gate
    Z,
    /// Hadamard gate
    H,
    /// S gate (Z^0.5)
    S,
    /// S-dagger gate (Z^-0.5)
    Sdg,
    /// T gate (Z^0.25)
    T,
    /// T-dagger gate (Z^-0.25)
    Tdg,
    /// Rotation around X axis: RX(theta)
    RX(f64),
    /// Rotation around Y axis: RY(theta)
    RY(f64),
    /// Rotation around Z axis: RZ(theta)
    RZ(f64),
    /// Phase gate: P(lambda)
    P(f64),
    /// U3 gate: U3(theta, phi, lambda)
    U3(f64, f64, f64),
    /// Controlled-NOT gate
    CX,
    /// Controlled-Y gate
    CY,
    /// Controlled-Z gate
    CZ,
    /// Toffoli (CCX) gate
    CCX,
}

/// A single quantum gate applied to specific qubits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    pub gate_type: GateType,
    pub qubits: Vec<usize>,
    pub parameters: Vec<f64>,
}

/// A quantum circuit represented as a sequence of gates
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    #[pyo3(get, set)]
    pub n_qubits: usize,
    pub gates: Vec<Gate>,
    #[pyo3(get, set)]
    pub name: Option<String>,
    #[serde(default)]
    pub schema_version: Option<u32>,
}

/// Error types for circuit operations
#[derive(Error, Debug)]
pub enum CircuitError {
    #[error("Invalid qubit index: {0} for {1}-qubit circuit")]
    InvalidQubit(usize, usize),
    #[error("Gate requires {expected} qubits but got {actual}")]
    WrongQubitCount { expected: usize, actual: usize },
    #[error("Circuit serialization error: {0}")]
    SerializationError(String),
    #[error("Missing circuit schema_version; expected {expected}")]
    MissingSchemaVersion {
        expected: u32,
    },
    #[error("Unsupported circuit schema_version: expected {expected}, got {actual}")]
    UnsupportedSchemaVersion {
        expected: u32,
        actual: u32,
    },
}

#[pymethods]
impl QuantumCircuit {
    /// Create a new empty circuit with n_qubits
    #[new]
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
            name: None,
            schema_version: None,
        }
    }

    /// Add a gate to the circuit (generic JSON interface for Python)
    pub fn add_gate_json(&mut self, json: &str) -> PyResult<()> {
        let gate: Gate = serde_json::from_str(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid gate JSON: {}", e)))?;
        
        self.add_gate(gate).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Serialize circuit to JSON
    pub fn to_json_py(&self) -> PyResult<String> {
        self.to_json().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

impl QuantumCircuit {
    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Gate) -> Result<(), CircuitError> {
        // Validate qubits
        for &qubit in &gate.qubits {
            if qubit >= self.n_qubits {
                return Err(CircuitError::InvalidQubit(qubit, self.n_qubits));
            }
        }

        // Validate qubit count for gate type
        let expected_qubits = match gate.gate_type {
            GateType::I | GateType::X | GateType::Y | GateType::Z
            | GateType::H | GateType::S | GateType::Sdg | GateType::T | GateType::Tdg => 1,
            GateType::RX(_) | GateType::RY(_) | GateType::RZ(_) | GateType::P(_) => 1,
            GateType::U3(_, _, _) => 1,
            GateType::CX | GateType::CY | GateType::CZ => 2,
            GateType::CCX => 3,
        };

        if gate.qubits.len() != expected_qubits {
            return Err(CircuitError::WrongQubitCount {
                expected: expected_qubits,
                actual: gate.qubits.len(),
            });
        }

        self.gates.push(gate);
        Ok(())
    }

    /// Get the depth of the circuit (number of sequential layers)
    pub fn depth(&self) -> usize {
        let mut qubit_last_used = vec![0; self.n_qubits];
        let mut current_depth = 0;

        for gate in &self.gates {
            let gate_start = gate.qubits.iter()
                .map(|&q| qubit_last_used[q])
                .max()
                .unwrap_or(0) + 1;

            for &qubit in &gate.qubits {
                qubit_last_used[qubit] = gate_start;
            }

            current_depth = current_depth.max(gate_start);
        }

        current_depth
    }

    /// Count gates by type
    pub fn gate_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            let gate_name = match gate.gate_type {
                GateType::I => "i",
                GateType::X => "x",
                GateType::Y => "y",
                GateType::Z => "z",
                GateType::H => "h",
                GateType::S => "s",
                GateType::Sdg => "sdg",
                GateType::T => "t",
                GateType::Tdg => "tdg",
                GateType::RX(_) => "rx",
                GateType::RY(_) => "ry",
                GateType::RZ(_) => "rz",
                GateType::P(_) => "p",
                GateType::U3(_, _, _) => "u3",
                GateType::CX => "cx",
                GateType::CY => "cy",
                GateType::CZ => "cz",
                GateType::CCX => "ccx",
            }.to_string();

            *counts.entry(gate_name).or_insert(0) += 1;
        }
        counts
    }

    /// Serialize circuit to JSON
    pub fn to_json(&self) -> Result<String, CircuitError> {
        serde_json::to_string(self)
            .map_err(|e| CircuitError::SerializationError(e.to_string()))
    }

    /// Deserialize circuit from JSON
    pub fn from_json(json: &str) -> Result<Self, CircuitError> {
        let mut circuit: QuantumCircuit = serde_json::from_str(json)
            .map_err(|e| CircuitError::SerializationError(e.to_string()))?;
        let version = circuit
            .schema_version
            .ok_or(CircuitError::MissingSchemaVersion {
                expected: SUPPORTED_SCHEMA_VERSION,
            })?;
        if version != SUPPORTED_SCHEMA_VERSION {
            return Err(CircuitError::UnsupportedSchemaVersion {
                expected: SUPPORTED_SCHEMA_VERSION,
                actual: version,
            });
        }
        Ok(circuit)
    }
}

impl Gate {
    /// Create a single-qubit gate
    pub fn single(gate_type: GateType, qubit: usize) -> Self {
        let parameters = match gate_type {
            GateType::RX(theta) | GateType::RY(theta) | GateType::RZ(theta) | GateType::P(theta) => vec![theta],
            GateType::U3(theta, phi, lambda) => vec![theta, phi, lambda],
            _ => vec![],
        };

        Self {
            gate_type,
            qubits: vec![qubit],
            parameters,
        }
    }

    /// Create a two-qubit gate
    pub fn double(gate_type: GateType, control: usize, target: usize) -> Self {
        Self {
            gate_type,
            qubits: vec![control, target],
            parameters: vec![],
        }
    }

    /// Create a three-qubit gate
    pub fn triple(gate_type: GateType, q0: usize, q1: usize, q2: usize) -> Self {
        Self {
            gate_type,
            qubits: vec![q0, q1, q2],
            parameters: vec![],
        }
    }
}