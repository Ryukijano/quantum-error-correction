use rust_kernels::circuit::{Gate, GateType, QuantumCircuit};

/// Different ansatz types for VQE
#[derive(Debug, Clone)]
pub enum AnsatzType {
    /// Simple hardware-efficient ansatz
    HardwareEfficient,
    /// Unitary Coupled Cluster ansatz
    UCC,
}

/// Variational quantum ansatz
#[derive(Debug)]
pub struct Ansatz {
    pub ansatz_type: AnsatzType,
    pub n_qubits: usize,
    pub n_parameters: usize,
}

impl Ansatz {
    /// Create UCC ansatz for H2 (4 qubits)
    pub fn ucc_h2() -> Self {
        Self {
            ansatz_type: AnsatzType::UCC,
            n_qubits: 4,
            // UCC ansatz parameters: 8 single excitations + 4 double excitations
            n_parameters: 12,
        }
    }

    /// Create hardware-efficient ansatz for H2
    pub fn hardware_efficient_h2() -> Self {
        Self {
            ansatz_type: AnsatzType::HardwareEfficient,
            n_qubits: 4,
            // 2 layers × 4 parameters per layer
            n_parameters: 8,
        }
    }

    /// Generate quantum circuit for given parameters
    pub fn circuit(&self, parameters: &[f64]) -> QuantumCircuit {
        match self.ansatz_type {
            AnsatzType::UCC => self.ucc_circuit(parameters),
            AnsatzType::HardwareEfficient => self.hardware_efficient_circuit(parameters),
        }
    }

    /// UCC ansatz circuit for H2
    fn ucc_circuit(&self, parameters: &[f64]) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(self.n_qubits);

        // Start with Hartree-Fock reference |0011⟩ (electrons on atoms 1 and 2)
        // In Jordan-Wigner, this corresponds to creating excitations from |0000⟩
        //
        // Here we prepare |0011⟩ by flipping the least-significant two qubits.
        circuit.add_gate(Gate::single(GateType::X, 0)).unwrap();
        circuit.add_gate(Gate::single(GateType::X, 1)).unwrap();

        // Single excitations: 8 parameters (0-7)
        let single_excitations = vec![
            (0, 2), // 0→2
            (0, 3), // 0→3
            (1, 2), // 1→2
            (1, 3), // 1→3
            (2, 0), // 2→0
            (2, 1), // 2→1
            (3, 0), // 3→0
            (3, 1), // 3→1
        ];

        for (i, (from, to)) in single_excitations.iter().enumerate() {
            if i < parameters.len() {
                self.add_ucc_single_excitation(&mut circuit, *from, *to, parameters[i]);
            }
        }

        // Double excitations: 4 parameters (8-11)
        let double_excitations = vec![
            (0, 1, 2, 3), // (01)→(23)
            (0, 1, 3, 2), // (01)→(32)
            (2, 3, 0, 1), // (23)→(01)
            (2, 3, 1, 0), // (23)→(10)
        ];

        for (i, (i1, j1, i2, j2)) in double_excitations.iter().enumerate() {
            let param_idx = 8 + i;
            if param_idx < parameters.len() {
                self.add_ucc_double_excitation(&mut circuit, *i1, *j1, *i2, *j2, parameters[param_idx]);
            }
        }

        circuit
    }

    /// Add UCC single excitation operator: exp(-iθ/2 (a†_to a_from - a†_from a_to))
    fn add_ucc_single_excitation(&self, circuit: &mut QuantumCircuit, from: usize, to: usize, theta: f64) {
        if from == to {
            return;
        }

        // For simplicity, implement as a rotation on the excitation operator
        // This is an approximation - full UCC would use more sophisticated decomposition

        // RY rotation on target qubit
        circuit.add_gate(Gate::single(GateType::RY(theta), to)).unwrap();

        // Entangle with source qubit
        circuit.add_gate(Gate::double(GateType::CX, from, to)).unwrap();
    }

    /// Add UCC double excitation operator
    fn add_ucc_double_excitation(&self, circuit: &mut QuantumCircuit, i1: usize, j1: usize, i2: usize, j2: usize, theta: f64) {
        // Simplified double excitation implementation
        // Full UCC would use more complex operators

        // Apply rotations
        circuit.add_gate(Gate::single(GateType::RY(theta * 0.5), i2)).unwrap();
        circuit.add_gate(Gate::single(GateType::RY(theta * 0.5), j2)).unwrap();

        // Entangling gates
        circuit.add_gate(Gate::double(GateType::CX, i1, i2)).unwrap();
        circuit.add_gate(Gate::double(GateType::CX, j1, j2)).unwrap();
    }

    /// Hardware-efficient ansatz circuit
    fn hardware_efficient_circuit(&self, parameters: &[f64]) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(self.n_qubits);

        // Layer 1: RY rotations on all qubits
        for i in 0..self.n_qubits {
            let param_idx = i;
            if param_idx < parameters.len() {
                circuit.add_gate(Gate::single(GateType::RY(parameters[param_idx]), i)).unwrap();
            }
        }

        // Layer 1: Entangling gates
        for i in 0..self.n_qubits - 1 {
            circuit.add_gate(Gate::double(GateType::CX, i, i + 1)).unwrap();
        }

        // Layer 2: More RY rotations
        for i in 0..self.n_qubits {
            let param_idx = self.n_qubits + i;
            if param_idx < parameters.len() {
                circuit.add_gate(Gate::single(GateType::RY(parameters[param_idx]), i)).unwrap();
            }
        }

        // Layer 2: Entangling gates (reverse direction)
        for i in (1..self.n_qubits).rev() {
            circuit.add_gate(Gate::double(GateType::CX, i - 1, i)).unwrap();
        }

        circuit
    }
}
