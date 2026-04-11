use crate::circuit::{CircuitError, Gate, GateType, QuantumCircuit};
use ndarray::{Array2, ArrayViewMut2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Complex number type for quantum amplitudes
type C64 = Complex64;

/// Statevector representation: shape (2^n, 2) with [real, imag] columns
pub type StateVector = Array2<f64>;

/// Abstraction over execution/evaluation backends.
///
/// Keeping this trait in the rust_kernels crate lets higher-level crates
/// depend on a stable contract instead of direct module-level function calls.
pub trait ExecutionBackend: Send + Sync {
    fn execute_circuit(
        &self,
        circuit: &QuantumCircuit,
        initial_state: Option<&Array2<f64>>,
    ) -> Result<StateVector, CircuitError>;

    fn expectation_value_pauli_string(
        &self,
        state: &Array2<f64>,
        pauli_string: &str,
    ) -> Result<f64, CircuitError>;
}

/// Default execution backend backed by the concrete rust-kernels engine.
#[derive(Debug, Default, Clone, Copy)]
pub struct RustExecutionBackend;

impl ExecutionBackend for RustExecutionBackend {
    fn execute_circuit(
        &self,
        circuit: &QuantumCircuit,
        initial_state: Option<&Array2<f64>>,
    ) -> Result<StateVector, CircuitError> {
        execute_circuit(circuit, initial_state)
    }

    fn expectation_value_pauli_string(
        &self,
        state: &Array2<f64>,
        pauli_string: &str,
    ) -> Result<f64, CircuitError> {
        expectation_value_pauli_string(state, pauli_string)
    }
}

/// Execute a quantum circuit on a statevector
pub fn execute_circuit<T: ndarray::Data<Elem = f64>>(circuit: &QuantumCircuit, initial_state: Option<&ndarray::ArrayBase<T, ndarray::Ix2>>) -> Result<StateVector, CircuitError> {
    let n_qubits = circuit.n_qubits;
    let dim = 1usize << n_qubits;

    // Initialize to |0...0⟩ if no initial state provided
    let mut state = if let Some(init) = initial_state {
        init.to_owned()
    } else {
        let mut state = Array2::zeros((dim, 2));
        state[[0, 0]] = 1.0; // |0⟩ state
        state
    };

    // Apply each gate
    for gate in &circuit.gates {
        apply_gate(&mut state.view_mut(), gate, n_qubits)?;
    }

    Ok(state)
}

/// Apply a single gate to the statevector
fn apply_gate(state: &mut ArrayViewMut2<f64>, gate: &Gate, n_qubits: usize) -> Result<(), CircuitError> {
    match gate.gate_type {
        GateType::I => Ok(()), // Identity does nothing

        // Single-qubit Pauli gates
        GateType::X => { apply_pauli_x(state, gate.qubits[0], n_qubits); Ok(()) }
        GateType::Y => { apply_pauli_y(state, gate.qubits[0], n_qubits); Ok(()) }
        GateType::Z => { apply_pauli_z(state, gate.qubits[0], n_qubits); Ok(()) }

        // Single-qubit Clifford gates
        GateType::H => { apply_hadamard(state, gate.qubits[0], n_qubits); Ok(()) }
        GateType::S => { apply_s_gate(state, gate.qubits[0], n_qubits); Ok(()) }
        GateType::Sdg => { apply_sdg_gate(state, gate.qubits[0], n_qubits); Ok(()) }
        GateType::T => { apply_t_gate(state, gate.qubits[0], n_qubits); Ok(()) }
        GateType::Tdg => { apply_tdg_gate(state, gate.qubits[0], n_qubits); Ok(()) }

        // Single-qubit rotations
        GateType::RX(theta) => { apply_rotation_x(state, gate.qubits[0], theta, n_qubits); Ok(()) }
        GateType::RY(theta) => { apply_rotation_y(state, gate.qubits[0], theta, n_qubits); Ok(()) }
        GateType::RZ(theta) => { apply_rotation_z(state, gate.qubits[0], theta, n_qubits); Ok(()) }
        GateType::P(lambda) => { apply_phase_gate(state, gate.qubits[0], lambda, n_qubits); Ok(()) }
        GateType::U3(theta, phi, lambda) => { apply_u3_gate(state, gate.qubits[0], theta, phi, lambda, n_qubits); Ok(()) }

        // Two-qubit gates
        GateType::CX => { apply_cnot(state, gate.qubits[0], gate.qubits[1], n_qubits); Ok(()) }
        GateType::CY => { apply_controlled_y(state, gate.qubits[0], gate.qubits[1], n_qubits); Ok(()) }
        GateType::CZ => { apply_controlled_z(state, gate.qubits[0], gate.qubits[1], n_qubits); Ok(()) }

        // Three-qubit gates
        GateType::CCX => { apply_toffoli(state, gate.qubits[0], gate.qubits[1], gate.qubits[2], n_qubits); Ok(()) }
    }
}

/// Get complex amplitude at index i from any array view
fn get_amplitude<T: ndarray::Data<Elem = f64>>(state: &ndarray::ArrayBase<T, ndarray::Ix2>, i: usize) -> C64 {
    C64::new(state[[i, 0]], state[[i, 1]])
}

/// Set complex amplitude at index i
fn set_amplitude(state: &mut ArrayViewMut2<f64>, i: usize, amp: C64) {
    state[[i, 0]] = amp.re;
    state[[i, 1]] = amp.im;
}

/// Apply Pauli-X gate (NOT gate)
pub fn apply_pauli_x(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;

    // Swap amplitudes where the qubit bit differs
    for i in 0..dim {
        if (i & bit) == 0 {
            let j = i ^ bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            set_amplitude(state, i, b);
            set_amplitude(state, j, a);
        }
    }
}

/// Apply Pauli-Y gate
pub fn apply_pauli_y(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;

    for i in 0..dim {
        if (i & bit) == 0 {
            let j = i ^ bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            // Y|i⟩ = i|j⟩, Y|j⟩ = -i|i⟩ where j = i ⊕ bit
            set_amplitude(state, i, C64::new(0.0, 1.0) * b);
            set_amplitude(state, j, -C64::new(0.0, 1.0) * a);
        }
    }
}

/// Apply Pauli-Z gate
pub fn apply_pauli_z(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;

    for i in 0..dim {
        if (i & bit) != 0 {
            let amp = get_amplitude(state, i);
            set_amplitude(state, i, -amp);
        }
    }
}

/// Apply Hadamard gate
fn apply_hadamard(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;
    let sqrt_2_inv = 1.0 / (2.0f64).sqrt();

    for i in 0..dim {
        if (i & bit) == 0 {
            let j = i ^ bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            // H|i⟩ = (|i⟩ + |j⟩)/√2, H|j⟩ = (|i⟩ - |j⟩)/√2
            set_amplitude(state, i, sqrt_2_inv * (a + b));
            set_amplitude(state, j, sqrt_2_inv * (a - b));
        }
    }
}

/// Apply S gate (Z^0.5)
fn apply_s_gate(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;

    for i in 0..dim {
        if (i & bit) != 0 {
            let amp = get_amplitude(state, i);
            set_amplitude(state, i, C64::new(0.0, 1.0) * amp); // multiply by i
        }
    }
}

/// Apply S-dagger gate (Z^-0.5)
fn apply_sdg_gate(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;

    for i in 0..dim {
        if (i & bit) != 0 {
            let amp = get_amplitude(state, i);
            set_amplitude(state, i, -C64::new(0.0, 1.0) * amp); // multiply by -i
        }
    }
}

/// Apply T gate (Z^0.25)
fn apply_t_gate(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;
    let e_ipi_4 = C64::from_polar(1.0, PI / 4.0);

    for i in 0..dim {
        if (i & bit) != 0 {
            let amp = get_amplitude(state, i);
            set_amplitude(state, i, e_ipi_4 * amp);
        }
    }
}

/// Apply T-dagger gate (Z^-0.25)
fn apply_tdg_gate(state: &mut ArrayViewMut2<f64>, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;
    let e_ipi_4 = C64::from_polar(1.0, -PI / 4.0);

    for i in 0..dim {
        if (i & bit) != 0 {
            let amp = get_amplitude(state, i);
            set_amplitude(state, i, e_ipi_4 * amp);
        }
    }
}

/// Apply rotation around X axis
fn apply_rotation_x(state: &mut ArrayViewMut2<f64>, qubit: usize, theta: f64, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();

    for i in 0..dim {
        if (i & bit) == 0 {
            let j = i ^ bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            set_amplitude(state, i, cos_half * a - C64::new(0.0, sin_half) * b);
            set_amplitude(state, j, -C64::new(0.0, sin_half) * a + cos_half * b);
        }
    }
}

/// Apply rotation around Y axis
fn apply_rotation_y(state: &mut ArrayViewMut2<f64>, qubit: usize, theta: f64, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();

    for i in 0..dim {
        if (i & bit) == 0 {
            let j = i ^ bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            set_amplitude(state, i, cos_half * a - sin_half * b);
            set_amplitude(state, j, sin_half * a + cos_half * b);
        }
    }
}

/// Apply rotation around Z axis
fn apply_rotation_z(state: &mut ArrayViewMut2<f64>, qubit: usize, theta: f64, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;
    let e_minus_i_theta_2 = C64::from_polar(1.0, -theta / 2.0);
    let e_plus_i_theta_2 = C64::from_polar(1.0, theta / 2.0);

    for i in 0..dim {
        if (i & bit) == 0 {
            let j = i ^ bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            set_amplitude(state, i, e_minus_i_theta_2 * a);
            set_amplitude(state, j, e_plus_i_theta_2 * b);
        }
    }
}

/// Apply phase gate
fn apply_phase_gate(state: &mut ArrayViewMut2<f64>, qubit: usize, lambda: f64, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;
    let phase = C64::from_polar(1.0, lambda);

    for i in 0..dim {
        if (i & bit) != 0 {
            let amp = get_amplitude(state, i);
            set_amplitude(state, i, phase * amp);
        }
    }
}

/// Apply U3 gate (general single-qubit unitary)
fn apply_u3_gate(state: &mut ArrayViewMut2<f64>, qubit: usize, theta: f64, phi: f64, lambda: f64, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let bit = 1 << qubit;

    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    let e_i_phi_plus_lambda_2 = C64::from_polar(1.0, (phi + lambda) / 2.0);
    let e_i_phi_minus_lambda_2 = C64::from_polar(1.0, (phi - lambda) / 2.0);

    for i in 0..dim {
        if (i & bit) == 0 {
            let j = i ^ bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            set_amplitude(state, i, cos_half * a - sin_half * e_i_phi_minus_lambda_2 * b);
            set_amplitude(state, j, sin_half * e_i_phi_plus_lambda_2 * a + cos_half * b);
        }
    }
}

/// Apply CNOT gate
fn apply_cnot(state: &mut ArrayViewMut2<f64>, control: usize, target: usize, n_qubits: usize) {
    if control == target {
        return; // Invalid
    }

    let dim = 1 << n_qubits;
    let control_bit = 1 << control;
    let target_bit = 1 << target;

    // Iterate over all basis states where control=1 AND target=0
    // This avoids double-counting the swaps
    for i in 0..dim {
        if (i & control_bit) != 0 && (i & target_bit) == 0 {
            // Control is |1⟩, target is |0⟩, so flip target to |1⟩
            let j = i ^ target_bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            set_amplitude(state, i, b);
            set_amplitude(state, j, a);
        }
    }
}

/// Apply controlled-Y gate
fn apply_controlled_y(state: &mut ArrayViewMut2<f64>, control: usize, target: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let control_bit = 1 << control;
    let target_bit = 1 << target;

    for i in 0..dim {
        if (i & control_bit) != 0 && (i & target_bit) == 0 {
            let j = i ^ target_bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            // Apply Y to target when control is |1⟩
            set_amplitude(state, i, C64::new(0.0, 1.0) * b);
            set_amplitude(state, j, -C64::new(0.0, 1.0) * a);
        }
    }
}

/// Apply controlled-Z gate
fn apply_controlled_z(state: &mut ArrayViewMut2<f64>, control: usize, target: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let control_bit = 1 << control;
    let target_bit = 1 << target;

    for i in 0..dim {
        if (i & control_bit) != 0 && (i & target_bit) != 0 {
            let amp = get_amplitude(state, i);
            set_amplitude(state, i, -amp);
        }
    }
}

/// Apply Toffoli (CCX) gate
fn apply_toffoli(state: &mut ArrayViewMut2<f64>, c1: usize, c2: usize, target: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let c1_bit = 1 << c1;
    let c2_bit = 1 << c2;
    let target_bit = 1 << target;

    for i in 0..dim {
        if (i & c1_bit) != 0 && (i & c2_bit) != 0 {
            // Both controls are |1⟩, so flip target
            let j = i ^ target_bit;
            let a = get_amplitude(state, i);
            let b = get_amplitude(state, j);
            set_amplitude(state, i, b);
            set_amplitude(state, j, a);
        }
    }
}

/// Compute expectation value ⟨ψ|H|ψ⟩ for a Pauli string
pub fn expectation_value_pauli_string<T: ndarray::Data<Elem = f64>>(state: &ndarray::ArrayBase<T, ndarray::Ix2>, pauli_string: &str) -> Result<f64, CircuitError> {
    if pauli_string.is_empty() {
        return Ok(1.0); // Identity
    }

    let n_qubits = (state.shape()[0] as f64).log2() as usize;
    if pauli_string.len() != n_qubits {
        return Err(CircuitError::WrongQubitCount {
            expected: n_qubits,
            actual: pauli_string.len(),
        });
    }

    // Create temporary circuit for Pauli measurement
    let mut circuit = QuantumCircuit::new(n_qubits);

    // Add basis rotations for measurement
    for (i, pauli) in pauli_string.chars().enumerate() {
        match pauli {
            'I' => {} // No rotation needed
            'X' => { circuit.add_gate(Gate::single(GateType::RY(-PI/2.0), i))?; }
            'Y' => { circuit.add_gate(Gate::single(GateType::RX(PI/2.0), i))?; }
            'Z' => {} // Z measurement is in computational basis
            _ => return Err(CircuitError::SerializationError(format!("Invalid Pauli: {}", pauli))),
        }
    }

    // Execute rotations
    let rotated_state = execute_circuit(&circuit, Some(state))?;

    // Compute ⟨P⟩ = ∑_i prob(i) * eigenvalue(i)
    // where eigenvalue(i) = ∏_{q: P_q != I} (-1)^{bit_q}
    let dim = 1 << n_qubits;
    let mut total_expectation = 0.0;
    
    for i in 0..dim {
        let amp = get_amplitude(&rotated_state, i);
        let prob = amp.norm_sqr();
        
        let mut eigenvalue = 1.0;
        for (q, pauli) in pauli_string.chars().enumerate() {
            if pauli != 'I' {
                let bit = (i >> q) & 1;
                if bit == 1 {
                    eigenvalue *= -1.0;
                }
            }
        }
        total_expectation += prob * eigenvalue;
    }

    Ok(total_expectation)
}

