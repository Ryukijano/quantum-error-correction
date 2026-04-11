use ndarray::Array2;
use rust_kernels::circuit::CircuitError;
use rust_kernels::execution::ExecutionBackend;
use serde::{Deserialize, Serialize};

/// Pauli operator (I, X, Y, Z)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// A Pauli string term in the Hamiltonian
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliTerm {
    pub pauli_string: Vec<Pauli>,
    pub coefficient: f64,
}

/// Molecular Hamiltonian in Pauli operator form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hamiltonian {
    pub n_qubits: usize,
    pub terms: Vec<PauliTerm>,
}

impl Hamiltonian {
    /// Create H2 minimal Hamiltonian (2-qubit approximation)
    pub fn h2_minimal() -> Self {
        Self {
            n_qubits: 2,
            terms: vec![
                PauliTerm { pauli_string: vec![Pauli::I, Pauli::I], coefficient: -1.052373245772859 },
                PauliTerm { pauli_string: vec![Pauli::I, Pauli::Z], coefficient: 0.39793742484318045 },
                PauliTerm { pauli_string: vec![Pauli::Z, Pauli::I], coefficient: -0.39793742484318045 },
                PauliTerm { pauli_string: vec![Pauli::Z, Pauli::Z], coefficient: -0.01128010425623538 },
                PauliTerm { pauli_string: vec![Pauli::X, Pauli::X], coefficient: 0.18093119978423156 },
            ],
        }
    }

    /// Create H2 Hamiltonian (STO-3G, Jordan-Wigner mapped)
    /// This is a simplified 4-qubit H2 Hamiltonian for demonstration
    pub fn h2_sto3g() -> Self {
        let mut terms = Vec::new();

        // Nuclear repulsion (constant term)
        terms.push(PauliTerm {
            pauli_string: vec![Pauli::I, Pauli::I, Pauli::I, Pauli::I],
            coefficient: 0.7151043390810812,
        });

        // One-electron terms
        let one_electron_terms = vec![
            (vec![Pauli::X, Pauli::I, Pauli::X, Pauli::I], -0.327608189674809),
            (vec![Pauli::X, Pauli::I, Pauli::I, Pauli::X], 0.155426690779928),
            (vec![Pauli::I, Pauli::X, Pauli::X, Pauli::I], -0.327608189674809),
            (vec![Pauli::I, Pauli::X, Pauli::I, Pauli::X], 0.155426690779928),
            (vec![Pauli::Z, Pauli::I, Pauli::Z, Pauli::I], 0.160841154845547),
            (vec![Pauli::Z, Pauli::I, Pauli::I, Pauli::Z], 0.172183932619155),
            (vec![Pauli::I, Pauli::Z, Pauli::Z, Pauli::I], 0.160841154845547),
            (vec![Pauli::I, Pauli::Z, Pauli::I, Pauli::Z], 0.172183932619155),
        ];

        for (pauli_string, coeff) in one_electron_terms {
            terms.push(PauliTerm {
                pauli_string,
                coefficient: coeff,
            });
        }

        // Two-electron terms
        let two_electron_terms = vec![
            (vec![Pauli::X, Pauli::X, Pauli::X, Pauli::X], 0.381860580620085),
            (vec![Pauli::X, Pauli::Y, Pauli::X, Pauli::Y], 0.381860580620085),
            (vec![Pauli::Y, Pauli::X, Pauli::Y, Pauli::X], 0.381860580620085),
            (vec![Pauli::Y, Pauli::Y, Pauli::Y, Pauli::Y], 0.381860580620085),
            (vec![Pauli::Z, Pauli::Z, Pauli::Z, Pauli::Z], 0.217661055668174),
        ];

        for (pauli_string, coeff) in two_electron_terms {
            terms.push(PauliTerm {
                pauli_string,
                coefficient: coeff,
            });
        }

        Self {
            n_qubits: 4,
            terms,
        }
    }

    /// Evaluate expectation value ⟨ψ|H|ψ⟩ on a statevector `state` encoded as shape `(2^n, 2)`
    /// with columns `[re, im]`.
    pub fn expectation_value(
        &self,
        state: &Array2<f64>,
        backend: &dyn ExecutionBackend,
    ) -> Result<f64, CircuitError> {
        let mut total_energy = 0.0;

        for term in &self.terms {
            let pauli_str: String = term
                .pauli_string
                .iter()
                .map(|p| match p {
                    Pauli::I => 'I',
                    Pauli::X => 'X',
                    Pauli::Y => 'Y',
                    Pauli::Z => 'Z',
                })
                .collect();

            let val = backend.expectation_value_pauli_string(state, &pauli_str)?;
            total_energy += term.coefficient * val;
        }

        Ok(total_energy)
    }
}
