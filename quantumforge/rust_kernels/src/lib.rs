#![allow(unsafe_op_in_unsafe_fn)]

use ndarray::{Array1, Array2, Array3, ArrayViewMut2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArray4,
};
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use thiserror::Error;

pub mod circuit;
pub mod execution;

pub use circuit::*;
pub use execution::*;

/// Custom error type for kernel operations
#[derive(Error, Debug)]
pub enum KernelError {
    #[error("Circuit error: {0}")]
    CircuitError(#[from] crate::circuit::CircuitError),
    #[error("Numpy array conversion error: {0}")]
    NumpyError(String),
    #[error("Invalid dimensions: expected {expected}, got {actual}")]
    DimensionError { expected: String, actual: String },
    #[error("Qubit index {0} out of range for {1}-qubit system")]
    QubitRangeError(usize, usize),
    #[error("Probability validation failed: {0}")]
    ProbabilityError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Square each element on CPU (baseline kernel; also validates numpy interop).
#[pyfunction]
fn square_u32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<&'py PyArray1<u32>> {
    let x = x.as_array();

    let out: Array1<u32> = py.allow_threads(|| {
        if let Some(x_slice) = x.as_slice() {
            let mut out = vec![0u32; x_slice.len()];
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, dst)| *dst = x_slice[i].wrapping_mul(x_slice[i]));
            Array1::from(out)
        } else {
            x.mapv(|v| v.wrapping_mul(v))
        }
    });

    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

/// Apply a single-qubit Pauli channel (I/X/Y/Z) to a statevector via Monte Carlo sampling.
///
/// - `psi`: complex128 statevector encoded as shape (2^n, 2) with columns [re, im].
/// - `target_qubit`: 0 = least-significant bit.
/// - `probs`: length-4 probabilities for [pI, pX, pY, pZ]. Must sum ~ 1.
/// - `seed`: RNG seed for reproducibility.
///
/// Returns a new statevector with the sampled Pauli applied.
#[pyfunction]
fn apply_pauli_channel_statevector<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
    n_qubits: usize,
    target_qubit: usize,
    probs: PyReadonlyArray1<'py, f64>,
    seed: u64,
) -> PyResult<&'py PyArray2<f64>> {
    let psi = psi.as_array();
    let probs = probs.as_array();

    if psi.ndim() != 2 || psi.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "psi must have shape (2^n, 2) with columns [re, im]",
        ));
    }
    let dim = 1usize << n_qubits;
    if psi.shape()[0] != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "psi first dimension must be 2^n_qubits",
        ));
    }
    if target_qubit >= n_qubits {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_qubit out of range",
        ));
    }
    if probs.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "probs must have length 4",
        ));
    }

    let w = WeightedIndex::new(probs.iter().cloned()).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid probs: {e}"))
    })?;
    let mut rng = StdRng::seed_from_u64(seed);
    let choice = w.sample(&mut rng);

    let out: Array2<f64> = py.allow_threads(|| {
        // Copy into owned array and apply the chosen Pauli.
        let mut out: Array2<f64> = psi.to_owned();
        match choice {
            0 => {} // I
            1 => apply_x(out.view_mut(), target_qubit),
            2 => apply_y(out.view_mut(), target_qubit),
            3 => apply_z(out.view_mut(), target_qubit),
            _ => unreachable!(),
        }
        out
    });

    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

#[derive(Clone, Copy, Debug, Default)]
struct C64 {
    re: f64,
    im: f64,
}

#[inline]
fn c(re: f64, im: f64) -> C64 {
    C64 { re, im }
}

#[inline]
fn c_add(a: C64, b: C64) -> C64 {
    c(a.re + b.re, a.im + b.im)
}

#[inline]
fn c_mul(a: C64, b: C64) -> C64 {
    c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn c_conj(a: C64) -> C64 {
    c(a.re, -a.im)
}

/// Apply a single-qubit Kraus channel to a density matrix:
///
/// ρ' = Σ_i K_i ρ K_i†
///
/// - `rho`: complex128 density matrix encoded as shape (2^n, 2^n, 2) with last dim [re, im]
/// - `kraus_ops`: complex128 Kraus matrices encoded as shape (k, 2, 2, 2)
/// - `target_qubit`: 0 = least-significant bit
#[pyfunction]
fn apply_kraus_1q_density_matrix<'py>(
    py: Python<'py>,
    rho: PyReadonlyArray3<'py, f64>,
    n_qubits: usize,
    target_qubit: usize,
    kraus_ops: PyReadonlyArray4<'py, f64>,
) -> PyResult<&'py PyArray3<f64>> {
    let rho = rho.as_array();
    let kraus = kraus_ops.as_array();

    if rho.ndim() != 3 || rho.shape()[2] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rho must have shape (2^n, 2^n, 2) with last dim [re, im]",
        ));
    }
    let dim = 1usize << n_qubits;
    if rho.shape()[0] != dim || rho.shape()[1] != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rho must have shape (2^n, 2^n, 2)",
        ));
    }
    if target_qubit >= n_qubits {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_qubit out of range",
        ));
    }

    if kraus.ndim() != 4 || kraus.shape()[1] != 2 || kraus.shape()[2] != 2 || kraus.shape()[3] != 2
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "kraus_ops must have shape (k, 2, 2, 2)",
        ));
    }

    let out: Array3<f64> = py.allow_threads(|| {
        let bit = 1usize << target_qubit;
        let mut out = Array3::<f64>::zeros((dim, dim, 2));

        for r0 in 0..dim {
            if (r0 & bit) != 0 {
                continue;
            }
            let r1 = r0 ^ bit;

            for c0 in 0..dim {
                if (c0 & bit) != 0 {
                    continue;
                }
                let c1 = c0 ^ bit;

                // Load 2x2 block
                let r00 = c(rho[[r0, c0, 0]], rho[[r0, c0, 1]]);
                let r01 = c(rho[[r0, c1, 0]], rho[[r0, c1, 1]]);
                let r10 = c(rho[[r1, c0, 0]], rho[[r1, c0, 1]]);
                let r11 = c(rho[[r1, c1, 0]], rho[[r1, c1, 1]]);

                let mut o00 = C64::default();
                let mut o01 = C64::default();
                let mut o10 = C64::default();
                let mut o11 = C64::default();

                for ki in 0..kraus.shape()[0] {
                    let k00 = c(kraus[[ki, 0, 0, 0]], kraus[[ki, 0, 0, 1]]);
                    let k01 = c(kraus[[ki, 0, 1, 0]], kraus[[ki, 0, 1, 1]]);
                    let k10 = c(kraus[[ki, 1, 0, 0]], kraus[[ki, 1, 0, 1]]);
                    let k11 = c(kraus[[ki, 1, 1, 0]], kraus[[ki, 1, 1, 1]]);

                    // temp = K * rho_block
                    let t00 = c_add(c_mul(k00, r00), c_mul(k01, r10));
                    let t01 = c_add(c_mul(k00, r01), c_mul(k01, r11));
                    let t10 = c_add(c_mul(k10, r00), c_mul(k11, r10));
                    let t11 = c_add(c_mul(k10, r01), c_mul(k11, r11));

                    // out_block = temp * K†
                    let ck00 = c_conj(k00);
                    let ck01 = c_conj(k01);
                    let ck10 = c_conj(k10);
                    let ck11 = c_conj(k11);

                    o00 = c_add(o00, c_add(c_mul(t00, ck00), c_mul(t01, ck01)));
                    o01 = c_add(o01, c_add(c_mul(t00, ck10), c_mul(t01, ck11)));
                    o10 = c_add(o10, c_add(c_mul(t10, ck00), c_mul(t11, ck01)));
                    o11 = c_add(o11, c_add(c_mul(t10, ck10), c_mul(t11, ck11)));
                }

                out[[r0, c0, 0]] = o00.re;
                out[[r0, c0, 1]] = o00.im;
                out[[r0, c1, 0]] = o01.re;
                out[[r0, c1, 1]] = o01.im;
                out[[r1, c0, 0]] = o10.re;
                out[[r1, c0, 1]] = o10.im;
                out[[r1, c1, 0]] = o11.re;
                out[[r1, c1, 1]] = o11.im;
            }
        }

        out
    });

    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

fn apply_x(mut psi: ArrayViewMut2<f64>, target: usize) {
    let dim = psi.shape()[0];
    let bit = 1usize << target;
    // swap amplitudes where that bit differs: i <-> i^bit
    // To avoid double swaps, only process i where bit is 0.
    for i in 0..dim {
        if (i & bit) != 0 {
            continue;
        }
        let j = i ^ bit;
        let a_re = psi[[i, 0]];
        let a_im = psi[[i, 1]];
        let b_re = psi[[j, 0]];
        let b_im = psi[[j, 1]];
        psi[[i, 0]] = b_re;
        psi[[i, 1]] = b_im;
        psi[[j, 0]] = a_re;
        psi[[j, 1]] = a_im;
    }
}

fn apply_z(mut psi: ArrayViewMut2<f64>, target: usize) {
    let dim = psi.shape()[0];
    let bit = 1usize << target;
    for i in 0..dim {
        if (i & bit) == 0 {
            continue;
        }
        psi[[i, 0]] = -psi[[i, 0]];
        psi[[i, 1]] = -psi[[i, 1]];
    }
}

fn apply_y(mut psi: ArrayViewMut2<f64>, target: usize) {
    // Y = iXZ; action:
    // |0> -> i|1>, |1> -> -i|0>
    let dim = psi.shape()[0];
    let bit = 1usize << target;
    for i in 0..dim {
        if (i & bit) != 0 {
            continue;
        }
        let j = i ^ bit;
        // a = psi[i], b = psi[j]
        let a_re = psi[[i, 0]];
        let a_im = psi[[i, 1]];
        let b_re = psi[[j, 0]];
        let b_im = psi[[j, 1]];

        // psi[i] = -i * b
        // (-i)(b_re + i b_im) = b_im - i b_re
        psi[[i, 0]] = b_im;
        psi[[i, 1]] = -b_re;

        // psi[j] = i * a
        // i(a_re + i a_im) = -a_im + i a_re
        psi[[j, 0]] = -a_im;
        psi[[j, 1]] = a_re;
    }
}

/// Execute a quantum circuit on a statevector
#[pyfunction(signature = (circuit_json, initial_state=None))]
fn execute_quantum_circuit<'py>(
    py: Python<'py>,
    circuit_json: &str,
    initial_state: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<&'py PyArray2<f64>> {
    let circuit = QuantumCircuit::from_json(circuit_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Circuit parsing error: {}", e)))?;

    // Copy initial state to owned to move into thread if provided
    let init_state_owned = initial_state.map(|arr| arr.as_array().to_owned());

    let final_state = py.allow_threads(|| {
        execute_circuit(&circuit, init_state_owned.as_ref())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Execution error: {}", e)))
    })?;

    Ok(final_state.into_pyarray_bound(py).into_gil_ref())
}

/// Apply multi-qubit correlated Pauli noise
#[pyfunction]
fn apply_correlated_pauli_noise_statevector<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
    n_qubits: usize,
    error_probs: PyReadonlyArray2<'py, f64>, // Shape: (2^n, 2^n) correlation matrix
    seed: u64,
) -> PyResult<&'py PyArray2<f64>> {
    let psi_owned = psi.as_array().to_owned();
    let error_probs_owned = error_probs.as_array().to_owned();

    if error_probs_owned.shape() != &[1 << n_qubits, 1 << n_qubits] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "error_probs must be (2^n, 2^n)"
        ));
    }

    let out = py.allow_threads(|| {
        // Sample error pattern from correlation matrix
        let error_pattern = sample_correlated_error(&error_probs_owned, seed);

        // Apply the sampled Pauli error
        let mut out = psi_owned;
        apply_pauli_error_pattern(out.view_mut(), error_pattern, n_qubits);
        out
    });

    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

/// Apply CNOT gate error (bit flip on control + target with correlation)
#[pyfunction]
fn apply_cnot_error_statevector<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
    n_qubits: usize,
    control: usize,
    target: usize,
    error_prob: f64,
    seed: u64,
) -> PyResult<&'py PyArray2<f64>> {
    let mut out = psi.as_array().to_owned();

    py.allow_threads(|| {
        apply_cnot_gate_error(out.view_mut(), n_qubits, control, target, error_prob, seed);
    });

    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

/// Compute expectation value of a Pauli string
#[pyfunction]
fn expectation_value_pauli_string_py<'py>(
    py: Python<'py>,
    state: PyReadonlyArray2<'py, f64>,
    pauli_string: &str,
) -> PyResult<f64> {
    let state_owned = state.as_array().to_owned();
    py.allow_threads(|| {
        expectation_value_pauli_string(&state_owned, pauli_string)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Expectation error: {}", e)))
    })
}

fn sample_correlated_error<T: ndarray::Data<Elem = f64>>(
    error_probs: &ndarray::ArrayBase<T, ndarray::Ix2>,
    seed: u64,
) -> u64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let flat_probs = error_probs.as_slice().unwrap();
    let dist = WeightedIndex::new(flat_probs.iter().cloned()).unwrap();
    dist.sample(&mut rng) as u64
}

fn apply_pauli_error_pattern(mut psi: ArrayViewMut2<f64>, error_pattern: u64, n_qubits: usize) {
    for qubit in 0..n_qubits {
        let pauli_type = (error_pattern >> (qubit * 2)) & 0x3;
        match pauli_type {
            0 => {}                                       // I
            1 => apply_pauli_x(&mut psi, qubit, n_qubits), // X
            2 => apply_pauli_y(&mut psi, qubit, n_qubits), // Y
            3 => apply_pauli_z(&mut psi, qubit, n_qubits), // Z
            _ => unreachable!(),
        }
    }
}

fn apply_cnot_gate_error(
    mut psi: ArrayViewMut2<f64>,
    n_qubits: usize,
    control: usize,
    target: usize,
    error_prob: f64,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    if rng.r#gen::<f64>() < error_prob {
        // Apply correlated error: flip both control and target
        apply_pauli_x(&mut psi, control, n_qubits);
        apply_pauli_x(&mut psi, target, n_qubits);
    }
}

#[pymodule]
fn rust_kernels(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(square_u32, m)?)?;
    m.add_function(wrap_pyfunction!(apply_pauli_channel_statevector, m)?)?;
    m.add_function(wrap_pyfunction!(apply_kraus_1q_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(execute_quantum_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(apply_correlated_pauli_noise_statevector, m)?)?;
    m.add_function(wrap_pyfunction!(apply_cnot_error_statevector, m)?)?;
    m.add_function(wrap_pyfunction!(expectation_value_pauli_string_py, m)?)?;
    
    // Add a class for Production readiness
    m.add_class::<QuantumCircuit>()?;
    
    Ok(())
}
