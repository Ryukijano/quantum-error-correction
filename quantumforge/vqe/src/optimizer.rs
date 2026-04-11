use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::neldermead::NelderMead;
use ndarray::Array2;
use ndarray::Array1;
use rust_kernels::execution::{ExecutionBackend, RustExecutionBackend};
use std::sync::Arc;
use std::cell::RefCell;

/// Combined VQE problem that implements CostFunction
pub struct VQEProblem {
    pub hamiltonian: Arc<super::hamiltonian::Hamiltonian>,
    pub ansatz: Arc<super::ansatz::Ansatz>,
    pub executor: Arc<dyn ExecutionBackend>,
    pub energy_history: RefCell<Vec<f64>>,
}

impl CostFunction for VQEProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<Self::Output, Error> {
        // Generate circuit from parameters
        let circuit = self.ansatz.circuit(parameters.as_slice().unwrap());

        // Execute circuit to get final statevector
        let state = self.executor
            .execute_circuit(&circuit, None)
            .map_err(|e| Error::msg(format!("Circuit execution failed: {e}")))?;

        // Compute expectation value ⟨ψ|H|ψ⟩ using the kernels' Pauli measurement routine
        let energy = self.hamiltonian
            .expectation_value(&state, self.executor.as_ref())
            .map_err(|e| Error::msg(format!("Expectation computation failed: {e}")))?;

        // Track energy
        self.energy_history.borrow_mut().push(energy);

        Ok(energy)
    }
}

/// Run VQE optimization
pub fn run_vqe(
    hamiltonian: Arc<super::hamiltonian::Hamiltonian>,
    ansatz: Arc<super::ansatz::Ansatz>,
    initial_parameters: Vec<f64>,
    max_iterations: usize,
) -> Result<(Vec<f64>, f64, Vec<f64>), Box<dyn std::error::Error>> {
    println!("🚀 Starting VQE optimization...");
    // Initial state is |0...0⟩ in statevector form (2^n, 2) with [re, im].
    let n_qubits = hamiltonian.n_qubits;
    let dim = 1usize << n_qubits;
    let mut initial_state = Array2::<f64>::zeros((dim, 2));
    initial_state[[0, 0]] = 1.0;
    println!(
        "Initial energy: {:.8}",
        hamiltonian.expectation_value(&initial_state, &RustExecutionBackend)?,
    );

    // Create problem
    let executor: Arc<dyn ExecutionBackend> = Arc::new(RustExecutionBackend::default());
    let problem = VQEProblem {
        hamiltonian: hamiltonian.clone(),
        ansatz,
        executor,
        energy_history: RefCell::new(Vec::new()),
    };

    // Nelder-Mead optimizer (derivative-free)
    let mut simplex = vec![];
    let initial_params_array = Array1::from(initial_parameters.clone());
    
    for i in 0..initial_parameters.len() + 1 {
        let mut p = initial_params_array.clone();
        if i > 0 {
            p[i - 1] += 0.1;
        }
        simplex.push(p);
    }

    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-6)?;

    // Run optimization
    let res = Executor::new(problem, solver)
        .configure(|state| state.max_iters(max_iterations as u64))
        .run()?;

    let best_params = res.state().best_param.clone().unwrap().to_vec();
    let final_energy = res.state().best_cost;
    let history = res.problem().problem.as_ref().unwrap().energy_history.borrow().clone();

    println!("✅ VQE converged after {} iterations", res.state().iter);
    println!("Final energy: {:.8} Hartree", final_energy);

    Ok((best_params, final_energy, history))
}
