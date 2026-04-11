use rand::prelude::*;
use rand_pcg::Pcg64;
use rust_kernels::circuit::{Gate, GateType, QuantumCircuit};
use rust_kernels::execution::{ExecutionBackend, RustExecutionBackend};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Simple Hamiltonian representation for GQE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hamiltonian {
    /// Number of qubits
    pub n_qubits: usize,
    /// Pauli terms: (coefficient, pauli_string)
    pub terms: Vec<(f64, String)>,
}

impl Hamiltonian {
    /// Create H2 minimal Hamiltonian (2-qubit approximation)
    pub fn h2_minimal() -> Self {
        Self {
            n_qubits: 2,
            terms: vec![
                (-1.052373245772859, "II".to_string()),
                (0.39793742484318045, "IZ".to_string()),
                (-0.39793742484318045, "ZI".to_string()),
                (-0.01128010425623538, "ZZ".to_string()),
                (0.18093119978423156, "XX".to_string()),
            ],
        }
    }
}

/// Generative Quantum Eigensolver (GQE)
/// Uses a generative model to propose quantum circuits and optimizes them
/// to find ground state energies of quantum systems.
pub struct GQE {
    /// Number of qubits
    n_qubits: usize,
    /// Maximum circuit depth
    max_depth: usize,
    /// Hamiltonian to evaluate against
    hamiltonian: Arc<Hamiltonian>,
    /// Random number generator
    rng: Pcg64,
    /// Current best energy found
    best_energy: f64,
    executor: Arc<dyn ExecutionBackend>,
    /// Current best circuit
    best_circuit: Option<rust_kernels::circuit::QuantumCircuit>,
}

/// Configuration for GQE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GQEConfig {
    pub n_qubits: usize,
    pub max_depth: usize,
    pub population_size: usize,
    pub n_generations: usize,
    pub mutation_rate: f64,
}

impl Default for GQEConfig {
    fn default() -> Self {
        Self {
            n_qubits: 2,
            max_depth: 5,
            population_size: 50,
            n_generations: 100,
            mutation_rate: 0.1,
        }
    }
}

impl GQE {
    /// Create a new GQE instance
    pub fn new(hamiltonian: Arc<Hamiltonian>, config: GQEConfig) -> Self {
        let seed = 42;
        let rng = Pcg64::seed_from_u64(seed);
        let executor: Arc<dyn ExecutionBackend> = Arc::new(RustExecutionBackend::default());

        Self {
            n_qubits: config.n_qubits,
            max_depth: config.max_depth,
            hamiltonian,
            rng,
            best_energy: f64::INFINITY,
            executor,
            best_circuit: None,
        }
    }

    /// Run the GQE algorithm
    pub fn run(&mut self, config: &GQEConfig) -> Result<GQEResult, GQError> {
        println!("🚀 Starting GQE optimization...");
        println!("Population size: {}, Generations: {}", config.population_size, config.n_generations);

        // Initialize population
        let mut population = self.initialize_population(config.population_size);
        let mut history = Vec::with_capacity(config.n_generations);

        for generation in 0..config.n_generations {
            // Evaluate fitness of current population (and locally tune rotation angles)
            let fitness_scores = self.evaluate_population(&mut population)?;

            // Update best solution
            for (circuit, &energy) in population.iter().zip(&fitness_scores) {
                if energy < self.best_energy {
                    self.best_energy = energy;
                    self.best_circuit = Some(circuit.clone());
                }
            }

            println!("Generation {}: Best energy = {:.8} Hartree", generation, self.best_energy);
            history.push(self.best_energy);

            // Create next generation
            population = self.create_next_generation(&population, &fitness_scores, config);

            // Apply mutations
            self.mutate_population(&mut population, config.mutation_rate);
        }

        let result = GQEResult {
            ground_state_energy: self.best_energy,
            optimal_circuit: self.best_circuit.clone().unwrap(),
            config: config.clone(),
            history,
        };

        println!("✅ GQE completed! Ground state energy: {:.8} Hartree", result.ground_state_energy);

        Ok(result)
    }

    /// Initialize random population of circuits
    fn initialize_population(&mut self, size: usize) -> Vec<QuantumCircuit> {
        (0..size)
            .map(|_| self.generate_random_circuit())
            .collect()
    }

    /// Generate a random quantum circuit
    fn generate_random_circuit(&mut self) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(self.n_qubits);

        // Random number of gates (1 to max_depth)
        let n_gates = self.rng.gen_range(1..=self.max_depth);

        for _ in 0..n_gates {
            let gate = self.generate_random_gate();
            circuit.add_gate(gate).unwrap(); // Safe since we generate valid gates
        }

        circuit
    }

    /// Generate a random gate
    fn generate_random_gate(&mut self) -> Gate {
        let gate_types = [
            GateType::H,
            GateType::X,
            GateType::Y,
            GateType::Z,
            GateType::S,
            GateType::T,
            GateType::RX(0.0),
            GateType::RY(0.0),
            GateType::RZ(0.0),
            GateType::CX,
        ];

        let gate_type_idx = self.rng.gen_range(0..gate_types.len());
        let mut gate_type = gate_types[gate_type_idx].clone();

        // Add parameters for rotation gates
        match gate_type {
            GateType::RX(_) => gate_type = GateType::RX(self.rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI)),
            GateType::RY(_) => gate_type = GateType::RY(self.rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI)),
            GateType::RZ(_) => gate_type = GateType::RZ(self.rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI)),
            _ => {}
        }

        // Generate qubits based on gate type
        let qubits = match gate_type {
            GateType::CX | GateType::CY | GateType::CZ => {
                let control = self.rng.gen_range(0..self.n_qubits);
                let mut target = self.rng.gen_range(0..self.n_qubits);
                while target == control {
                    target = self.rng.gen_range(0..self.n_qubits);
                }
                vec![control, target]
            }
            _ => vec![self.rng.gen_range(0..self.n_qubits)],
        };

        let parameters = match gate_type {
            GateType::RX(theta) | GateType::RY(theta) | GateType::RZ(theta) => vec![theta],
            _ => vec![],
        };

        Gate {
            gate_type,
            qubits,
            parameters,
        }
    }

    /// Compute energy for a given circuit (lower is better).
    fn energy_of(&self, circuit: &QuantumCircuit) -> Result<f64, GQError> {
        let state = self.executor
            .execute_circuit(circuit, None)
            .map_err(|e| GQError::EvaluationError(e.to_string()))?;

        let mut total_energy = 0.0;
        for (coeff, pauli_string) in &self.hamiltonian.terms {
            let exp = self
                .executor
                .expectation_value_pauli_string(&state, pauli_string)
                .map_err(|e| GQError::EvaluationError(e.to_string()))?;
            total_energy += coeff * exp;
        }
        Ok(total_energy)
    }

    /// Local hill-climb on RX/RY/RZ angles for a circuit (keeps only improvements).
    fn tune_rotation_angles(&mut self, circuit: &mut QuantumCircuit, steps: usize, step_size: f64) -> Result<f64, GQError> {
        let mut best_energy = self.energy_of(circuit)?;

        // Precompute indices of rotation gates; if none, nothing to tune.
        let rot_indices: Vec<usize> = circuit.gates.iter().enumerate()
            .filter_map(|(i, g)| match g.gate_type {
                GateType::RX(_) | GateType::RY(_) | GateType::RZ(_) => Some(i),
                _ => None,
            })
            .collect();
        if rot_indices.is_empty() {
            return Ok(best_energy);
        }

        for _ in 0..steps {
            let idx = rot_indices[self.rng.gen_range(0..rot_indices.len())];
            let old_gate = circuit.gates[idx].clone();

            let delta = self.rng.gen_range(-step_size..step_size);
            match old_gate.gate_type {
                GateType::RX(theta) => {
                    let new_theta = (theta + delta).clamp(-std::f64::consts::PI, std::f64::consts::PI);
                    circuit.gates[idx].gate_type = GateType::RX(new_theta);
                    circuit.gates[idx].parameters = vec![new_theta];
                }
                GateType::RY(theta) => {
                    let new_theta = (theta + delta).clamp(-std::f64::consts::PI, std::f64::consts::PI);
                    circuit.gates[idx].gate_type = GateType::RY(new_theta);
                    circuit.gates[idx].parameters = vec![new_theta];
                }
                GateType::RZ(theta) => {
                    let new_theta = (theta + delta).clamp(-std::f64::consts::PI, std::f64::consts::PI);
                    circuit.gates[idx].gate_type = GateType::RZ(new_theta);
                    circuit.gates[idx].parameters = vec![new_theta];
                }
                _ => {}
            }

            let new_energy = self.energy_of(circuit)?;
            if new_energy < best_energy {
                best_energy = new_energy;
            } else {
                // revert
                circuit.gates[idx] = old_gate;
            }
        }

        Ok(best_energy)
    }

    /// Evaluate fitness of population (lower energy is better).
    /// Also performs a small local search over rotation angles to turn \"good structure\" into \"good energy\".
    fn evaluate_population(&mut self, population: &mut [QuantumCircuit]) -> Result<Vec<f64>, GQError> {
        let tune_steps = 20;
        let tune_step_size = 0.6;

        population
            .iter_mut()
            .map(|circuit| {
                // Light local tuning: only affects circuits with rotation gates.
                self.tune_rotation_angles(circuit, tune_steps, tune_step_size)
            })
            .collect()
    }

    /// Create next generation using tournament selection
    fn create_next_generation(
        &mut self,
        population: &[QuantumCircuit],
        fitness_scores: &[f64],
        config: &GQEConfig,
    ) -> Vec<QuantumCircuit> {
        let mut next_population = Vec::with_capacity(config.population_size);

        // --- Elitism: keep the best circuits unchanged to prevent regression.
        let mut ranked: Vec<usize> = (0..population.len()).collect();
        ranked.sort_by(|&a, &b| fitness_scores[a].partial_cmp(&fitness_scores[b]).unwrap());
        let elite_count = (config.population_size / 10).max(1);
        for &idx in ranked.iter().take(elite_count) {
            next_population.push(population[idx].clone());
        }

        // --- Diversity: add some random immigrants to avoid premature convergence.
        let immigrant_rate = 0.10;

        while next_population.len() < config.population_size {
            if self.rng.gen::<f64>() < immigrant_rate {
                next_population.push(self.generate_random_circuit());
                continue;
            }

            // Tournament selection: pick best of random subset
            let candidates: Vec<usize> = (0..3)
                .map(|_| self.rng.gen_range(0..population.len()))
                .collect();

            let winner = candidates
                .iter()
                .min_by(|&&a, &&b| fitness_scores[a].partial_cmp(&fitness_scores[b]).unwrap())
                .unwrap();

            next_population.push(population[*winner].clone());
        }

        next_population
    }

    /// Apply mutations to population
    fn mutate_population(&mut self, population: &mut [QuantumCircuit], mutation_rate: f64) {
        for circuit in population.iter_mut() {
            if self.rng.gen::<f64>() < mutation_rate {
                self.mutate_circuit(circuit);
            }
        }
    }

    /// Mutate a single circuit
    fn mutate_circuit(&mut self, circuit: &mut QuantumCircuit) {
        // Mutations:
        // - structural: add/remove/replace a gate
        // - parametric: small perturbations to RX/RY/RZ angles (key for convergence)
        match self.rng.gen_range(0..4) {
            0 => {
                // Add a random gate
                if circuit.gates.len() < self.max_depth {
                    let gate = self.generate_random_gate();
                    circuit.add_gate(gate).unwrap();
                }
            }
            1 => {
                // Remove a random gate
                if !circuit.gates.is_empty() {
                    let idx = self.rng.gen_range(0..circuit.gates.len());
                    circuit.gates.remove(idx);
                }
            }
            2 => {
                // Modify a random gate
                if !circuit.gates.is_empty() {
                    let idx = self.rng.gen_range(0..circuit.gates.len());
                    // If it's a rotation gate, prefer a small angle tweak over a full replace.
                    let step = 0.25_f64; // radians
                    match circuit.gates[idx].gate_type {
                        GateType::RX(theta) => {
                            let new_theta = (theta + self.rng.gen_range(-step..step))
                                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                            circuit.gates[idx].gate_type = GateType::RX(new_theta);
                            circuit.gates[idx].parameters = vec![new_theta];
                        }
                        GateType::RY(theta) => {
                            let new_theta = (theta + self.rng.gen_range(-step..step))
                                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                            circuit.gates[idx].gate_type = GateType::RY(new_theta);
                            circuit.gates[idx].parameters = vec![new_theta];
                        }
                        GateType::RZ(theta) => {
                            let new_theta = (theta + self.rng.gen_range(-step..step))
                                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                            circuit.gates[idx].gate_type = GateType::RZ(new_theta);
                            circuit.gates[idx].parameters = vec![new_theta];
                        }
                        _ => {
                            circuit.gates[idx] = self.generate_random_gate();
                        }
                    }
                }
            }
            3 => {
                // Angle-only mutation: pick a rotation gate and nudge its parameter.
                if !circuit.gates.is_empty() {
                    let idx = self.rng.gen_range(0..circuit.gates.len());
                    let step = 0.50_f64; // radians
                    match circuit.gates[idx].gate_type {
                        GateType::RX(theta) => {
                            let new_theta = (theta + self.rng.gen_range(-step..step))
                                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                            circuit.gates[idx].gate_type = GateType::RX(new_theta);
                            circuit.gates[idx].parameters = vec![new_theta];
                        }
                        GateType::RY(theta) => {
                            let new_theta = (theta + self.rng.gen_range(-step..step))
                                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                            circuit.gates[idx].gate_type = GateType::RY(new_theta);
                            circuit.gates[idx].parameters = vec![new_theta];
                        }
                        GateType::RZ(theta) => {
                            let new_theta = (theta + self.rng.gen_range(-step..step))
                                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                            circuit.gates[idx].gate_type = GateType::RZ(new_theta);
                            circuit.gates[idx].parameters = vec![new_theta];
                        }
                        _ => {}
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Result of GQE optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GQEResult {
    pub ground_state_energy: f64,
    pub optimal_circuit: QuantumCircuit,
    pub config: GQEConfig,
    pub history: Vec<f64>,
}

/// Errors that can occur during GQE execution
#[derive(Debug, thiserror::Error)]
pub enum GQError {
    #[error("Circuit evaluation error: {0}")]
    EvaluationError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqe_basic() {
        // Create a simple 2-qubit Hamiltonian (H2 molecule approximation)
        let hamiltonian = Arc::new(Hamiltonian::h2_minimal());

        let config = GQEConfig {
            n_qubits: 2,
            max_depth: 3,
            population_size: 10,
            n_generations: 5,
            mutation_rate: 0.1,
        };

        let mut gqe = GQE::new(hamiltonian, config.clone());

        let result = gqe.run(&config).unwrap();

        // Check that we got some result
        assert!(result.ground_state_energy.is_finite());
        assert_eq!(result.optimal_circuit.n_qubits, 2);
        assert!(result.optimal_circuit.gates.len() <= config.max_depth);
    }

    #[test]
    fn test_random_circuit_generation() {
        let hamiltonian = Arc::new(Hamiltonian::h2_minimal());
        let config = GQEConfig::default();
        let mut gqe = GQE::new(hamiltonian, config);

        let circuit = gqe.generate_random_circuit();

        assert_eq!(circuit.n_qubits, 2);
        assert!(circuit.gates.len() >= 1);
        assert!(circuit.gates.len() <= 5);
    }
}