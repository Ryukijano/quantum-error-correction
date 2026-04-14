use gqe::{GQE, GQEConfig, Hamiltonian};
use std::sync::Arc;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Generative Quantum Eigensolver (GQE) Demo");
    println!("Finding ground state energy of H2 molecule using evolutionary optimization\n");

    // Create H2 Hamiltonian
    let hamiltonian = Arc::new(Hamiltonian::h2_minimal());
    println!("H2 Hamiltonian terms:");
    for (coeff, pauli) in &hamiltonian.terms {
        println!("  {:.6} * {}", coeff, pauli);
    }
    println!();

    // Configure GQE
    let config = GQEConfig {
        n_qubits: 2,
        // Need enough gates to represent an entangling + 2-layer rotation ansatz (VQE uses 6 gates).
        max_depth: 8,
        population_size: 50,
        n_generations: 200,
        mutation_rate: 0.2,
    };

    // Run GQE
    let mut gqe = GQE::new(hamiltonian, config.clone());
    let result = gqe.run(&config)?;

    println!("\n🎯 Results:");
    println!("Ground state energy: {:.8} Hartree", result.ground_state_energy);
    println!("Optimal circuit depth: {}", result.optimal_circuit.depth());
    println!("Optimal circuit gates: {}", result.optimal_circuit.gates.len());

    // Save history to JSON
    let history_json = serde_json::to_string(&result.history)?;
    let mut file = File::create("gqe_history.json")?;
    file.write_all(history_json.as_bytes())?;
    println!("📈 History saved to gqe_history.json");

    // Reference energy for this 2-qubit "H2 minimal" Hamiltonian (exact diagonalization).
    let known_ground_state = -1.8572750302023795_f64;
    let error = (result.ground_state_energy - known_ground_state).abs();
    println!("\n📊 Comparison with known value:");
    println!("Known H2 ground state: {:.6} Hartree", known_ground_state);
    println!("Absolute error: {:.6} Hartree", error);

    Ok(())
}
