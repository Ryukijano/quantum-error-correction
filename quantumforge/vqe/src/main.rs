mod hamiltonian;
mod ansatz;
mod optimizer;

use std::sync::Arc;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Rust VQE for H2 Molecule");
    println!("=====================================");

    // Create H2 Hamiltonian (minimal 2-qubit version for faster comparison)
    let hamiltonian = Arc::new(hamiltonian::Hamiltonian::h2_minimal());
    println!("📊 Hamiltonian loaded: {} qubits, {} terms",
             hamiltonian.n_qubits, hamiltonian.terms.len());

    // Create UCC ansatz (for 2 qubits, let's use a simpler ansatz or just keep ucc_h2 with 2 qubits)
    // Actually, let's modify Ansatz to support 2-qubit UCC.
    let n_qubits = 2;
    // For the built-in hardware-efficient template, we need 2 layers of RY rotations:
    // first layer: n_qubits params, second layer: n_qubits params => 2*n_qubits total.
    // With only 2 params, the second layer is effectively disabled and the ansatz is under-expressive.
    let n_parameters = 2 * n_qubits;
    let ansatz = Arc::new(ansatz::Ansatz {
        ansatz_type: ansatz::AnsatzType::HardwareEfficient,
        n_qubits,
        n_parameters,
    });
    println!("🧬 Ansatz: Hardware-Efficient with {} parameters", ansatz.n_parameters);

    // Initial parameters (random)
    let mut initial_params = vec![0.0; ansatz.n_parameters];
    for i in 0..initial_params.len() {
        initial_params[i] = 0.1 * (i as f64 - (ansatz.n_parameters as f64) / 2.0);
    }

    println!("🎯 Starting optimization with {} parameters", initial_params.len());
    println!("Initial parameters: {:?}", initial_params);

    // Run VQE
    let (optimal_params, ground_energy, history) = optimizer::run_vqe(
        hamiltonian,
        ansatz,
        initial_params,
        200, // max iterations (more room to converge)
    )?;

    println!("\n🎉 VQE Optimization Complete!");
    println!("=====================================");
    println!("Ground state energy: {:.8} Hartree", ground_energy);
    println!("Optimal parameters: {:?}", optimal_params);

    // Save history to JSON
    let history_json = serde_json::to_string(&history)?;
    let mut file = File::create("vqe_history.json")?;
    file.write_all(history_json.as_bytes())?;
    println!("📈 History saved to vqe_history.json");

    // Note on reference energies:
    // - -1.137 Hartree is the *4-qubit STO-3G* H2 ground state (different Hamiltonian).
    // - This demo uses the 2-qubit "minimal H2" toy Hamiltonian; its exact ground energy is ~ -1.85727503.
    println!("\n📝 Reference: exact ground energy for this 2-qubit H2-minimal Hamiltonian is ~ -1.85727503 Hartree");

    Ok(())
}
