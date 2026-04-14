"""
GQE Multi-Backend Comparison for H2 Molecule
Compares Rust (qhybrid) vs Python implementations using different simulators
"""
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# H2 minimal Hamiltonian (2-qubit)
H2_TERMS = [
    (-1.052373245772859, "II"),
    (0.39793742484318045, "IZ"),
    (-0.39793742484318045, "ZI"),
    (-0.01128010425623538, "ZZ"),
    (0.18093119978423156, "XX"),
]

EXACT_GROUND = -1.8572750302023795

def pauli_expectation_qiskit(circuit, pauli_string, backend):
    """Compute Pauli expectation using Qiskit backend."""
    from qiskit.quantum_info import SparsePauliOp
    
    sv = Statevector.from_instruction(circuit)
    
    # Create Pauli operator
    op = SparsePauliOp.from_list([(pauli_string, 1.0)])
    expectation = sv.expectation_value(op).real
    
    return expectation

def evaluate_hamiltonian(circuit, backend_type='statevector'):
    """Evaluate H2 Hamiltonian for a circuit."""
    total = 0.0
    
    if backend_type == 'statevector':
        sv = Statevector.from_instruction(circuit)
        for coeff, pauli in H2_TERMS:
            from qiskit.quantum_info import SparsePauliOp
            op = SparsePauliOp.from_list([(pauli, 1.0)])
            exp_val = sv.expectation_value(op).real
            total += coeff * exp_val
    
    return total

def generate_random_circuit(n_qubits, max_depth):
    """Generate a random quantum circuit."""
    qc = QuantumCircuit(n_qubits)
    n_gates = random.randint(1, max_depth)
    
    gate_types = ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cx']
    
    for _ in range(n_gates):
        gate = random.choice(gate_types)
        
        if gate == 'cx':
            ctrl = random.randint(0, n_qubits-1)
            targ = random.randint(0, n_qubits-1)
            while targ == ctrl:
                targ = random.randint(0, n_qubits-1)
            qc.cx(ctrl, targ)
        elif gate in ['rx', 'ry', 'rz']:
            qubit = random.randint(0, n_qubits-1)
            angle = random.uniform(-np.pi, np.pi)
            getattr(qc, gate)(angle, qubit)
        else:
            qubit = random.randint(0, n_qubits-1)
            getattr(qc, gate)(qubit)
    
    return qc

def mutate_circuit(qc, mutation_type='add'):
    """Mutate a circuit."""
    new_qc = qc.copy()
    
    if mutation_type == 'add' and len(new_qc.data) < 10:
        gate_types = ['h', 'x', 'y', 'z', 's']
        gate = random.choice(gate_types)
        qubit = random.randint(0, new_qc.num_qubits-1)
        getattr(new_qc, gate)(qubit)
    elif mutation_type == 'remove' and len(new_qc.data) > 0:
        idx = random.randint(0, len(new_qc.data)-1)
        new_qc.data.pop(idx)
    
    return new_qc

def python_gqe(backend_name='statevector', use_gpu=False):
    """Python GQE implementation."""
    n_qubits = 2
    population_size = 50
    n_generations = 200
    mutation_rate = 0.5
    
    history = []
    best_energy = float('inf')
    best_circuit = None
    
    # Initialize population
    population = [generate_random_circuit(n_qubits, 10) for _ in range(population_size)]
    
    for gen in range(n_generations):
        # Evaluate fitness
        fitness = []
        for circ in population:
            try:
                energy = evaluate_hamiltonian(circ, backend_type='statevector')
                fitness.append(energy)
            except:
                fitness.append(float('inf'))
        
        # Update best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_energy:
            best_energy = fitness[min_idx]
            best_circuit = population[min_idx].copy()
        
        history.append(best_energy)
        
        # Selection (tournament)
        new_population = []
        for _ in range(population_size):
            tournament = random.sample(list(zip(population, fitness)), 3)
            winner = min(tournament, key=lambda x: x[1])[0]
            new_population.append(winner.copy())
        
        # Mutation
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                mut_type = random.choice(['add', 'remove'])
                new_population[i] = mutate_circuit(new_population[i], mut_type)
        
        # Add elitism
        new_population[0] = best_circuit.copy()
        
        population = new_population
    
    return {
        'backend': f'Python-{backend_name}',
        'energy': best_energy,
        'history': history,
        'generations': n_generations
    }

def rust_gqe():
    """Load Rust GQE results."""
    with open('gqe_history.json', 'r') as f:
        history = json.load(f)
    
    return {
        'backend': 'qhybrid (Rust)',
        'energy': min(history),
        'history': history,
        'generations': len(history)
    }

def run_gqe_comparison():
    print("Running GQE on multiple backends...")
    print("="*60)
    
    results = []
    
    # 1. Rust
    print("1. Loading Rust GQE results...")
    results.append(rust_gqe())
    
    # 2. Python/NumPy
    print("2. Running Python/NumPy GQE...")
    start = time.perf_counter()
    py_result = python_gqe('statevector')
    py_result['time'] = time.perf_counter() - start
    results.append(py_result)
    
    # Print summary
    print("\n" + "="*60)
    print(f"{'Backend':<25} | {'Energy':<15} | {'Error':<12} | {'Time (s)':<10}")
    print("-"*60)
    
    for r in results:
        err = abs(r['energy'] - EXACT_GROUND)
        time_str = f"{r.get('time', 0):.3f}" if 'time' in r else "N/A"
        print(f"{r['backend']:<25} | {r['energy']:<15.8f} | {err:<12.2e} | {time_str:<10}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    for r in results:
        if 'history' in r:
            plt.plot(r['history'], label=r['backend'], marker='.', markersize=3, alpha=0.7)
    
    plt.axhline(y=EXACT_GROUND, color='red', linestyle='--', linewidth=2, label='Exact Ground State')
    
    plt.xlabel('Generation')
    plt.ylabel('Energy (Hartree)')
    plt.title('GQE Convergence Comparison Across Backends (H2 Molecule)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    import os
    os.makedirs('docs/assets', exist_ok=True)
    plt.savefig('docs/assets/gqe_backends_comparison.png', dpi=300)
    print("\nGQE backend comparison saved to docs/assets/gqe_backends_comparison.png")
    
    # Save results
    with open('gqe_backends_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to gqe_backends_results.json")

if __name__ == "__main__":
    run_gqe_comparison()
