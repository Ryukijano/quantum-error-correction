import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import qhybrid_kernels
import os

def numpy_statevector_simulator(qc: QuantumCircuit):
    """A very simple pure NumPy statevector simulator for comparison."""
    n_qubits = qc.num_qubits
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    
    # Transpile to basic gates
    qc_decomposed = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
    
    for instruction in qc_decomposed.data:
        gate = instruction.operation
        qubits = [qc_decomposed.find_bit(q).index for q in instruction.qubits]
        
        if gate.name == 'u':
            theta, phi, lam = gate.params
            u_mat = np.array([
                [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
            ])
            state = apply_gate(state, u_mat, qubits[0], n_qubits)
        elif gate.name == 'cx':
            cx_mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])
            state = apply_2q_gate(state, cx_mat, qubits[0], qubits[1], n_qubits)
            
    return state

def apply_gate(state, gate_mat, qubit, n_qubits):
    state = state.reshape([2]*(n_qubits))
    target_axis = n_qubits - 1 - qubit
    state = np.tensordot(gate_mat, state, axes=([1], [target_axis]))
    state = np.moveaxis(state, 0, target_axis)
    return state.flatten()

def apply_2q_gate(state, gate_mat, q_ctrl, q_trgt, n_qubits):
    state = state.reshape([2]*(n_qubits))
    ctrl_axis = n_qubits - 1 - q_ctrl
    trgt_axis = n_qubits - 1 - q_trgt
    gate_mat = gate_mat.reshape((2, 2, 2, 2))
    state = np.tensordot(gate_mat, state, axes=((2, 3), (ctrl_axis, trgt_axis)))
    state = np.moveaxis(state, [0, 1], [ctrl_axis, trgt_axis])
    return state.flatten()

def run_benchmark():
    qubit_range = range(4, 27) # Stress test up to 26 qubits
    results = []
    
    print(f"{'Qubits':<10} | {'Rust (ms)':<12} | {'NumPy (ms)':<12} | {'Aer-CPU (ms)':<12} | {'Aer-GPU (ms)':<12}")
    print("-" * 75)
    
    numpy_too_slow = False
    
    # Backends
    backend_cpu = Aer.get_backend('statevector_simulator')
    
    backend_gpu = Aer.get_backend('statevector_simulator')
    backend_gpu.set_options(device='GPU')
    
    # We'll try to enable cuStateVec if possible
    backend_custatevec = Aer.get_backend('statevector_simulator')
    try:
        backend_custatevec.set_options(device='GPU', cuStateVec_enable=True)
        custatevec_works = True
    except Exception:
        custatevec_works = False

    for n_qubits in qubit_range:
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits): qc.h(i)
        for i in range(n_qubits - 1): qc.cx(i, i+1)
            
        # 1. Rust Benchmark
        circuit_json = qhybrid_kernels.qiskit_to_qhybrid_json(qc)
        start = time.perf_counter()
        qhybrid_kernels.execute_quantum_circuit(circuit_json)
        rust_time = (time.perf_counter() - start) * 1000
        
        # 2. NumPy Benchmark
        numpy_time = None
        if not numpy_too_slow:
            try:
                start = time.perf_counter()
                numpy_statevector_simulator(qc)
                numpy_time = (time.perf_counter() - start) * 1000
                if numpy_time > 15000:
                    numpy_too_slow = True
            except Exception:
                numpy_too_slow = True
        
        # 3. Aer CPU Benchmark
        start = time.perf_counter()
        backend_cpu.run(transpile(qc, backend_cpu)).result()
        aer_cpu_time = (time.perf_counter() - start) * 1000
        
        # 4. Aer GPU Benchmark
        start = time.perf_counter()
        backend_gpu.run(transpile(qc, backend_gpu)).result()
        aer_gpu_time = (time.perf_counter() - start) * 1000
        
        # 5. Aer cuStateVec Benchmark (Optional)
        aer_cusv_time = None
        if custatevec_works:
            start = time.perf_counter()
            backend_custatevec.run(transpile(qc, backend_custatevec)).result()
            aer_cusv_time = (time.perf_counter() - start) * 1000

        numpy_str = f"{numpy_time:.2f}" if numpy_time else "SKIP"
        print(f"{n_qubits:<10} | {rust_time:<12.2f} | {numpy_str:<12} | {aer_cpu_time:<12.2f} | {aer_gpu_time:<12.2f}")
        
        results.append({
            'qubits': n_qubits,
            'rust': rust_time,
            'numpy': numpy_time,
            'aer_cpu': aer_cpu_time,
            'aer_gpu': aer_gpu_time,
            'aer_cusv': aer_cusv_time
        })
        
    df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Filter valid results for plotting
    valid_numpy = df.dropna(subset=['numpy'])
    plt.plot(valid_numpy['qubits'], valid_numpy['numpy'], label='NumPy (Naive)', marker='o', linestyle='--')
    
    plt.plot(df['qubits'], df['aer_cpu'], label='Qiskit Aer (CPU)', marker='s', linestyle='-')
    plt.plot(df['qubits'], df['aer_gpu'], label='Qiskit Aer (GPU)', marker='d', linestyle='-')
    
    if custatevec_works:
        valid_cusv = df.dropna(subset=['aer_cusv'])
        plt.plot(valid_cusv['qubits'], valid_cusv['aer_cusv'], label='Qiskit Aer (cuStateVec)', marker='x', linestyle='-.')
        
    plt.plot(df['qubits'], df['rust'], label='qhybrid (Rust - CPU)', marker='^', linestyle='-', linewidth=2, color='green')
    
    plt.yscale('log')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Execution Time (ms) - Log Scale')
    plt.title('HPC Quantum Simulator Comparison (CPU vs GPU vs Rust)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    os.makedirs('docs/assets', exist_ok=True)
    plt.savefig('docs/assets/speedup_hpc.png', dpi=300)
    print(f"\nHPC benchmark chart saved to docs/assets/speedup_hpc.png")

if __name__ == "__main__":
    run_benchmark()
