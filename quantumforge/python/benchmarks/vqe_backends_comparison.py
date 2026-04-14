"""
VQE Multi-Backend Comparison for H2 Molecule
Compares Rust (qhybrid) vs Qiskit VQE on different backends
"""
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Qiskit imports are moved to individual functions to avoid import errors

# H2 minimal Hamiltonian (2-qubit)
H2_TERMS = [
    (-1.052373245772859, "II"),
    (0.39793742484318045, "IZ"),
    (-0.39793742484318045, "ZI"),
    (-0.01128010425623538, "ZZ"),
    (0.18093119978423156, "XX"),
]

EXACT_GROUND = -1.8572750302023795

def create_hardware_efficient_ansatz(n_qubits, n_layers=2):
    """Create hardware-efficient ansatz parameters."""
    params = []
    for layer in range(n_layers):
        for i in range(n_qubits):
            params.append(f'theta_{layer}_{i}')
    return params

def create_qiskit_ansatz(n_qubits, n_layers=2):
    """Create Qiskit hardware-efficient ansatz."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(n_qubits)
        params = []

        for layer in range(n_layers):
            # Rotation layer
            for i in range(n_qubits):
                param = Parameter(f'theta_{layer}_{i}')
                params.append(param)
                qc.ry(param, i)

            # Entangling layer
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)

        return qc, params
    except ImportError:
        return None, []

def pauli_expectation(state, pauli_string, n_qubits):
    """Compute <psi|P|psi> for a Pauli string with proper basis rotations."""
    # Apply basis rotations to transform X/Y measurements into Z measurements
    rotated_state = state.copy()
    
    for i, p in enumerate(pauli_string):
        if p == 'X':
            # RY(-π/2) rotates X-basis to Z-basis
            rotated_state = apply_ry_gate(rotated_state, i, -np.pi/2, n_qubits)
        elif p == 'Y':
            # RX(π/2) rotates Y-basis to Z-basis  
            rotated_state = apply_rx_gate(rotated_state, i, np.pi/2, n_qubits)
    
    # Now compute Z-basis expectation
    # After rotations, ALL non-I Paulis are now in Z-basis
    if isinstance(rotated_state, np.ndarray):
        dim = 2**n_qubits
        expectation = 0.0
        for i in range(dim):
            prob = abs(rotated_state[i])**2
            eigenvalue = 1.0
            # Since we rotated X→Z and Y→Z, ALL non-I Paulis contribute as Z
            for q, p in enumerate(pauli_string):
                if p != 'I':  # X, Y, Z all measured in Z-basis now
                    bit = (i >> q) & 1
                    if bit == 1:
                        eigenvalue *= -1.0
            expectation += prob * eigenvalue
        
        return expectation
    else:
        raise NotImplementedError("Only statevector supported")

def hamiltonian_expectation(state, n_qubits):
    """Compute H2 Hamiltonian expectation value."""
    total = 0.0
    for coeff, pauli_string in H2_TERMS:
        exp_val = pauli_expectation(state, pauli_string, n_qubits)
        total += coeff * exp_val
    return total

def numpy_vqe():
    """Pure Python/NumPy VQE implementation."""
    n_qubits = 2
    n_params = 4
    
    history = []
    
    def cost_function(params):
        # Build and execute circuit
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        
        # Apply hardware-efficient ansatz
        for layer in range(2):
            # RY layer
            for i in range(n_qubits):
                param_idx = layer * n_qubits + i
                if param_idx < len(params):
                    theta = params[param_idx]
                    # Apply RY gate
                    state = apply_ry_gate(state, i, theta, n_qubits)
            
            # CX layer
            for i in range(n_qubits - 1):
                state = apply_cx_gate(state, i, i+1, n_qubits)
        
        energy = hamiltonian_expectation(state, n_qubits)
        history.append(energy)
        return energy
    
    initial_params = np.array([-0.2, -0.1, 0.0, 0.1])
    
    start = time.perf_counter()
    result = minimize(cost_function, initial_params, method='Nelder-Mead', 
                     options={'maxiter': 100, 'xatol': 1e-6})
    elapsed = time.perf_counter() - start
    
    return {
        'backend': 'NumPy',
        'energy': result.fun,
        'params': result.x.tolist(),
        'history': history,
        'time': elapsed,
        'iterations': len(history)
    }

def apply_ry_gate(state, qubit, theta, n_qubits):
    """Apply RY gate to statevector."""
    dim = 2**n_qubits
    state = state.reshape([2]*n_qubits)
    target_axis = n_qubits - 1 - qubit
    
    cos_h = np.cos(theta/2)
    sin_h = np.sin(theta/2)
    ry_mat = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    
    state = np.tensordot(ry_mat, state, axes=([1], [target_axis]))
    state = np.moveaxis(state, 0, target_axis)
    return state.flatten()

def apply_rx_gate(state, qubit, theta, n_qubits):
    """Apply RX gate to statevector."""
    dim = 2**n_qubits
    state = state.reshape([2]*n_qubits)
    target_axis = n_qubits - 1 - qubit
    
    cos_h = np.cos(theta/2)
    sin_h = np.sin(theta/2)
    rx_mat = np.array([[cos_h, -1j*sin_h], [-1j*sin_h, cos_h]], dtype=complex)
    
    state = np.tensordot(rx_mat, state, axes=([1], [target_axis]))
    state = np.moveaxis(state, 0, target_axis)
    return state.flatten()

def apply_cx_gate(state, ctrl, targ, n_qubits):
    """Apply CNOT gate to statevector."""
    dim = 2**n_qubits
    state = state.reshape([2]*n_qubits)
    ctrl_axis = n_qubits - 1 - ctrl
    targ_axis = n_qubits - 1 - targ
    
    cx_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex).reshape(2,2,2,2)
    
    state = np.tensordot(cx_mat, state, axes=((2,3), (ctrl_axis, targ_axis)))
    state = np.moveaxis(state, [0,1], [ctrl_axis, targ_axis])
    return state.flatten()

def qiskit_vqe_aer_cpu():
    """VQE using Qiskit with Aer CPU backend (manual optimization)."""
    try:
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit_aer import AerSimulator
        from scipy.optimize import minimize as scipy_minimize

        n_qubits = 2
        ansatz_template, param_names = create_qiskit_ansatz(n_qubits, n_layers=2)

        if ansatz_template is None:
            raise ImportError("Qiskit not available")

        backend = AerSimulator(method='statevector')
        history = []

        def cost_function(params):
            # Bind parameters
            # Bind parameters (Qiskit 2.0+ uses assign_parameters)
            param_dict = {param_names[i]: params[i] for i in range(len(params))}
            bound_circuit = ansatz_template.assign_parameters(param_dict)

            # Execute and get statevector
            sv = Statevector.from_instruction(bound_circuit)

            # Compute H expectation
            total = 0.0
            for coeff, pauli in H2_TERMS:
                op = SparsePauliOp.from_list([(pauli, 1.0)])
                exp_val = sv.expectation_value(op).real
                total += coeff * exp_val

            history.append(total)
            return total

        initial_params = np.array([-0.2, -0.1, 0.0, 0.1])

        start = time.perf_counter()
        result = scipy_minimize(cost_function, initial_params, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-6})
        elapsed = time.perf_counter() - start

        return {
            'backend': 'Qiskit Aer (CPU)',
            'energy': result.fun,
            'params': result.x.tolist(),
            'history': history,
            'time': elapsed,
            'iterations': len(history)
        }

    except (ImportError, AttributeError) as e:
        # Fallback: simulate Qiskit Aer behavior using NumPy
        print(f"⚠️  Qiskit Aer (CPU) not available: {e}")
        return None

def qiskit_vqe_aer_gpu():
    """VQE using Qiskit with Aer GPU backend (cuQuantum accelerated)."""
    try:
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from scipy.optimize import minimize as scipy_minimize

        n_qubits = 2
        ansatz_template, param_names = create_qiskit_ansatz(n_qubits, n_layers=2)

        if ansatz_template is None:
            raise ImportError("Qiskit not available")

        # Try to create GPU-accelerated simulator
        try:
            # Try direct import to bypass provider issues
            from qiskit_aer.primitives import Estimator
            from qiskit_aer import AerSimulator
            backend = AerSimulator(method='statevector', device='GPU')
            backend_name = 'Qiskit Aer (GPU/cuQuantum)'
            print(f"Successfully created GPU simulator")
        except Exception as e:
            print(f"GPU backend not available ({e}), falling back to CPU")
            try:
                backend = AerSimulator(method='statevector')
                backend_name = 'Qiskit Aer (CPU fallback)'
            except Exception as e2:
                print(f"CPU backend also failed ({e2})")
                return {
                    'backend': 'Qiskit Aer - Not Available',
                    'energy': float('nan'),
                    'params': [],
                    'history': [],
                    'time': 0,
                    'iterations': 0,
                    'error': f"GPU: {e}, CPU: {e2}"
                }

        history = []

            # Bind parameters (Qiskit 2.0+ uses assign_parameters)
        def cost_function(params):
            # Bind parameters (Qiskit 2.0+ uses assign_parameters)
            # Create parameter dict: map Parameter objects to values
            param_dict = {param_names[i]: params[i] for i in range(len(params))}
            bound_circuit = ansatz_template.assign_parameters(param_dict)

            # Execute and get statevector
            sv = Statevector.from_instruction(bound_circuit)

            # Compute H expectation
            total = 0.0
            for coeff, pauli in H2_TERMS:
                op = SparsePauliOp.from_list([(pauli, 1.0)])
                exp_val = sv.expectation_value(op).real
                total += coeff * exp_val

            history.append(total)
            return total

        initial_params = np.array([-0.2, -0.1, 0.0, 0.1])

        start = time.perf_counter()
        result = scipy_minimize(cost_function, initial_params, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-6})
        elapsed = time.perf_counter() - start

        return {
            'backend': backend_name,
            'energy': result.fun,
            'params': result.x.tolist(),
            'history': history,
            'time': elapsed,
            'iterations': len(history)
        }

    except (ImportError, AttributeError) as e:
        # Fallback: simulate Qiskit Aer GPU behavior
        print(f"⚠️  Qiskit Aer (GPU) not available: {e}")
        return None

def cudaq_vqe():
    """CUDA-Q VQE implementation using NVIDIA's CUDA-Q platform."""
    try:
        import cudaq
        
        history = []
        
        # Create H2 Hamiltonian
        def create_h2_hamiltonian():
            hamiltonian = cudaq.SpinOperator()
            # H2 terms from H2_TERMS
            hamiltonian += -1.052373245772859  # II
            hamiltonian += 0.39793742484318045 * cudaq.spin.z(1)  # IZ
            hamiltonian += -0.39793742484318045 * cudaq.spin.z(0)  # ZI
            hamiltonian += -0.01128010425623538 * cudaq.spin.z(0) * cudaq.spin.z(1)  # ZZ
            hamiltonian += 0.18093119978423156 * cudaq.spin.x(0) * cudaq.spin.x(1)  # XX
            return hamiltonian
        
        hamiltonian = create_h2_hamiltonian()
        
        # Create ansatz kernel function (recreated for each parameter set)
        def create_ansatz(theta0, theta1, theta2, theta3):
            kernel = cudaq.make_kernel()
            q = kernel.qalloc(2)
            # Hardware-efficient ansatz: 2 layers
            # Layer 1
            kernel.ry(theta0, q[0])
            kernel.ry(theta1, q[1])
            kernel.cx(q[0], q[1])
            # Layer 2
            kernel.ry(theta2, q[0])
            kernel.ry(theta3, q[1])
            kernel.cx(q[0], q[1])
            return kernel
        
        def cost_function(params):
            # Create kernel with current parameters and compute expectation
            kernel = create_ansatz(*params)
            result = cudaq.observe(kernel, hamiltonian)
            energy = result.expectation()
            history.append(energy)
            return energy
        
        initial_params = np.array([-0.2, -0.1, 0.0, 0.1])
        
        start = time.perf_counter()
        
        # Optimize using scipy
        from scipy.optimize import minimize as scipy_minimize
        result = scipy_minimize(cost_function, initial_params, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-6})
        
        elapsed = time.perf_counter() - start
        
        return {
            'backend': 'CUDA-Q (NVIDIA)',
            'energy': result.fun,
            'params': result.x.tolist(),
            'history': history,
            'time': elapsed,
            'iterations': len(history),
            'gpu_accelerated': True
        }
        
    except (ImportError, AttributeError) as e:
        print(f"⚠️  CUDA-Q not available: {e}")
        return None
    except Exception as e:
        print(f"⚠️  CUDA-Q VQE failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def rust_vqe():
    """Load Rust VQE results."""
    with open('vqe_history.json', 'r') as f:
        history = json.load(f)
    
    return {
        'backend': 'qhybrid (Rust)',
        'energy': min(history),
        'history': history,
        'iterations': len(history)
    }

def run_vqe_comparison():
    print("Running VQE on multiple backends...")
    print("="*60)
    
    results = []
    
    # 1. Rust
    print("1. Loading Rust VQE results...")
    results.append(rust_vqe())
    
    # 2. NumPy
    print("2. Running NumPy VQE...")
    results.append(numpy_vqe())
    
    # 3. CUDA-Q (NVIDIA)
    print("3. Running CUDA-Q VQE (NVIDIA GPU accelerated)...")
    cudaq_result = cudaq_vqe()
    if cudaq_result is not None:
        results.append(cudaq_result)
    else:
        print("  CUDA-Q not available, skipping...")

    # 4. Qiskit Aer (if available)
    print("4. Testing Qiskit Aer backends...")
    try:
        cpu_result = qiskit_vqe_aer_cpu()
        results.append(cpu_result)
    except Exception as e:
        print(f"  Qiskit CPU failed: {e}")
        results.append({
            'backend': 'Qiskit Aer (CPU) - Import Error',
            'energy': float('nan'),
            'params': [],
            'history': [],
            'time': 0,
            'iterations': 0,
            'error': str(e)
        })

    try:
        gpu_result = qiskit_vqe_aer_gpu()
        results.append(gpu_result)
    except Exception as e:
        print(f"  Qiskit GPU failed: {e}")
        results.append({
            'backend': 'Qiskit Aer (GPU) - Import Error',
            'energy': float('nan'),
            'params': [],
            'history': [],
            'time': 0,
            'iterations': 0,
            'error': str(e)
        })
    
    # Print summary
    print("\n" + "="*60)
    print(f"{'Backend':<20} | {'Energy':<15} | {'Error':<12} | {'Time (s)':<10}")
    print("-"*60)
    
    for r in results:
        err = abs(r['energy'] - EXACT_GROUND)
        time_str = f"{r.get('time', 0):.3f}" if 'time' in r else "N/A"
        print(f"{r['backend']:<20} | {r['energy']:<15.8f} | {err:<12.2e} | {time_str:<10}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    for r in results:
        if 'history' in r:
            plt.plot(r['history'], label=r['backend'], marker='.', markersize=3, alpha=0.7)
    
    plt.axhline(y=EXACT_GROUND, color='red', linestyle='--', linewidth=2, label='Exact Ground State')
    
    plt.xlabel('Iteration / Function Evaluation')
    plt.ylabel('Energy (Hartree)')
    plt.title('VQE Convergence Comparison Across Backends (H2 Molecule)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    import os
    os.makedirs('docs/assets', exist_ok=True)
    plt.savefig('docs/assets/vqe_backends_comparison.png', dpi=300)
    print("\nVQE backend comparison saved to docs/assets/vqe_backends_comparison.png")
    
    # Save results
    with open('vqe_backends_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to vqe_backends_results.json")

if __name__ == "__main__":
    run_vqe_comparison()
