import json
import matplotlib.pyplot as plt
import os

def plot_convergence():
    # Load VQE history
    try:
        with open('vqe_history.json', 'r') as f:
            vqe_history = json.load(f)
    except FileNotFoundError:
        print("VQE history not found.")
        vqe_history = []

    # Load GQE history
    try:
        with open('gqe_history.json', 'r') as f:
            gqe_history = json.load(f)
    except FileNotFoundError:
        print("GQE history not found.")
        gqe_history = []

    # Exact ground energy for the 2-qubit "H2 minimal" Hamiltonian used by VQE/GQE.
    # (Computed by diagonalizing the 4x4 matrix for the coefficients in `Hamiltonian::h2_minimal()`.)
    h2_ground_state = -1.8572750302023795

    # Convert VQE history (every cost eval) into a monotone "best-so-far" curve for easier comparison.
    if vqe_history:
        best = float("inf")
        vqe_best = []
        for e in vqe_history:
            if e < best:
                best = e
            vqe_best.append(best)
    else:
        vqe_best = []

    os.makedirs('docs/assets', exist_ok=True)

    # --- VQE-only plot
    if vqe_best:
        plt.figure(figsize=(10, 5))
        plt.plot(vqe_best, label='VQE best-so-far (Nelder-Mead)', color='blue', linestyle='-')
        plt.axhline(y=h2_ground_state, color='red', linestyle='--', label='Exact Ground (2-qubit H2 minimal)')
        plt.xlabel('Cost evaluations')
        plt.ylabel('Energy (Hartree)')
        plt.title('VQE Convergence (2-qubit H2 minimal)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig('docs/assets/vqe_convergence.png', dpi=300)
        print("Saved docs/assets/vqe_convergence.png")

    # --- GQE-only plot
    if gqe_history:
        plt.figure(figsize=(10, 5))
        plt.plot(gqe_history, label='GQE best (per generation)', color='green', linestyle='-')
        plt.axhline(y=h2_ground_state, color='red', linestyle='--', label='Exact Ground (2-qubit H2 minimal)')
        plt.xlabel('Generation')
        plt.ylabel('Energy (Hartree)')
        plt.title('GQE Convergence (2-qubit H2 minimal)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig('docs/assets/gqe_convergence.png', dpi=300)
        print("Saved docs/assets/gqe_convergence.png")

    # --- Combined plot
    plt.figure(figsize=(12, 6))
    if vqe_best:
        plt.plot(vqe_best, label='VQE best-so-far (Nelder-Mead)', color='blue', linestyle='-')
    if gqe_history:
        plt.plot(gqe_history, label='GQE best (per generation)', color='green', linestyle='-')
    plt.axhline(y=h2_ground_state, color='red', linestyle='--', label='Exact Ground (2-qubit H2 minimal)')

    plt.xlabel('Step (not directly comparable: evals vs generations)')
    plt.ylabel('Energy (Hartree)')
    plt.title('VQE vs. GQE Convergence (2-qubit H2 minimal)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.savefig('docs/assets/algorithm_comparison.png', dpi=300)
    print("Saved docs/assets/algorithm_comparison.png")

if __name__ == "__main__":
    plot_convergence()
