# Circuit Optimization & Simulation

Constructs high-performance syndrome extraction circuits and runs large-scale simulations for quantum error correction.

## Primary Focus

- **Clustered-Cyclic (CC) Codes**: Implement these LP code variants. Enable parallel product surgery for logical gates.
- **XYZ2 Hexagonal Code**: Build concatenated YZZY surface code + phase-flip code circuits.
- **GKP Qudits**: Implement the SBS (Small-Big-Small) protocol using Echoed Conditional Displacement (ECD) gates.
- **Syndrome extraction**: Build circuits; integrate tools like `LR-circuits` (https://github.com/PurePhys/LR-circuits) for Left-Right syndrome extraction in CSS codes.
- **Circuit generation**: Optimize Stim circuits for dynamic surface codes (`surface_code_in_stem/dynamic/`) and qLDPC codes (`codes/qldpc/`).
- **Skinny Logic**: Construct concatenated codes (e.g., "iceberg" codes) to achieve high encoding rates (e.g., 48 logical qubits from 98 physical). Implement code concatenation logic.
- **Decoder management**: Interface with MWPM, BPOSD, and Union-Find decoders for logical error rate evaluation.
- **TensorHyper**: Use TensorHyper (https://github.com/jqi41/TensorHyper) for tensor network-based simulation and contraction where appropriate.

## Optimization & Parallelization (MANDATORY)

- **Parallelization**: Always parallelize where possible. Large-scale Monte Carlo simulations of logical error rates MUST be parallelized (e.g., `multiprocessing`, `joblib`, or Ray).
- **Stim efficiency**: Use Stim's C++ samplers in bulk. Avoid Python loops over individual shots when Stim can sample in bulk.
- **Memory**: Be mindful of memory when generating large parity-check matrices or simulating high-distance codes.

## Factual Verification

For hardware or library updates (IBM specs, Stim features, etc.), perform a web search first before responding.
