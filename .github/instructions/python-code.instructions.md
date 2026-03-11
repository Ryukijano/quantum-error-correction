---
applyTo: "**/*.py"
---

## Python contribution requirements

- Preserve existing import paths and re-exports used by notebooks/scripts.
- Use descriptive names for qubits, detectors, and rounds; avoid cryptic one-letter variables outside tight loops.
- Validate user inputs for public functions (distance, rounds, probabilities, shot counts).
- For numerically heavy sections, prefer vectorized NumPy operations over Python loops when equivalent.
- Keep error messages actionable and mention expected ranges/types.
