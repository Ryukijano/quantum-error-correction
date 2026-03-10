# Benchmark reproduction map

This document maps each benchmark template in `benchmarks/specs/` to commonly cited literature-level claims and highlights caveats when using these templates as proxies instead of full decoder-in-the-loop simulations.

## 1) Threshold sweep (`threshold_sweep.yaml`)

- **Target claim:** logical error curves for increasing code distance cross near a threshold region (surface code style behavior).
- **Typical references:** Fowler et al., *Surface codes: Towards practical large-scale quantum computation* (PRA 2012); Dennis et al., *Topological quantum memory* (JMP 2002).
- **Caveats:** this benchmark uses a compact power-law surrogate rather than explicit syndrome sampling and MWPM decoding, so extracted threshold values are illustrative and not publication-grade.

## 2) Distance scaling (`distance_scaling.yaml`)

- **Target claim:** below threshold, logical error rate drops roughly exponentially with code distance.
- **Typical references:** textbook surface-code scaling arguments and supplemental scaling analyses in architecture papers.
- **Caveats:** assumes a fixed effective threshold and prefactor; hardware-specific correlated noise, leakage, and decoder mismatch are not represented.

## 3) Repetition code exponential suppression (`repetition_suppression.json`)

- **Target claim:** odd-distance repetition code under iid bit-flip noise exhibits exponential suppression from majority vote decoding.
- **Typical references:** standard repetition-code derivations; didactic Stim tutorials.
- **Caveats:** model is exact only for iid single-parameter bit-flip channels and perfect measurements, without time-correlated noise or measurement faults.

## 4) Overhead comparison (`overhead_comparison.yaml`)

- **Target claim:** compare order-of-magnitude resource and latency trade-offs between surface-code and qLDPC-style assumptions.
- **Typical references:** recent qLDPC resource-estimation discussions (e.g., Panteleev-Kalachev line of work and follow-on architecture analyses).
- **Caveats:** qLDPC terms here are intentionally templated assumptions (rate, scaling exponent, latency law) and should be replaced with family-specific decoder/architecture data before drawing hardware conclusions.

## Practical reproducibility notes

1. Pin package versions and random seeds for deterministic CSV outputs.
2. Archive generated `benchmarks/results/*.csv` with commits.
3. Treat plots as quick diagnostics; for papers, regenerate from raw artifacts and include confidence intervals from Monte Carlo trials.
