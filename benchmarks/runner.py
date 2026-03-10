"""Benchmark execution and artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.config import BenchmarkSpec
from benchmarks.models import (
    estimate_qldpc_overhead,
    estimate_surface_overhead,
    qldpc_logical_error_rate,
    repetition_logical_error_rate,
    suppression_factors,
    surface_code_logical_error_rate,
)


def _distance_list(parameters: dict[str, Any]) -> list[int]:
    distances = parameters.get("distances", [3, 5, 7])
    return [int(d) for d in distances]


def run_spec(spec: BenchmarkSpec, output_dir: str | Path) -> pd.DataFrame:
    """Execute a benchmark spec and materialize dataframe metrics."""
    if spec.benchmark_type == "threshold_sweep":
        df = _run_threshold_sweep(spec.parameters)
    elif spec.benchmark_type == "distance_scaling":
        df = _run_distance_scaling(spec.parameters)
    elif spec.benchmark_type == "repetition_suppression":
        df = _run_repetition_suppression(spec.parameters)
    elif spec.benchmark_type == "overhead_comparison":
        df = _run_overhead_comparison(spec.parameters)
    else:
        raise ValueError(f"Unsupported benchmark type: {spec.benchmark_type}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = spec.output_prefix or spec.name
    csv_path = out / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    _save_plot(spec, df, out / f"{prefix}.png")
    return df


def _run_threshold_sweep(parameters: dict[str, Any]) -> pd.DataFrame:
    distances = _distance_list(parameters)
    p_values = [float(p) for p in parameters.get("physical_error_rates", [1e-4, 3e-4, 1e-3, 3e-3])]
    threshold = float(parameters.get("threshold", 1e-2))

    rows: list[dict[str, Any]] = []
    for d in distances:
        for p in p_values:
            logical = surface_code_logical_error_rate(d, p, threshold=threshold)
            qubits, cycles, latency = estimate_surface_overhead(d)
            rows.append(
                {
                    "benchmark": "threshold_sweep",
                    "distance": d,
                    "physical_error_rate": p,
                    "logical_error_rate": logical,
                    "qubit_overhead": qubits,
                    "cycle_count_estimate": cycles,
                    "decode_latency_us_estimate": latency,
                }
            )
    return pd.DataFrame(rows)


def _run_distance_scaling(parameters: dict[str, Any]) -> pd.DataFrame:
    distances = _distance_list(parameters)
    p = float(parameters.get("physical_error_rate", 1e-3))
    threshold = float(parameters.get("threshold", 1e-2))

    logical_rates = [surface_code_logical_error_rate(d, p, threshold=threshold) for d in distances]
    suppressions = suppression_factors(logical_rates)

    rows = []
    for d, logical, suppression in zip(distances, logical_rates, suppressions):
        qubits, cycles, latency = estimate_surface_overhead(d)
        rows.append(
            {
                "benchmark": "distance_scaling",
                "distance": d,
                "physical_error_rate": p,
                "logical_error_rate": logical,
                "suppression_factor": suppression,
                "qubit_overhead": qubits,
                "cycle_count_estimate": cycles,
                "decode_latency_us_estimate": latency,
            }
        )
    return pd.DataFrame(rows)


def _run_repetition_suppression(parameters: dict[str, Any]) -> pd.DataFrame:
    distances = _distance_list(parameters)
    p_values = [float(p) for p in parameters.get("physical_error_rates", [1e-3, 2e-3])]

    rows = []
    for p in p_values:
        logical_rates = [repetition_logical_error_rate(d, p) for d in distances]
        suppressions = suppression_factors(logical_rates)
        for d, logical, suppression in zip(distances, logical_rates, suppressions):
            rows.append(
                {
                    "benchmark": "repetition_suppression",
                    "distance": d,
                    "physical_error_rate": p,
                    "logical_error_rate": logical,
                    "suppression_factor": suppression,
                    "qubit_overhead": d,
                    "cycle_count_estimate": d,
                    "decode_latency_us_estimate": float(d),
                }
            )
    return pd.DataFrame(rows)


def _run_overhead_comparison(parameters: dict[str, Any]) -> pd.DataFrame:
    distances = _distance_list(parameters)
    p = float(parameters.get("physical_error_rate", 1e-3))
    rows = []

    for d in distances:
        s_ler = surface_code_logical_error_rate(d, p, threshold=float(parameters.get("surface_threshold", 1e-2)))
        s_q, s_c, s_lat = estimate_surface_overhead(d)
        rows.append(
            {
                "benchmark": "overhead_comparison",
                "assumption": "surface",
                "distance": d,
                "physical_error_rate": p,
                "logical_error_rate": s_ler,
                "qubit_overhead": s_q,
                "cycle_count_estimate": s_c,
                "decode_latency_us_estimate": s_lat,
            }
        )

        q_ler = qldpc_logical_error_rate(
            d,
            p,
            threshold=float(parameters.get("qldpc_threshold", 5e-3)),
            alpha=float(parameters.get("qldpc_alpha", 0.4)),
        )
        q_q, q_c, q_lat = estimate_qldpc_overhead(d, rate=float(parameters.get("qldpc_rate", 0.1)))
        rows.append(
            {
                "benchmark": "overhead_comparison",
                "assumption": "qldpc",
                "distance": d,
                "physical_error_rate": p,
                "logical_error_rate": q_ler,
                "qubit_overhead": q_q,
                "cycle_count_estimate": q_c,
                "decode_latency_us_estimate": q_lat,
            }
        )
    return pd.DataFrame(rows)


def _save_plot(spec: BenchmarkSpec, df: pd.DataFrame, plot_path: Path) -> None:
    """Save a basic plot for quick report generation."""
    plt.figure(figsize=(7, 4.5))
    if spec.benchmark_type == "threshold_sweep":
        for distance, group in df.groupby("distance"):
            plt.loglog(group["physical_error_rate"], group["logical_error_rate"], marker="o", label=f"d={distance}")
        plt.xlabel("Physical error rate")
        plt.ylabel("Logical error rate")
        plt.legend()
    elif spec.benchmark_type in {"distance_scaling", "repetition_suppression"}:
        if "physical_error_rate" in df.columns and df["physical_error_rate"].nunique() > 1:
            for physical_error_rate, group in df.groupby("physical_error_rate"):
                plt.semilogy(group["distance"], group["logical_error_rate"], marker="o", label=f"p={physical_error_rate}")
            plt.legend()
        else:
            plt.semilogy(df["distance"], df["logical_error_rate"], marker="o")
        plt.xlabel("Code distance")
        plt.ylabel("Logical error rate")
    else:
        pivot = df.pivot(index="distance", columns="assumption", values="qubit_overhead")
        pivot.plot(kind="bar", ax=plt.gca())
        plt.ylabel("Qubit overhead")

    plt.title(spec.name)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
