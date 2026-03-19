"""Estimate physical-qubit footprint for a target logical error rate.

This script computes an approximate code distance from a simple threshold-scaling
law and then maps that distance to a physical-qubit footprint for a chosen code
family. The goal is to provide a quick, research-friendly overhead estimator,
not an exact architectural resource calculator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None

sys.path.append(str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class FootprintEstimate:
    family: str
    physical_error_rate: float
    target_logical_error_rate: float
    threshold: float
    prefactor: float
    required_distance: int
    physical_qubits: int


_CODE_MODELS: dict[str, dict[str, float]] = {
    "surface": {"threshold": 1.0e-2, "prefactor": 0.12, "qubit_factor": 2.0},
    "hexagonal": {"threshold": 1.5e-2, "prefactor": 0.12, "qubit_factor": 1.8},
    "walking": {"threshold": 1.4e-2, "prefactor": 0.14, "qubit_factor": 1.8},
    "iswap": {"threshold": 1.3e-2, "prefactor": 0.14, "qubit_factor": 1.8},
    "xyz2": {"threshold": 1.2e-2, "prefactor": 0.15, "qubit_factor": 2.1},
    "toric": {"threshold": 1.1e-2, "prefactor": 0.18, "qubit_factor": 2.0},
    "hypergraph_product": {"threshold": 2.5e-2, "prefactor": 0.2, "qubit_factor": 2.5},
}


def _ensure_style() -> None:
    if sns is not None:
        sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def estimate_distance(target_logical_error_rate: float, physical_error_rate: float, threshold: float, prefactor: float) -> int:
    if not (0.0 < physical_error_rate < threshold):
        raise ValueError("physical_error_rate must be positive and below the threshold for the scaling law to apply.")
    if not (0.0 < target_logical_error_rate < 1.0):
        raise ValueError("target_logical_error_rate must be in (0, 1).")

    ratio = target_logical_error_rate / prefactor
    exponent = math.log(physical_error_rate / threshold)
    if exponent >= 0:
        raise ValueError("physical_error_rate must be below threshold.")
    d_real = 2.0 * math.log(ratio) / exponent - 1.0
    d = max(3, int(math.ceil(d_real)))
    if d % 2 == 0:
        d += 1
    return d


def qubits_for_family(family: str, distance: int) -> int:
    model = _CODE_MODELS[family]
    return int(math.ceil(model["qubit_factor"] * distance * distance))


def estimate_footprint(
    family: str,
    physical_error_rate: float,
    target_logical_error_rate: float,
) -> FootprintEstimate:
    model = _CODE_MODELS[family]
    distance = estimate_distance(
        target_logical_error_rate=target_logical_error_rate,
        physical_error_rate=physical_error_rate,
        threshold=model["threshold"],
        prefactor=model["prefactor"],
    )
    return FootprintEstimate(
        family=family,
        physical_error_rate=physical_error_rate,
        target_logical_error_rate=target_logical_error_rate,
        threshold=model["threshold"],
        prefactor=model["prefactor"],
        required_distance=distance,
        physical_qubits=qubits_for_family(family, distance),
    )


def _write_csv(estimates: list[FootprintEstimate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FootprintEstimate.__dataclass_fields__))
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(estimate.__dict__)


def _plot(estimates: list[FootprintEstimate], output_prefix: Path) -> None:
    _ensure_style()
    if not estimates:
        return

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    palette = sns.color_palette("colorblind", n_colors=len(estimates)) if sns is not None else [None] * len(estimates)
    for color, estimate in zip(palette, estimates):
        ax.bar(
            estimate.family,
            estimate.physical_qubits,
            color=color,
            label=fr"$d={estimate.required_distance}$, $p_\mathrm{{th}}={estimate.threshold:.3g}$",
        )
    ax.set_title("Approximate teraquop footprint estimate")
    ax.set_ylabel("Estimated physical qubits")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Family / estimate", loc="best")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate physical qubit footprint to reach a target logical error rate.")
    parser.add_argument("--family", type=str, default="surface", choices=sorted(_CODE_MODELS))
    parser.add_argument("--physical-error-rate", type=float, default=1e-3)
    parser.add_argument("--target-logical-error-rate", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=str, default="artifacts/footprints")
    parser.add_argument("--quick", action="store_true", help="Produce a small report over a few representative families.")
    args = parser.parse_args()

    if args.quick:
        families = ["surface", "hexagonal", "toric"]
    else:
        families = [args.family]

    estimates = [
        estimate_footprint(family, args.physical_error_rate, args.target_logical_error_rate)
        for family in families
    ]

    output_dir = Path(args.output_dir)
    _write_csv(estimates, output_dir / "teraquop_footprint.csv")
    _plot(estimates, output_dir / "teraquop_footprint")

    summary = [estimate.__dict__ for estimate in estimates]
    with (output_dir / "teraquop_footprint.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
