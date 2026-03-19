"""Estimate and plot QEC logical-error thresholds with publication-quality styling.

The script sweeps physical error rates for a code family and decoder, evaluates
logical error rate using Stim sampling, and exports both a CSV table and a
high-resolution figure. It supports a fast `--quick` mode for iteration and a
production mode for paper-quality sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from surface_code_in_stem.decoders import MWPMDecoder, SparseBlossomDecoder, UnionFindDecoder
from surface_code_in_stem.decoders.base import DecoderMetadata, DecoderProtocol
from surface_code_in_stem.dynamic import hexagonal_surface_code, iswap_surface_code, walking_surface_code, xyz2_hexagonal_code
from surface_code_in_stem.surface_code import surface_code_circuit_string


BuilderFn = Callable[[int, int, float], str]


@dataclass(frozen=True)
class ThresholdPoint:
    distance: int
    physical_error_rate: float
    logical_error_rate: float


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
            "font.weight": "regular",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def _builder_for_name(name: str) -> BuilderFn:
    name = name.lower()
    if name == "surface":
        return lambda distance, rounds, p: surface_code_circuit_string(distance=distance, rounds=rounds, p=p)
    if name == "hexagonal":
        return lambda distance, rounds, p: hexagonal_surface_code(distance, rounds, p)
    if name == "walking":
        return lambda distance, rounds, p: walking_surface_code(distance, rounds, p)
    if name == "iswap":
        return lambda distance, rounds, p: iswap_surface_code(distance, rounds, p)
    if name == "xyz2":
        return lambda distance, rounds, p: xyz2_hexagonal_code(distance, rounds, p)
    raise ValueError(f"Unknown builder '{name}'.")


def _decoder_for_name(name: str) -> DecoderProtocol:
    name = name.lower()
    if name == "mwpm":
        return MWPMDecoder()
    if name == "union_find":
        return UnionFindDecoder()
    if name == "sparse_blossom":
        return SparseBlossomDecoder()
    raise ValueError(f"Unknown decoder '{name}'.")


def _evaluate_logical_error_rate(circuit_string: str, decoder: DecoderProtocol, shots: int, seed: int) -> float:
    import stim

    circuit = stim.Circuit(circuit_string)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detector_samples, observable_samples = sampler.sample(shots, separate_observables=True)
    metadata = DecoderMetadata(
        num_observables=circuit.num_observables,
        detector_error_model=circuit.detector_error_model(decompose_errors=True),
        circuit=circuit,
        seed=seed,
    )
    output = decoder.decode(detector_samples, metadata)
    predictions = np.asarray(output.logical_predictions, dtype=observable_samples.dtype)
    return float(np.mean(np.logical_xor(predictions, observable_samples)[:, 0]))


def _sweep_threshold(
    *,
    builder_name: str,
    decoder_name: str,
    distances: list[int],
    p_values: np.ndarray,
    shots: int,
    rounds_offset: int,
    seed: int,
) -> list[ThresholdPoint]:
    builder = _builder_for_name(builder_name)
    decoder = _decoder_for_name(decoder_name)

    points: list[ThresholdPoint] = []
    for d_idx, distance in enumerate(distances):
        rounds = max(1, distance + rounds_offset)
        for p_idx, p in enumerate(p_values):
            circuit_string = builder(distance, rounds, float(p))
            ler = _evaluate_logical_error_rate(circuit_string, decoder, shots=shots, seed=seed + 31 * d_idx + p_idx)
            points.append(ThresholdPoint(distance=distance, physical_error_rate=float(p), logical_error_rate=ler))
    return points


def _estimate_crossing(points: list[ThresholdPoint], low_distance: int, high_distance: int) -> float | None:
    low = sorted((p for p in points if p.distance == low_distance), key=lambda x: x.physical_error_rate)
    high = sorted((p for p in points if p.distance == high_distance), key=lambda x: x.physical_error_rate)
    if len(low) != len(high) or not low:
        return None

    diffs = np.array([h.logical_error_rate - l.logical_error_rate for l, h in zip(low, high)], dtype=np.float64)
    p_values = np.array([p.physical_error_rate for p in low], dtype=np.float64)

    for idx in range(len(diffs) - 1):
        if diffs[idx] == 0:
            return float(p_values[idx])
        if diffs[idx] * diffs[idx + 1] < 0:
            x0, x1 = p_values[idx], p_values[idx + 1]
            y0, y1 = diffs[idx], diffs[idx + 1]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return None


def _plot_threshold(points: list[ThresholdPoint], *, builder: str, decoder: str, output_dir: Path, crossing: float | None) -> None:
    _ensure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = {}
    for point in points:
        df.setdefault(point.distance, []).append(point)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    colors = (sns.color_palette("colorblind", n_colors=len(df)) if sns is not None else None)

    for idx, (distance, curve) in enumerate(sorted(df.items())):
        color = None if colors is None else colors[idx]
        curve = sorted(curve, key=lambda x: x.physical_error_rate)
        xs = [p.physical_error_rate for p in curve]
        ys = [p.logical_error_rate for p in curve]
        ax.semilogy(xs, ys, marker="o", linewidth=2.2, markersize=6, color=color, label=fr"$d={distance}$")

    if crossing is not None:
        ax.axvline(crossing, linestyle="--", linewidth=2.0, color="black", alpha=0.75, label=fr"Estimated threshold $\approx {crossing:.4f}$")

    ax.set_title(f"{builder.title()} code threshold with {decoder.upper()} decoding")
    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Logical error rate $p_L$")
    ax.set_ylim(bottom=1e-5)
    ax.legend(title="Code distance / threshold", loc="best")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.75)
    fig.tight_layout()

    fig.savefig(output_dir / f"threshold_{builder}_{decoder}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"threshold_{builder}_{decoder}.png", bbox_inches="tight")
    plt.close(fig)


def _write_csv(points: list[ThresholdPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["distance", "physical_error_rate", "logical_error_rate"])
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "distance": point.distance,
                    "physical_error_rate": point.physical_error_rate,
                    "logical_error_rate": point.logical_error_rate,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate logical-error thresholds for QEC codes.")
    parser.add_argument("--builder", type=str, default="surface", choices=["surface", "hexagonal", "walking", "iswap", "xyz2"])
    parser.add_argument("--decoder", type=str, default="mwpm", choices=["mwpm", "union_find", "sparse_blossom"])
    parser.add_argument("--quick", action="store_true", help="Use a small sweep for fast iteration.")
    parser.add_argument("--output-dir", type=str, default="artifacts/thresholds")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if find_spec("stim") is None:
        summary = {
            "builder": args.builder,
            "decoder": args.decoder,
            "quick": args.quick,
            "skipped": True,
            "reason": "stim is not available in the current Python environment",
        }
        print(json.dumps(summary, indent=2))
        return

    if args.quick:
        distances = [3, 5]
        p_values = np.linspace(0.002, 0.02, 7)
        shots = 256
        rounds_offset = 0
    else:
        distances = [3, 5, 7, 9]
        p_values = np.linspace(0.001, 0.02, 13)
        shots = 2048
        rounds_offset = 0

    points = _sweep_threshold(
        builder_name=args.builder,
        decoder_name=args.decoder,
        distances=distances,
        p_values=p_values,
        shots=shots,
        rounds_offset=rounds_offset,
        seed=args.seed,
    )
    crossing = _estimate_crossing(points, distances[0], distances[-1])

    output_dir = Path(args.output_dir)
    _write_csv(points, output_dir / f"threshold_{args.builder}_{args.decoder}.csv")
    _plot_threshold(points, builder=args.builder, decoder=args.decoder, output_dir=output_dir, crossing=crossing)

    summary = {
        "builder": args.builder,
        "decoder": args.decoder,
        "quick": args.quick,
        "distances": distances,
        "shots": shots,
        "threshold_estimate": crossing,
    }
    with (output_dir / f"threshold_{args.builder}_{args.decoder}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
