"""Benchmark decoders and code families with publication-quality outputs.

This script runs two complementary benchmark suites:

1. Circuit-level decoders on surface/dynamic codes using Stim sampling.
2. Parity-matrix decoders (Neural BP / GNN / JAX Neural BP) on synthetic qLDPC
   syndrome-recovery tasks.

Both suites support a fast `--quick` mode for iteration and a production mode
for broader sweeps suitable for reports or papers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from importlib.util import find_spec

import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from surface_code_in_stem.decoders import MWPMDecoder, SparseBlossomDecoder, UnionFindDecoder
from surface_code_in_stem.decoders.base import DecoderMetadata, DecoderProtocol
from surface_code_in_stem.decoders.gnn_decoder import NeuralBPDecoder, qLDPCGNNDecoder
from surface_code_in_stem.dynamic import hexagonal_surface_code, iswap_surface_code, walking_surface_code, xyz2_hexagonal_code
from surface_code_in_stem.rl_nested_learning import _logical_error_rate
from surface_code_in_stem.surface_code import surface_code_circuit_string

try:
    from surface_code_in_stem.decoders.jax_gnn_decoder import JAXNeuralBPDecoder
    HAS_JAX_DECODER = True
except Exception:  # pragma: no cover - optional dependency
    HAS_JAX_DECODER = False


BuilderFn = Callable[[int, int, float], str]


def hamming_code_parity(r: int) -> np.ndarray:
    n = 2 ** r - 1
    h = np.zeros((r, n), dtype=np.uint8)
    for i in range(n):
        bits = format(i + 1, f"0{r}b")
        for j, bit in enumerate(bits):
            h[j, i] = int(bit)
    return h


def hypergraph_product(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r1, n1 = h1.shape
    r2, n2 = h2.shape
    hx = np.hstack([np.kron(h1, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(r1, dtype=np.uint8), h2.T)])
    hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), h2), np.kron(h1.T, np.eye(r2, dtype=np.uint8))])
    return hx.astype(np.uint8, copy=False), hz.astype(np.uint8, copy=False)


def toric_code_parity(size: int) -> tuple[np.ndarray, np.ndarray]:
    num_qubits = 2 * size * size
    num_stabilizers = size * size
    hx = np.zeros((num_stabilizers, num_qubits), dtype=np.uint8)
    hz = np.zeros((num_stabilizers, num_qubits), dtype=np.uint8)

    for row in range(size):
        for col in range(size):
            check_idx = row * size + col
            h_right = 2 * (row * size + col)
            h_left = 2 * (row * size + ((col - 1) % size))
            v_down = 2 * (row * size + col) + 1
            v_up = 2 * (((row - 1) % size) * size + col) + 1
            hx[check_idx, [h_right, h_left, v_down, v_up]] = 1

            h_top = 2 * (row * size + col)
            h_bottom = 2 * (((row + 1) % size) * size + col)
            v_left = 2 * (row * size + col) + 1
            v_right = 2 * (row * size + ((col + 1) % size)) + 1
            hz[check_idx, [h_top, h_bottom, v_left, v_right]] = 1

    return hx, hz


@dataclass(frozen=True)
class BenchmarkRow:
    domain: str
    family: str
    decoder: str
    distance: int
    physical_error_rate: float
    shots: int
    metric_name: str
    metric_value: float


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


def _circuit_builders() -> dict[str, BuilderFn]:
    return {
        "surface": lambda distance, rounds, p: surface_code_circuit_string(distance=distance, rounds=rounds, p=p),
        "hexagonal": lambda distance, rounds, p: hexagonal_surface_code(distance, rounds, p),
        "walking": lambda distance, rounds, p: walking_surface_code(distance, rounds, p),
        "iswap": lambda distance, rounds, p: iswap_surface_code(distance, rounds, p),
        "xyz2": lambda distance, rounds, p: xyz2_hexagonal_code(distance, rounds, p),
    }


def _circuit_decoders() -> dict[str, DecoderProtocol]:
    return {
        "mwpm": MWPMDecoder(),
        "union_find": UnionFindDecoder(),
        "sparse_blossom": SparseBlossomDecoder(),
    }


def _synthetic_qldpc_task(name: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "toric":
        return toric_code_parity(size)
    if name == "hypergraph_product":
        h = hamming_code_parity(3)
        return hypergraph_product(h, h)
    raise ValueError(f"Unknown qLDPC family '{name}'.")


def _evaluate_circuit_logical_error(builder: BuilderFn, decoder: DecoderProtocol, *, distance: int, rounds: int, p: float, shots: int, seed: int) -> float:
    circuit_artifact = builder(distance, rounds, p)
    circuit_string = circuit_artifact if isinstance(circuit_artifact, str) else str(circuit_artifact)
    return _logical_error_rate(circuit_string, shots=shots, seed=seed, decoder=decoder)


def _evaluate_parity_decoder(
    *,
    decoder_name: str,
    hx: np.ndarray,
    hz: np.ndarray,
    p: float,
    shots: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    num_qubits = hx.shape[1]
    errors_x = rng.random((shots, num_qubits)) < p
    errors_z = rng.random((shots, num_qubits)) < p

    syndrome_x = (errors_z @ hx.T) % 2
    syndrome_z = (errors_x @ hz.T) % 2

    result: dict[str, float] = {}

    def _score_decoder(decoder, syndrome: np.ndarray, parity: np.ndarray, true_errors: np.ndarray, prefix: str) -> None:
        if isinstance(decoder, qLDPCGNNDecoder):
            import torch
            logits = decoder(torch.tensor(syndrome, dtype=torch.int64), torch.tensor(parity, dtype=torch.float32).to_sparse())
            pred = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(np.uint8)
        elif HAS_JAX_DECODER and isinstance(decoder, JAXNeuralBPDecoder):
            pred = (decoder.decode_batch(decoder.init_params, syndrome.astype(np.float32), parity) > 0.5).astype(np.uint8)
        elif isinstance(decoder, NeuralBPDecoder):
            import torch
            probs = decoder(
                torch.tensor(syndrome, dtype=torch.float32),
                torch.tensor(parity, dtype=torch.float32).to_sparse(),
            )
            pred = (probs.detach().cpu().numpy() > 0.5).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported decoder type: {type(decoder)!r}")

        bit_error_rate = float(np.mean(pred != true_errors))
        block_error_rate = float(np.mean(np.any(pred != true_errors, axis=1)))
        syndrome_residual = float(np.mean(((pred @ parity.T) % 2) != syndrome))
        result[f"{prefix}_bit_error_rate"] = bit_error_rate
        result[f"{prefix}_block_error_rate"] = block_error_rate
        result[f"{prefix}_syndrome_residual"] = syndrome_residual

    decoder = {
        "neural_bp": NeuralBPDecoder(num_vars=num_qubits, num_checks=hx.shape[0], max_iter=6),
        "gnn": qLDPCGNNDecoder(num_vars=num_qubits, num_checks=hx.shape[0], hidden_dim=32, num_layers=3),
    }.get(decoder_name)
    if decoder is None and HAS_JAX_DECODER:
        decoder = JAXNeuralBPDecoder(num_vars=num_qubits, num_checks=hx.shape[0], max_iter=6)
    if decoder is None:
        raise RuntimeError(f"Unsupported decoder '{decoder_name}'.")

    _score_decoder(decoder, syndrome_x, hx, errors_z.astype(np.uint8), prefix="x")
    _score_decoder(decoder, syndrome_z, hz, errors_x.astype(np.uint8), prefix="z")
    result["mean_block_error_rate"] = float((result["x_block_error_rate"] + result["z_block_error_rate"]) / 2.0)
    result["mean_bit_error_rate"] = float((result["x_bit_error_rate"] + result["z_bit_error_rate"]) / 2.0)
    return result


def _write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in BenchmarkRow.__dataclass_fields__.values()])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _plot_rows(rows: list[BenchmarkRow], *, title: str, output_prefix: Path) -> None:
    _ensure_style()
    if not rows:
        return
    families = sorted(set(row.family for row in rows))
    decoders = sorted(set(row.decoder for row in rows))

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    palette = sns.color_palette("colorblind", n_colors=len(decoders)) if sns is not None else [None] * len(decoders)

    for color, decoder in zip(palette, decoders):
        subset = [row for row in rows if row.decoder == decoder]
        xs = [f"{row.family}\n$d={row.distance}$" for row in subset]
        ys = [row.metric_value for row in subset]
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=6, label=decoder, color=color)

    ax.set_title(title)
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Decoder")
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _build_circuit_rows(quick: bool, seed: int) -> list[BenchmarkRow]:
    builders = _circuit_builders()
    decoders = _circuit_decoders()
    if quick:
        families = ["surface", "hexagonal"]
        distances = [3, 5]
        p_values = [0.001, 0.01]
        shots = 128
    else:
        families = ["surface", "hexagonal", "walking", "iswap", "xyz2"]
        distances = [3, 5, 7]
        p_values = [0.001, 0.005, 0.01]
        shots = 512

    rows: list[BenchmarkRow] = []
    for family in families:
        builder = builders[family]
        for decoder_name, decoder in decoders.items():
            for distance in distances:
                for p_idx, p in enumerate(p_values):
                    metric = _evaluate_circuit_logical_error(
                        builder,
                        decoder,
                        distance=distance,
                        rounds=max(2, distance),
                        p=p,
                        shots=shots,
                        seed=seed + 19 * distance + p_idx,
                    )
                    rows.append(
                        BenchmarkRow(
                            domain="circuit",
                            family=family,
                            decoder=decoder_name,
                            distance=distance,
                            physical_error_rate=p,
                            shots=shots,
                            metric_name="logical_error_rate",
                            metric_value=metric,
                        )
                    )
    return rows


def _build_qldpc_rows(quick: bool, seed: int) -> list[BenchmarkRow]:
    if quick:
        families = ["toric"]
        sizes = [3]
        p_values = [0.01]
        shots = 256
        decoder_names = ["neural_bp", "gnn"] + (["jax_neural_bp"] if HAS_JAX_DECODER else [])
    else:
        families = ["toric", "hypergraph_product"]
        sizes = [3, 5]
        p_values = [0.005, 0.01, 0.02]
        shots = 512
        decoder_names = ["neural_bp", "gnn"] + (["jax_neural_bp"] if HAS_JAX_DECODER else [])

    rows: list[BenchmarkRow] = []
    for family in families:
        for size in sizes:
            hx, hz = _synthetic_qldpc_task(family, size)
            for decoder_name in decoder_names:
                for p_idx, p in enumerate(p_values):
                    metrics = _evaluate_parity_decoder(
                        decoder_name=decoder_name,
                        hx=hx,
                        hz=hz,
                        p=p,
                        shots=shots,
                        seed=seed + 41 * size + p_idx,
                    )
                    rows.append(
                        BenchmarkRow(
                            domain="qldpc",
                            family=f"{family}_n{hx.shape[1]}",
                            decoder=decoder_name,
                            distance=size,
                            physical_error_rate=p,
                            shots=shots,
                            metric_name="mean_block_error_rate",
                            metric_value=metrics["mean_block_error_rate"],
                        )
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark decoders across circuit and qLDPC families.")
    parser.add_argument("--quick", action="store_true", help="Use a fast benchmark configuration.")
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks")
    parser.add_argument("--suite", type=str, choices=["circuit", "qldpc", "all"], default="all")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rows: list[BenchmarkRow] = []
    stim_available = find_spec("stim") is not None
    if args.suite in {"circuit", "all"}:
        if not stim_available:
            print("[warn] Stim is not available; skipping circuit-level benchmark suite.")
        else:
            rows.extend(_build_circuit_rows(args.quick, args.seed))
    if args.suite in {"qldpc", "all"}:
        rows.extend(_build_qldpc_rows(args.quick, args.seed + 1000))

    output_dir = Path(args.output_dir)
    _write_csv(rows, output_dir / f"decoder_benchmark_{args.suite}.csv")

    summary = [row.__dict__ for row in rows]
    with (output_dir / f"decoder_benchmark_{args.suite}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if rows:
        _plot_rows(rows, title="Decoder benchmark summary", output_prefix=output_dir / f"decoder_benchmark_{args.suite}")

    print(json.dumps({"rows": len(rows), "suite": args.suite, "quick": args.quick}, indent=2))


if __name__ == "__main__":
    main()
