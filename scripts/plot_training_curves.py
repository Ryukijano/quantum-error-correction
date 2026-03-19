"""Plot RL training curves for QEC decoding and calibration.

The script loads JSON histories written by `scripts/train_sota_rl.py` and
exports publication-quality panels showing reward, success rate, and logical
error rate trends. If no histories are found and `--quick` is set, it creates a
small synthetic demo so the plotting pipeline stays executable in fresh clones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None

sys.path.append(str(Path(__file__).resolve().parents[1]))


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


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(np.float64, copy=False)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0], dtype=np.float64)
    return np.concatenate([pad, smoothed])


def _load_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"History file {path} must contain a JSON list.")
    return payload


def _synthetic_history(kind: str, length: int = 32) -> list[dict[str, float]]:
    rng = np.random.default_rng(abs(hash(kind)) % (2**32))
    episodes = np.arange(1, length + 1, dtype=np.float64)
    if kind == "ppo":
        reward = np.tanh(np.linspace(-1.5, 1.8, length)) + 0.08 * rng.standard_normal(length)
        rl_success = np.clip(np.linspace(0.35, 0.88, length) + 0.05 * rng.standard_normal(length), 0.0, 1.0)
        mwpm_success = np.clip(np.linspace(0.45, 0.9, length) + 0.03 * rng.standard_normal(length), 0.0, 1.0)
        return [
            {"episode": float(e), "reward": float(r), "rl_success": float(rs), "mwpm_success": float(ms)}
            for e, r, rs, ms in zip(episodes, reward, rl_success, mwpm_success)
        ]
    reward = -np.abs(np.linspace(0.08, 0.02, length) + 0.01 * rng.standard_normal(length))
    logical_error_rate = np.clip(np.linspace(0.08, 0.015, length) + 0.01 * rng.standard_normal(length), 0.0, 1.0)
    effective_p = np.clip(np.linspace(0.05, 0.012, length) + 0.005 * rng.standard_normal(length), 0.0, 1.0)
    return [
        {"episode": float(e), "reward": float(r), "logical_error_rate": float(ler), "effective_p": float(p)}
        for e, r, ler, p in zip(episodes, reward, logical_error_rate, effective_p)
    ]


def _extract_series(history: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in history if key in row], dtype=np.float64)


def _episode_axis(history: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([float(row.get("episode", idx + 1)) for idx, row in enumerate(history)], dtype=np.float64)


def _plot_ppo(ax_reward, ax_success, history: list[dict[str, float]], window: int) -> None:
    episodes = _episode_axis(history)
    reward = _rolling_mean(_extract_series(history, "reward"), window)
    rl_success = _rolling_mean(_extract_series(history, "rl_success"), window)
    mwpm_success = _rolling_mean(_extract_series(history, "mwpm_success"), window)

    palette = sns.color_palette("colorblind") if sns is not None else [None] * 6
    ax_reward.plot(episodes, reward, color=palette[0], linewidth=2.2, label="RL reward")
    ax_reward.set_title("Transformer-PPO decoding")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="best")

    ax_success.plot(episodes, rl_success, color=palette[1], linewidth=2.2, label="RL success")
    ax_success.plot(episodes, mwpm_success, color=palette[2], linewidth=2.2, linestyle="--", label="MWPM success")
    ax_success.set_xlabel("Episode")
    ax_success.set_ylabel("Success rate")
    ax_success.set_ylim(0.0, 1.05)
    ax_success.legend(loc="best")


def _plot_sac(ax_reward, ax_error, history: list[dict[str, float]], window: int) -> None:
    episodes = _episode_axis(history)
    reward = _rolling_mean(_extract_series(history, "reward"), window)
    ler = _rolling_mean(_extract_series(history, "logical_error_rate"), window)
    effective_p = _rolling_mean(_extract_series(history, "effective_p"), window)

    palette = sns.color_palette("colorblind") if sns is not None else [None] * 6
    ax_reward.plot(episodes, reward, color=palette[3], linewidth=2.2, label="Calibration reward")
    ax_reward.set_title("Continuous calibration")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="best")

    ax_error.plot(episodes, ler, color=palette[4], linewidth=2.2, label="$p_L$")
    ax_error.plot(episodes, effective_p, color=palette[5], linewidth=2.2, linestyle="--", label="Effective $p$")
    ax_error.set_xlabel("Episode")
    ax_error.set_ylabel("Error rate")
    ax_error.set_yscale("log")
    ax_error.legend(loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RL training curves for QEC experiments.")
    parser.add_argument("--history-dir", type=str, default="artifacts/rl_training")
    parser.add_argument("--ppo-history", type=str, default="ppo_history.json")
    parser.add_argument("--sac-history", type=str, default="sac_history.json")
    parser.add_argument("--output-dir", type=str, default="artifacts/rl_figures")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Fallback to synthetic demo histories if files are missing.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir)

    ppo_history = _load_history(history_dir / args.ppo_history)
    sac_history = _load_history(history_dir / args.sac_history)

    if not ppo_history and args.quick:
        ppo_history = _synthetic_history("ppo", length=24)
    if not sac_history and args.quick:
        sac_history = _synthetic_history("sac", length=24)

    if not ppo_history and not sac_history:
        raise FileNotFoundError(
            "No training histories were found. Run scripts/train_sota_rl.py first or pass --quick to generate demo curves."
        )

    _ensure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex="col")
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    if ppo_history:
        _plot_ppo(ax00, ax01, ppo_history, args.window)
    else:
        ax00.axis("off")
        ax01.axis("off")

    if sac_history:
        _plot_sac(ax10, ax11, sac_history, args.window)
    else:
        ax10.axis("off")
        ax11.axis("off")

    fig.suptitle("Syndrome-Net RL training curves", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(output_dir / "rl_training_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "rl_training_curves.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "ppo_final_reward": float(_extract_series(ppo_history, "reward")[-1]) if ppo_history else None,
        "ppo_final_success": float(_extract_series(ppo_history, "rl_success")[-1]) if ppo_history else None,
        "sac_final_reward": float(_extract_series(sac_history, "reward")[-1]) if sac_history else None,
        "sac_final_logical_error_rate": float(_extract_series(sac_history, "logical_error_rate")[-1]) if sac_history else None,
        "quick": args.quick,
    }
    with (output_dir / "rl_training_curves_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
