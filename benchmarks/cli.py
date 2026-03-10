"""CLI entrypoint for running quantum error-correction benchmark specs."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.config import load_spec
from benchmarks.runner import run_spec


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for benchmark execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "specs",
        nargs="+",
        help="One or more YAML/JSON spec files to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory where CSV and plots are written.",
    )
    return parser


def main() -> None:
    """CLI main function."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec_path in args.specs:
        spec = load_spec(spec_path)
        frame = run_spec(spec, output_dir=output_dir)
        print(f"[{spec.name}] rows={len(frame)} -> {output_dir}")


if __name__ == "__main__":
    main()
