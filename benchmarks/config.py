"""Configuration loading for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class BenchmarkSpec:
    """Normalized benchmark specification parsed from YAML/JSON."""

    name: str
    benchmark_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    output_prefix: str | None = None


SUPPORTED_BENCHMARKS = {
    "threshold_sweep",
    "distance_scaling",
    "repetition_suppression",
    "overhead_comparison",
}


def _read_raw_spec(path: Path) -> dict[str, Any]:
    """Read a benchmark specification from YAML or JSON."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"JSON spec must map to a dictionary: {path}")
        return data
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"YAML spec must map to a dictionary: {path}")
        return data
    raise ValueError(f"Unsupported spec format '{suffix}' for {path}")


def load_spec(path: str | Path) -> BenchmarkSpec:
    """Load and validate a benchmark spec file."""
    spec_path = Path(path)
    raw = _read_raw_spec(spec_path)

    benchmark_type = raw.get("benchmark_type")
    if benchmark_type not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark_type '{benchmark_type}'. "
            f"Supported: {sorted(SUPPORTED_BENCHMARKS)}"
        )

    name = raw.get("name") or spec_path.stem
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("'parameters' must be a dictionary")

    output_prefix = raw.get("output_prefix")
    if output_prefix is not None and not isinstance(output_prefix, str):
        raise ValueError("'output_prefix' must be a string when provided")

    return BenchmarkSpec(
        name=name,
        benchmark_type=benchmark_type,
        parameters=parameters,
        output_prefix=output_prefix,
    )
