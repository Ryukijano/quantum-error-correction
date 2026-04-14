#!/usr/bin/env bash

set -euo pipefail
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts/benchmarks}"
OUTPUT_PATH="${OUTPUT_PATH:-${ARTIFACT_DIR}/ci_contract_runtime.json}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

run_pytest() {
  echo "Running: python3 -m pytest -q $*"
  python3 -m pytest -q "$@"
}

echo "INFO: repo root is ${ROOT_DIR}"
mkdir -p "${ARTIFACT_DIR}"
cd "${ROOT_DIR}"

run_pytest tests/test_gym_env.py
run_pytest tests/test_sampling_backend_contracts.py tests/test_benchmark_decoder_contracts.py tests/test_runtime_contracts.py

python3 scripts/bench_runtime_contracts.py --output "${OUTPUT_PATH}"

python3 - "${OUTPUT_PATH}" <<'PY'
from pathlib import Path
import json
import sys

payload_path = Path(sys.argv[1])
payload = json.loads(payload_path.read_text(encoding="utf-8"))

required_payload_fields = (
    "event_count",
    "coalesced_count",
    "metric_count",
    "metric_profile_event_count",
    "event_kind_counts",
    "coalesced_kind_counts",
    "metric_profile_field_hits",
    "queue_drops",
    "done_events",
)
for field in required_payload_fields:
    if field not in payload:
        raise SystemExit(f"Missing payload field: {field}")

if payload["done_events"] != 1:
    raise SystemExit(f"done_events expected to be 1, got {payload['done_events']}")

if payload["event_count"] <= 0:
    raise SystemExit(f"event_count must be >0, got {payload['event_count']}")

if payload["metric_count"] <= 0:
    raise SystemExit(f"metric_count must be >0, got {payload['metric_count']}")

if payload["metric_profile_event_count"] != payload["metric_count"]:
    raise SystemExit(
        "metric_profile_event_count must equal metric_count, "
        f"got {payload['metric_profile_event_count']} != {payload['metric_count']}"
    )

metric_profile_fields = (
    "backend_id",
    "backend_enabled",
    "backend_version",
    "fallback_reason",
    "trace_id",
    "sample_trace_id",
    "backend_chain",
    "contract_flags",
    "profiler_flags",
    "sample_us",
    "sample_rate",
    "trace_tokens",
    "ler_ci",
)
metric_field_hits = payload["metric_profile_field_hits"]
for field in metric_profile_fields:
    actual = metric_field_hits.get(field)
    if actual != payload["metric_count"]:
        raise SystemExit(
            f"metric_profile_field_hits[{field!r}] expected {payload['metric_count']}, got {actual}"
        )

print("CI contract payload validation passed.")
print(f"runtime_json={payload_path}")
print(f"events={payload['event_count']} coalesced={payload['coalesced_count']} metrics={payload['metric_count']}")
PY
