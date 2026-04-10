# Benchmarking and Runtime Contract Reference

This document defines the minimum expected metadata fields for benchmark and runtime streams used by CI and regression checks.

## 1. Common runtime event fields (`rl_services` / Streamlit history rows)

Canonical keys emitted by the training loop:

| Field | Type | Meaning |
|---|---|---|
| `backend_id` | str | Selected backend identifier for a sample call. |
| `backend_enabled` | bool | Whether selected backend was successfully enabled. |
| `backend_chain` | str | Fallback chain summary text (eg. `requested:stim->selected:stim`). |
| `backend_chain_tokens` | list[str] | Ordered tokenized chain path in memory; serialized for CSV/JSON output. |
| `contract_flags` | str | Comma list of contract outcomes (eg. `backend_enabled,contract_met`). |
| `profiler_flags` | str | Comma list of profiler outcomes (eg. `sample_trace_present,trace_chain_recorded`). |
| `fallback_reason` | str \| null | Fallback explanation when fallback path is used. |
| `sample_us` | float | Sample latency in microseconds. |
| `sample_rate` | float | Sample rate estimate (samples/sec). |
| `sample_trace_id` | str \| null | Trace identifier for payload correlation. |
| `trace_tokens` | list[str] \| null | UI-provided tokens for trace correlation. |
| `trace_id` | str \| null | Optional unique trace key for the sample event. |
| `details` | any | Optional backend/routing details payload. |
| `ler_ci` | any | Optional RL statistical confidence payload. |

## 2. Decoder benchmark row fields (`BenchmarkRow`)

Canonical fields emitted by `scripts/benchmark_decoders.py`:

| Field | Type | Meaning |
|---|---|---|
| `domain` | str | Benchmark family (`circuit`, `qldpc`, ...). |
| `family` | str | Family name used in the run. |
| `decoder` | str | Decoder identifier. |
| `distance` | int | Code distance or size. |
| `physical_error_rate` | float | Physical error rate for run. |
| `shots` | int | Number of samples/shots. |
| `metric_name` | str | Name of measured value. |
| `metric_value` | float | Scalar metric value. |
| `backend` | str | Backend tag used in this row. |
| `backend_enabled` | bool | Whether backend was active or fallback path used. |
| `backend_version` | str \| null | Backend version/identity string. |
| `fallback_reason` | str \| null | Fallback explanation if backend unavailable. |
| `sample_trace_id` | str \| null | Trace identifier for reproducibility checks. |
| `backend_chain` | str \| null | Fallback chain path summary. |
| `backend_chain_tokens` | list[str] \| null | Ordered chain tokens for compatibility checks. |
| `contract_flags` | str \| null | Backend contract status flags. |
| `profiler_flags` | str \| null | Backend profiler status flags. |

## 3. Serialization note

- Keep list-shaped fields in memory (`backend_chain_tokens` as `list[str]`).
- Serialize list fields at output boundary (`json.dumps`) for:
  - CSV rows
  - JSON artifact persistence where a string payload is expected by downstream tools.

## 4. CI assertions

Recommended assertions:

1. `backend_chain_tokens` exists and contains a non-empty sequence when backend probing is enabled.
2. `contract_flags` and `profiler_flags` are non-empty when payload instrumentation is enabled.
3. `sample_trace_id` is present for rows where `backend_chain` indicates tracing.
4. fallback ratio and trace coverage remain deterministic for a fixed seed.
