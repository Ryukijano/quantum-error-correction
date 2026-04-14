Phase 1 Contract Map: Ising predecoder integration
=================================================

This document defines the exact contracts the Ising decoder adapter must satisfy before any code
changes are made in `surface_code_in_stem/decoders/`.

## 1) Decoder protocol contract (must preserve)

`surface_code_in_stem.decoders.base.DecoderProtocol` is the required public contract:

- `decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput`
- `BoolArray` expected as 2D bool array `(shots, num_detectors)`.
- Must return `DecoderOutput`:
  - `logical_predictions: bool ndarray, shape == (shots, num_observables)`
  - `decoder_name: str` (the adapter identity)
  - `diagnostics: Mapping[str, Any]`

Current built-ins also require:

- deterministic output for identical `(detector_events, metadata)` inputs
- hard validation:
  - `detector_events.ndim == 2`
  - `num_observables > 0`
  - output shape exactly matches expected shape

### Metadata contract currently used upstream

- `DecoderMetadata.num_observables` is required for sizing output.
- `DecoderMetadata.detector_error_model` is required for MWPM and parity/matching paths.
- `DecoderMetadata.extra` is available for optional context pass-through.
- `DecoderMetadata.seed` is propagated from environments and useful for deterministic tracing.

## 2) Current syndrome-net flow that must remain unchanged

### Decoder selection

1. UI / config creates a `decoder_name` (optional string).
2. Service/runners propagate `decoder_name` through:
   - `app/services/circuit_services.py` → `build_threshold_decoder()`
   - `app/services/rl_services.py` → `RLTrainingConfig.decoder_name`
   - `app/rl_runner.py` → `RLRunner(..., decoder_name=...)`
   - `app/rl_strategies.py` → `StrategyRuntime.decoder_name`
   - `_BaseTrainingStrategy.build_env(...)` → `EnvBuildContext(decoder_name=...)`
   - `QECGymEnv.__init__(decoder_name=...)` → `_resolve_baseline_decoder(decoder_name)`

3. Decoder lookup path:
   - `surface_code_in_stem.rl_control.gym_env._resolve_baseline_decoder`
   - `get_container().get_decoder(name)` inside `syndrome_net.container`.
4. Fallback behavior:
   - if lookup fails or adapter crashes at construction time, fallback to `MWPMDecoder`.

### Env info and metrics contract

- `QECGymEnv.reset()` currently sets info fields:
  - `baseline_decoder_requested`
  - `baseline_decoder`
  - `mwpm_prediction`
  - `mwpm_correct`
  - `binary_syndrome`
  - optional `mwpm_decode_error`
  - optional `sampling_backend` trace payload
- `sampled` info from env is later merged into RL metrics via:
  - `app/rl_strategies._sampling_trace_payload()`
  - `app/rl_strategies._append_sampling_fields()`
  - `_METADATA_FIELDS` in `app/services/rl_services.py`

`metric_payload_for_history` also merges recognized metadata fields from metric payload and top-level
event keys; therefore any new Ising diagnostics must be emitted both in metric payload and/or `info`
fields with names already tracked by `_METADATA_FIELDS` where possible.

## 3) Decoder registry & discoverability contract

`syndrome_net.container.DIContainer.register_defaults()`:

- imports default decoders directly: `MWPM`, `UnionFind`, `SparseBlossom`, `CudaQ`, `cuqnn`, `JAX`.
- registers discovered entry points in `"syndrome_net.decoders"`.
- adds fallback registration for any missing defaults by name (`default_decoders` map).

Plan for Ising integration:

- Add adapter import in `surface_code_in_stem/decoders/__init__.py` (with unavailable fallback wrapper pattern).
- Ensure `DIContainer.register_defaults()` always sees/retains existing defaults and includes `ising` either:
  - by direct default list registration, or
  - via plugin discovery if entry points are used.
- Keep existing defaults/lookup behavior stable; do not change existing default names.

## 4) Ising-Decoding contract alignment

### Core input/output shape contract (important)

From `Ising-Decoding/code/evaluation/logical_error_rate.py`:

- Raw detector events are `dets` shape `(B, 2*T*half)` where `half = (D^2 - 1) / 2`.
- `PreDecoderMemoryEvalModule.forward` returns `out` shape `(B, 1 + num_detectors)` where:
  - `out[:, 0]` = pre-decoder logical bit
  - `out[:, 1:]` = residual detector bits for global decoder
- Residual prediction shape is aligned to syndrome length for MWPM-like downstream usage.

For syndrome-net surface circuits, `c.num_detectors == 2 * rounds * ((distance ** 2 - 1) // 2)`, so mapping is
compatible if adapter uses identical `distance/rounds` interpretation.

### Inference model contract

- `code/model/predecoder.py` model:
  - input tensor to model: `(B, 4, T, D, D)` (`4` channels: `x_syn`, `z_syn`, `x_pres`, `z_pres`)
  - output tensor: `(B, out_channels, T, D, D)` where `out_channels` should be 4 for public heads.
- `_ensure_inference_io_channels` in Ising workflow:
  - sets `model.out_channels = 4` if unset
  - infers `model.input_channels` if unset
  - adjusts final `num_filters[-1]` to `out_channels`

### Model/spec contract to retain

- model family IDs currently in `code/model/registry.py` are `1..5`.
- all checkpoints are loaded through `ModelFactory` with `cfg.model.version` (`predecoder_memory_v1`).
- `_load_model` handles:
  - Safetensors via `PREDECODER_SAFETENSORS_CHECKPOINT`
  - direct `.pt` checkpoints and resolves DDP `"module."` prefixes.

## 5) Ising adapter acceptance criteria (Phase 1 target)

Acceptance (mandatory for contract audit):

1. **Signature compatibility**
   - adapter class is protocol-compatible (`name` + `decode(...) -> DecoderOutput`).
2. **Input coercion**
   - accepts any array-like detector payload, coerces to bool/uint8 as needed.
   - validates detector width matches expected `2 * rounds * ((distance^2-1)//2)`.
3. **Output shape**
   - returns `logical_predictions.shape == (shots, metadata.num_observables)`.
   - dtype `bool` with values `0/1` interpreted as logical predictions.
4. **Residual + MWPM path**
   - adapter computes residual syndromes first, then performs MWPM for final logical predictions.
   - keeps output semantics identical to existing env baseline path.
5. **Determinism**
   - identical `(detector_events, metadata)` yields identical output and diagnostics.
6. **Diagnostics contract**
   - include:
     - `backend` (adapter identity, likely `ising`),
     - `predecoder_backend` / `predecoder_available`,
     - `predecoder_fallback_reason` when applicable,
     - latency (`predecoder_latency_ms`, and keep existing `backend`/`sample_us` style fields),
     - fallback chain field if a secondary decoder is used.
7. **Fallback semantics**
   - failure paths degrade deterministically to existing fallback decoder path:
     - load/format/model errors,
     - dimension mismatch and unsupported config,
     - runtime inference errors.
   - diagnostics must indicate degraded path and reason.
8. **Compatibility with existing observability**
   - preserve existing `mwpm_*` and sampling trace payload in env info.
   - add Ising-specific fields additively (no removal/renaming).

## 6) Recommended Phase 1 implementation checkpoints

1. Register adapter + config surface (`ising`) with deterministic fallback defaults.
2. Implement `decoders` namespace import + availability fallback wrapper.
3. Add `EnvBuildContext` path coverage tests:
   - requested decoder name appears in `baseline_decoder_requested`,
   - missing decoder still falls back to `mwpm`.
4. Add shape/diagnostics tests for adapter:
   - wrong detector width fails fast,
   - valid conversion path returns correct shape and boolean dtype,
   - fallback path emits expected diagnostics and remains deterministic.
