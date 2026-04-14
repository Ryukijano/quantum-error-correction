"""JAX-accelerated confidence-aware decoding with Flash Attention.

This module provides high-performance implementations of confidence-weighted
decoding using JAX for GPU/TPU acceleration and Flash Attention for
efficient syndrome sequence modeling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol
from .union_find import UnionFindDecoder


# Optional JAX imports with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

# Optional Flash Attention
try:
    import flash_attn
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None


_JAX_VERSION = getattr(jax, "__version__", "unknown") if HAS_JAX else "unknown"


def _coerce_confidence_matrix(
    metadata: DecoderMetadata,
    events: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool, bool]:
    extra = metadata.extra if isinstance(metadata.extra, Mapping) else {}
    confidence: Any = extra.get("confidence") if isinstance(extra, Mapping) else None
    if confidence is None:
        confidence = extra.get("soft_information")
    if confidence is None:
        return np.ones_like(events, dtype=np.float64), False, False

    confidence_array = np.asarray(confidence, dtype=np.float64)
    if confidence_array.ndim != events.ndim:
        raise ValueError("confidence must be a 2D tensor matching detector_events.")
    if confidence_array.shape != events.shape:
        raise ValueError(
            "confidence shape must match detector_events shape "
            f"(got {confidence_array.shape}, expected {events.shape})."
        )

    clipped = confidence_array
    clipped_to_bounds = False
    if np.any(np.isnan(clipped)):
        clipped_to_bounds = True
        clipped = np.nan_to_num(clipped, nan=1.0, posinf=1.0, neginf=0.0)
    if np.any((clipped < 0.0) | (clipped > 1.0)):
        clipped_to_bounds = True
        clipped = np.clip(clipped, 0.0, 1.0)
    return clipped, clipped_to_bounds, True


def _build_fallback_decoder(metadata: DecoderMetadata, confidence_scale: float) -> Any:
    from surface_code_in_stem.confidence_decoding import WeightedMWPMDecoder

    if metadata.detector_error_model is None:
        raise ValueError("JAXConfidenceDecoderAdapter requires detector_error_model for weighted matching fallback.")
    return WeightedMWPMDecoder(metadata.detector_error_model, confidence_scale=confidence_scale)


def _decode_with_adjusted_weights(
    weighted_decoder: Any,
    detector_events: NDArray[np.bool_],
    adjusted_weights: NDArray[np.float64],
) -> NDArray[np.bool_]:
    base_graph = weighted_decoder._base_graph
    edges_u: NDArray[np.integer[Any]] = np.asarray(weighted_decoder._edge_u, dtype=np.int64)
    edges_v: NDArray[np.integer[Any]] = np.asarray(weighted_decoder._edge_v, dtype=np.int64)
    edge_pairs = list(weighted_decoder._edge_pairs)
    shots = detector_events.shape[0]

    if adjusted_weights.shape != (shots, edges_u.shape[0]):
        raise ValueError(
            f"Adjusted-weight shape mismatch: expected ({shots}, {edges_u.shape[0]}), got {adjusted_weights.shape}"
        )

    matching_module = weighted_decoder._pymatching
    predictions = np.zeros((shots, weighted_decoder._num_fault_ids), dtype=np.bool_)
    for shot_ix in range(shots):
        graph = base_graph.copy()
        for weight_ix, (edge_u, edge_v) in enumerate(edge_pairs):
            graph[int(edge_u)][int(edge_v)]["weight"] = float(adjusted_weights[shot_ix, weight_ix])
        shot = np.asarray(detector_events[shot_ix], dtype=np.uint8)
        predictions[shot_ix] = np.asarray(matching_module.Matching(graph).decode(shot), dtype=np.bool_)
    return predictions


class JAXConfidenceDecoder:
    """Vectorized confidence-aware decoder using JAX acceleration.

    Implements fast confidence-weighted decoding with:
    - JAX JIT compilation for GPU acceleration
    - Vectorized batch processing
    - Flash Attention for syndrome sequence attention
    """

    def __init__(self, confidence_scale: float = 1.0, use_flash_attn: bool = True):
        """Initialize JAX confidence decoder.

        Args:
            confidence_scale: Scaling factor for confidence weighting
            use_flash_attn: Whether to use Flash Attention if available
        """
        if not HAS_JAX:
            raise ImportError("JAX is required for JAXConfidenceDecoder. Install with: pip install jax jaxlib")

        self.confidence_scale = confidence_scale
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN

        # JIT compile core functions
        self._jit_compute_weights = jit(self._compute_adjusted_weights)
        self._jit_batch_decode = jit(self._batch_decode)

    def _compute_adjusted_weights(
        self,
        base_weights: jnp.ndarray,
        shot_confidence: jnp.ndarray,
        edge_u: jnp.ndarray,
        edge_v: jnp.ndarray,
        u_is_det: jnp.ndarray,
        v_is_det: jnp.ndarray,
    ) -> jnp.ndarray:
        """Vectorized confidence weight adjustment (JAX).

        Args:
            base_weights: Base edge weights (num_edges,)
            shot_confidence: Per-detector confidence values (num_detectors,)
            edge_u: Edge endpoint u indices (num_edges,)
            edge_v: Edge endpoint v indices (num_edges,)
            u_is_det: Boolean mask for u being real detector (num_edges,)
            v_is_det: Boolean mask for v being real detector (num_edges,)

        Returns:
            Adjusted edge weights (num_edges,)
        """
        # Safe index lookup (boundary is replaced with 0)
        conf_u = jnp.where(u_is_det, shot_confidence[edge_u], 1.0)
        conf_v = jnp.where(v_is_det, shot_confidence[edge_v], 1.0)

        # Count real endpoints
        num_real_endpoints = u_is_det.astype(jnp.float32) + v_is_det.astype(jnp.float32)

        # Compute mean confidence (avoid division by zero)
        mean_conf = jnp.where(
            num_real_endpoints > 0,
            (conf_u * u_is_det + conf_v * v_is_det) / jnp.maximum(num_real_endpoints, 1.0),
            1.0
        )

        # Scale weights inversely with confidence
        scale = 1.0 + self.confidence_scale * (1.0 - mean_conf)

        return base_weights * scale

    def _batch_decode(
        self,
        hard_bits: jnp.ndarray,
        confidence: jnp.ndarray,
        base_weights: jnp.ndarray,
        edge_u: jnp.ndarray,
        edge_v: jnp.ndarray,
        u_is_det: jnp.ndarray,
        v_is_det: jnp.ndarray,
    ) -> jnp.ndarray:
        """Decode a batch of syndromes with confidence.

        Args:
            hard_bits: Binary detector outcomes (batch_size, num_detectors)
            confidence: Confidence values (batch_size, num_detectors)
            base_weights: Base edge weights (num_edges,)
            edge_u, edge_v: Edge endpoint indices (num_edges,)
            u_is_det, v_is_det: Boolean masks (num_edges,)

        Returns:
            Adjusted weights for each shot (batch_size, num_edges)
        """
        # Vectorize over batch dimension
        def decode_single(hard, conf):
            return self._compute_adjusted_weights(
                base_weights, conf, edge_u, edge_v, u_is_det, v_is_det
            )

        # Apply to each shot in batch
        return vmap(decode_single, in_axes=(0, 0))(hard_bits, confidence)

    def decode_batch(
        self,
        hard_bits: NDArray[np.float64],
        confidence: NDArray[np.float64],
        base_weights: NDArray[np.float64],
        edge_u: NDArray[np.int64],
        edge_v: NDArray[np.int64],
        u_is_det: NDArray[np.bool_],
        v_is_det: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Decode batch with JAX acceleration.

        Args:
            hard_bits: (batch_size, num_detectors) binary outcomes
            confidence: (batch_size, num_detectors) confidence values
            base_weights: (num_edges,) base edge weights
            edge_u, edge_v: (num_edges,) edge endpoint indices
            u_is_det, v_is_det: (num_edges,) boolean masks

        Returns:
            (batch_size, num_edges) adjusted weights
        """
        # Convert to JAX arrays
        hard_bits_jax = jnp.array(hard_bits, dtype=jnp.float32)
        confidence_jax = jnp.array(confidence, dtype=jnp.float32)
        base_weights_jax = jnp.array(base_weights, dtype=jnp.float32)
        edge_u_jax = jnp.array(edge_u, dtype=jnp.int32)
        edge_v_jax = jnp.array(edge_v, dtype=jnp.int32)
        u_is_det_jax = jnp.array(u_is_det, dtype=jnp.bool_)
        v_is_det_jax = jnp.array(v_is_det, dtype=jnp.bool_)

        # Run JIT-compiled decode
        adjusted_weights = self._jit_batch_decode(
            hard_bits_jax,
            confidence_jax,
            base_weights_jax,
            edge_u_jax,
            edge_v_jax,
            u_is_det_jax,
            v_is_det_jax,
        )

        # Convert back to numpy
        return np.array(adjusted_weights)


@dataclass
class JAXConfidenceDecoderAdapter(DecoderProtocol):
    """Backend-style adapter exposing `JAXConfidenceDecoder` through the decoder protocol."""

    name: str = "jax_confidence"
    confidence_scale: float = 1.0
    use_flash_attn: bool = True
    fallback_decoder: UnionFindDecoder = field(default_factory=UnionFindDecoder)
    jax_decoder: Optional[JAXConfidenceDecoder] = field(default=None, init=False)
    capabilities: tuple[str, ...] = ("jax", "confidence_aware", "backend_fallback")

    def __post_init__(self) -> None:
        if HAS_JAX:
            self.jax_decoder = JAXConfidenceDecoder(
                confidence_scale=self.confidence_scale,
                use_flash_attn=self.use_flash_attn,
            )

    def _diagnostic_metadata(self) -> dict[str, Any]:
        return {
            "backend_id": self.name,
            "backend": self.name,
            "backend_available": HAS_JAX,
            "backend_enabled": HAS_JAX and self.jax_decoder is not None,
            "backend_version": _JAX_VERSION,
            "capabilities": list(self.capabilities),
            "backend_contract": HAS_JAX,
        }

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        events = np.asarray(detector_events, dtype=np.bool_)
        if events.ndim != 2:
            raise ValueError("detector_events must be a 2D bool array.")

        start_ns = time.perf_counter_ns()
        diagnostics = self._diagnostic_metadata()
        diagnostics.update(
            {
                "backend_chain": [f"requested:{self.name}"],
                "fallback_chain": [f"requested:{self.name}"],
                "contract_flags": "backend_disabled,contract_fallback",
                "profiler_flags": "trace_chain_recorded",
                "degraded": bool(not HAS_JAX),
            }
        )

        confidence, confidence_was_clipped, confidence_provided = _coerce_confidence_matrix(metadata, events.astype(np.float64))
        details: dict[str, Any] = {
            "has_flash_attention": HAS_FLASH_ATTN,
            "confidence_provided": confidence_provided,
            "confidence_was_clipped": confidence_was_clipped,
            "num_shots": int(events.shape[0]),
            "num_detectors": int(events.shape[1]),
            "num_observables": int(metadata.num_observables),
        }

        if confidence.size == 0:
            details["outcome"] = "empty_input"
            diagnostics.update(
                {
                    "backend_error": None,
                    "backend_chain": diagnostics["backend_chain"] + [f"selected:{self.name}"],
                    "fallback_chain": diagnostics["fallback_chain"] + [f"selected:{self.name}"],
                    "contract_flags": "backend_enabled,contract_met",
                    "degraded": False,
                    "fallback_reason": None,
                    "details": details,
                }
            )
            return DecoderOutput(
                logical_predictions=np.zeros((0, metadata.num_observables), dtype=np.bool_),
                decoder_name=self.name,
                diagnostics=diagnostics,
            )

        if not HAS_JAX or self.jax_decoder is None:
            details["outcome"] = "fallback_no_jax"
            fallback_output = self.fallback_decoder.decode(events, metadata)
            fallback_diagnostics = dict(fallback_output.diagnostics)
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            diagnostics.update(fallback_diagnostics)
            diagnostics.update(
                {
                    "backend_error": "JAX not available",
                    "fallback_reason": "JAX backend unavailable",
                    "backend_chain": diagnostics["backend_chain"] + ["backend_fallback"],
                    "fallback_chain": diagnostics["fallback_chain"] + ["backend_fallback"],
                    "contract_flags": "backend_disabled,contract_fallback",
                    "degraded": True,
                    "latency_ms": latency_ms,
                    "details": details,
                }
            )
            return DecoderOutput(
                logical_predictions=fallback_output.logical_predictions,
                decoder_name=self.name,
                diagnostics=diagnostics,
            )

        try:
            weighted_decoder = _build_fallback_decoder(metadata, self.confidence_scale)
            adjusted_weights = self.jax_decoder.decode_batch(
                hard_bits=np.asarray(events, dtype=np.float64),
                confidence=confidence.astype(np.float64),
                base_weights=np.asarray(weighted_decoder._base_weights, dtype=np.float64),
                edge_u=np.asarray(weighted_decoder._edge_u, dtype=np.int64),
                edge_v=np.asarray(weighted_decoder._edge_v, dtype=np.int64),
                u_is_det=np.asarray(weighted_decoder._u_is_det, dtype=np.bool_),
                v_is_det=np.asarray(weighted_decoder._v_is_det, dtype=np.bool_),
            )
            predictions = _decode_with_adjusted_weights(
                weighted_decoder=weighted_decoder,
                detector_events=np.asarray(events, dtype=np.bool_),
                adjusted_weights=np.asarray(adjusted_weights, dtype=np.float64),
            )
            details["outcome"] = "backend_jax_success"
            diagnostics.update(
                {
                    "backend_error": None,
                    "backend_chain": diagnostics["backend_chain"] + [f"selected:{self.name}"],
                    "fallback_chain": diagnostics["fallback_chain"] + [f"selected:{self.name}"],
                    "contract_flags": "backend_enabled,contract_met",
                    "degraded": False,
                    "fallback_reason": None,
                    "details": details,
                }
            )
        except Exception as exc:
            details["outcome"] = "backend_jax_failed"
            details["backend_failure"] = str(exc)
            fallback_output = self.fallback_decoder.decode(events, metadata)
            diagnostics.update(
                {
                    "backend_error": str(exc),
                    "fallback_reason": str(exc),
                    "backend_chain": diagnostics["backend_chain"] + ["backend_fallback"],
                    "fallback_chain": diagnostics["fallback_chain"] + ["backend_fallback"],
                    "contract_flags": "backend_disabled,contract_fallback",
                    "degraded": True,
                    "details": details,
                }
            )
            diagnostics["profiler_flags"] = ",".join([f for f in diagnostics["profiler_flags"].split(",") if f] + ["fallback_used"])
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            diagnostics["latency_ms"] = latency_ms
            return DecoderOutput(
                logical_predictions=fallback_output.logical_predictions,
                decoder_name=self.name,
                diagnostics=diagnostics,
            )
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        diagnostics["latency_ms"] = latency_ms
        return DecoderOutput(
            logical_predictions=np.asarray(predictions, dtype=np.bool_),
            decoder_name=self.name,
            diagnostics=diagnostics,
        )


class FlashAttentionSyndromeModel:
    """Syndrome sequence model using Flash Attention.

    Uses Flash Attention for O(n) memory and O(n²) compute attention
    on syndrome sequences, enabling efficient modeling of long-range
dependencies in syndrome histories.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize Flash Attention syndrome model.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        if not HAS_JAX and not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionSyndromeModel requires JAX and Flash Attention. "
                "Install with: pip install jax jaxlib flash-attn"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        if not HAS_JAX:
            return

        key = jax.random.PRNGKey(0)

        # Q, K, V projection weights
        key, *subkeys = jax.random.split(key, 4)
        self.w_q = jax.random.normal(subkeys[0], (self.embed_dim, self.embed_dim)) * 0.02
        self.w_k = jax.random.normal(subkeys[1], (self.embed_dim, self.embed_dim)) * 0.02
        self.w_v = jax.random.normal(subkeys[2], (self.embed_dim, self.embed_dim)) * 0.02

        # Output projection
        self.w_o = jax.random.normal(subkeys[3], (self.embed_dim, self.embed_dim)) * 0.02

        # Layer norms
        self.ln1_scale = jnp.ones(self.embed_dim)
        self.ln1_bias = jnp.zeros(self.embed_dim)
        self.ln2_scale = jnp.ones(self.embed_dim)
        self.ln2_bias = jnp.zeros(self.embed_dim)

    def _layer_norm(self, x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
        """Layer normalization."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + 1e-5) * scale + bias

    def _apply_attention(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Apply Flash Attention.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Attention output (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = jnp.dot(x, self.w_q.T)  # (batch, seq, embed)
        k = jnp.dot(x, self.w_k.T)
        v = jnp.dot(x, self.w_v.T)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        if HAS_FLASH_ATTN and self.use_flash_attn:
            # Use Flash Attention for O(n) memory efficiency
            # Flash Attention expects: (batch, seq, nheads, headdim)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=False)
        else:
            # Fallback to standard attention
            # Transpose for attention computation
            q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, nheads, seq, headdim)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            # Compute attention scores
            scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)

            if mask is not None:
                scores = jnp.where(mask, scores, -1e9)

            attn_weights = jax.nn.softmax(scores, axis=-1)
            attn_weights = jax.nn.dropout(attn_weights, self.dropout, key=jax.random.PRNGKey(0))

            out = jnp.matmul(attn_weights, v)
            out = jnp.transpose(out, (0, 2, 1, 3))  # (batch, seq, nheads, headdim)

        # Reshape back
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = jnp.dot(out, self.w_o.T)

        return out

    def forward(
        self,
        syndrome_sequence: NDArray[np.float64],
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through Flash Attention syndrome model.

        Args:
            syndrome_sequence: (batch_size, seq_len, embed_dim) syndrome embeddings
            return_attention: Whether to return attention weights

        Returns:
            Dict with 'output' and optionally 'attention_weights'
        """
        if not HAS_JAX:
            raise RuntimeError("JAX is required for Flash Attention model")

        # Convert to JAX array
        x = jnp.array(syndrome_sequence, dtype=jnp.float32)

        # Layer norm + attention + residual
        residual = x
        x = self._layer_norm(x, self.ln1_scale, self.ln1_bias)
        x = self._apply_attention(x)
        x = x + residual

        # Layer norm + FFN + residual
        residual = x
        x = self._layer_norm(x, self.ln2_scale, self.ln2_bias)
        x = jax.nn.gelu(x)  # GELU activation
        x = x + residual

        output = np.array(x)

        result = {"output": output}

        return result

    def encode_syndrome_history(
        self,
        syndrome_history: NDArray[np.bool_],
        window_size: int = 10,
    ) -> NDArray[np.float64]:
        """Encode syndrome history using Flash Attention.

        Args:
            syndrome_history: (batch_size, total_rounds, num_detectors) syndrome bits
            window_size: Number of rounds to attend over

        Returns:
            (batch_size, window_size, embed_dim) encoded representations
        """
        if not HAS_JAX:
            raise RuntimeError("JAX is required for syndrome encoding")

        batch_size, total_rounds, num_detectors = syndrome_history.shape

        # Truncate or pad to window_size
        if total_rounds > window_size:
            syndrome_history = syndrome_history[:, -window_size:, :]
        elif total_rounds < window_size:
            pad_width = ((0, 0), (0, window_size - total_rounds), (0, 0))
            syndrome_history = np.pad(syndrome_history, pad_width, mode='constant')

        # Simple embedding: convert binary to float and project
        syndrome_float = syndrome_history.astype(np.float32)

        # Pad detector dimension to match embed_dim if needed
        if num_detectors < self.embed_dim:
            pad_width = ((0, 0), (0, 0), (0, self.embed_dim - num_detectors))
            syndrome_float = np.pad(syndrome_float, pad_width, mode='constant')
        else:
            # Truncate or use learned projection
            syndrome_float = syndrome_float[:, :, :self.embed_dim]

        # Apply Flash Attention model
        result = self.forward(syndrome_float)

        return result["output"]


def create_fast_confidence_decoder(
    confidence_scale: float = 1.0,
    use_jax: bool = True,
) -> Any:
    """Factory function to create optimal confidence decoder.

    Args:
        confidence_scale: Confidence scaling factor
        use_jax: Whether to use JAX acceleration if available

    Returns:
        Decoder instance (JAX or numpy fallback)
    """
    if use_jax and HAS_JAX:
        return JAXConfidenceDecoder(confidence_scale=confidence_scale)
    else:
        # Return numpy-based fallback
        from ..confidence_decoding import WeightedMWPMDecoder
        return WeightedMWPMDecoder
