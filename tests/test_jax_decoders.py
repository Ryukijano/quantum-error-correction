import numpy as np
import pytest

pytest.importorskip("stim")

from codes.qldpc.parity_builder import toric_code_parity

def test_jax_neural_bp_decoder():
    """Test the JAX Neural BP decoder if JAX is available."""
    try:
        import jax
        from surface_code_in_stem.decoders.jax_gnn_decoder import JAXNeuralBPDecoder
    except ImportError:
        pytest.skip("JAX not installed")
        
    distance = 3
    hx, hz = toric_code_parity(distance)
    num_checks = hx.shape[0]
    num_vars = hx.shape[1]
    
    decoder = JAXNeuralBPDecoder(num_vars=num_vars, num_checks=num_checks, max_iter=3)
    
    batch_size = 4
    syndrome = np.random.randint(0, 2, (batch_size, num_checks)).astype(np.float32)
    
    probs = decoder.decode_batch(decoder.init_params, syndrome, hx)
    
    assert probs.shape == (batch_size, num_vars)
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"
