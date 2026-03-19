import torch
import numpy as np
import pytest

pytest.importorskip("stim")

from surface_code_in_stem.decoders.gnn_decoder import NeuralBPDecoder, qLDPCGNNDecoder
from codes.qldpc.parity_builder import toric_code_parity

def test_neural_bp_decoder_forward():
    """Test the Neural BP decoder can process syndromes without crashing."""
    distance = 3
    # Build a tiny toric code parity matrix for testing
    hx, hz = toric_code_parity(distance)
    
    # We'll just test the X-syndrome decoding (Z errors)
    num_checks = hx.shape[0]
    num_vars = hx.shape[1]
    
    decoder = NeuralBPDecoder(num_vars=num_vars, num_checks=num_checks, max_iter=3)
    
    # Create a batch of mock syndromes
    batch_size = 4
    syndrome = torch.randint(0, 2, (batch_size, num_checks), dtype=torch.float32)
    
    parity_tensor = torch.FloatTensor(hx).to_sparse()
    
    probs = decoder(syndrome, parity_tensor)
    
    assert probs.shape == (batch_size, num_vars)
    assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"


def test_gnn_decoder_forward():
    """Test the GNN decoder runs a forward pass correctly."""
    distance = 3
    hx, hz = toric_code_parity(distance)
    
    num_checks = hx.shape[0]
    num_vars = hx.shape[1]
    
    # Small hidden dim for fast test
    decoder = qLDPCGNNDecoder(num_vars=num_vars, num_checks=num_checks, hidden_dim=16, num_layers=2)
    
    batch_size = 2
    syndrome = torch.randint(0, 2, (batch_size, num_checks))
    
    parity_tensor = torch.FloatTensor(hx).to_sparse()
    
    logits = decoder(syndrome, parity_tensor)
    
    assert logits.shape == (batch_size, num_vars)
    
