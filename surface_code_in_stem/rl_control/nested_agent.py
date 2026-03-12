import torch
import torch.nn as nn
from typing import Any

from .titans import TitansMemory

class BaseAgent:
    """Base class for RL agents."""
    def __init__(self):
        pass
        
    def act(self, state: Any) -> Any:
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        raise NotImplementedError

class NestedLearningAgent(BaseAgent):
    """
    Nested Learning Agent for QEC.
    Prevents catastrophic forgetting by using a nested optimization paradigm:
    - Inner loop: fast updates to decoder weights.
    - Outer loop: slow updates to hyperparameters or architecture.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Use TITANS memory for long-term syndrome history
        self.memory_module = TitansMemory(input_dim=state_dim, hidden_dim=hidden_dim)
        
        # Inner loop model (e.g., fast decoder weights)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Optimizers for nested loops
        self.inner_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.outer_optimizer = torch.optim.Adam(self.memory_module.parameters(), lr=1e-4)
        
    def act(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action based on the current state and memory."""
        with torch.no_grad():
            context = self.memory_module(state)
            action_logits = self.policy_net(context)
            return torch.argmax(action_logits, dim=-1)
            
    def inner_loop(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> float:
        """
        Fast update: Update decoder weights (policy) based on recent experience.
        """
        self.inner_optimizer.zero_grad()
        
        # Ensure states are 3D for TITANS (batch, seq, dim) if needed, or 2D handled by wrapper
        if states.dim() == 2:
            states = states.unsqueeze(1)
            
        context = self.memory_module(states)
        action_logits = self.policy_net(context)
        
        # Simple policy gradient or cross entropy loss for demonstration
        criterion = nn.CrossEntropyLoss()
        loss = criterion(action_logits, actions) * rewards.mean() # Simplified loss
        
        loss.backward()
        self.inner_optimizer.step()
        
        return loss.item()
        
    def outer_loop(self, states: torch.Tensor, performance_metric: float) -> float:
        """
        Slow update: Update memory representations and hyperparameters.
        """
        self.outer_optimizer.zero_grad()
        
        if states.dim() == 2:
            states = states.unsqueeze(1)
            
        # The outer loop optimizes the memory module to improve overall performance
        # Here we use a surrogate loss for demonstration
        context = self.memory_module(states)
        loss = -torch.mean(context) * performance_metric # Encourage useful representations
        
        loss.backward()
        self.outer_optimizer.step()
        
        return loss.item()
        
    def store_experience(self, state: torch.Tensor, context_value: torch.Tensor):
        """Store experience in TITANS memory."""
        # In the recurrent formulation, memory is updated implicitly during forward passes
        # or via backprop through time. Explicit storage is less relevant for the
        # neural memory variant unless we implement a replay buffer.
        pass
