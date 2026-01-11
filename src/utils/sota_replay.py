"""
Experience Replay Buffer for SOTA DQN Training.

Provides a simple, efficient replay buffer implementation for
the Letter Decomposition DQN with support for action masks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Transition:
    """A single transition in the replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: Optional[np.ndarray] = None
    next_action_mask: Optional[np.ndarray] = None


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    
    Stores transitions and provides random batches for training.
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        obs_dim: int = 313,
        action_space_size: Optional[int] = None,
    ):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Observation dimension
            action_space_size: Action space size (for mask storage)
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_space_size = action_space_size
        
        # Preallocate arrays for efficiency
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Optional action masks
        if action_space_size is not None:
            self.action_masks = np.zeros((capacity, action_space_size), dtype=np.float32)
            self.next_action_masks = np.zeros((capacity, action_space_size), dtype=np.float32)
        else:
            self.action_masks = None
            self.next_action_masks = None
        
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Add a transition to the buffer."""
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        if self.action_masks is not None and action_mask is not None:
            self.action_masks[idx] = action_mask
        if self.next_action_masks is not None and next_action_mask is not None:
            self.next_action_masks[idx] = next_action_mask
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
               Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, 
                      action_masks, next_action_masks)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        masks = self.action_masks[indices] if self.action_masks is not None else None
        next_masks = self.next_action_masks[indices] if self.next_action_masks is not None else None
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            masks,
            next_masks,
        )
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size
