"""
SOTA DQN with Letter Decomposition Architecture.

This module implements the Letter Decomposition trick for Wordle:
- Instead of outputting ~13,000 Q-values (one per word), the network
  outputs 130 Q-values (26 letters × 5 positions).
- Word scores are computed as the sum of letter Q-values at each position.
- This reduces action space complexity by ~99% and enables generalization.

Reference: "Mastering Wordle with Reinforcement Learning" methodology.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Get the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LetterDecompositionDQN(nn.Module):
    """
    DQN with Letter Decomposition output layer.
    
    Architecture:
        Input (313) → FC(512) → ReLU → FC(256) → ReLU → FC(130)
    
    The 130 outputs represent Q-values for each (letter, position) pair:
        - Positions 0-25: Q(A-Z, position 0)
        - Positions 26-51: Q(A-Z, position 1)
        - ...
        - Positions 104-129: Q(A-Z, position 4)
    """
    
    NUM_LETTERS = 26
    NUM_POSITIONS = 5
    OUTPUT_DIM = NUM_LETTERS * NUM_POSITIONS  # 130
    
    def __init__(
        self,
        input_dim: int = 313,
        hidden_dims: Tuple[int, ...] = (512, 256),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Letter Decomposition DQN.
        
        Args:
            input_dim: Input observation dimension (default: 313)
            hidden_dims: Tuple of hidden layer dimensions
            device: Device to place the model on
        """
        super().__init__()
        
        self._device = device or get_device()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Output layer: 130 Q-values (26 letters × 5 positions)
        layers.append(nn.Linear(prev_dim, self.OUTPUT_DIM))
        
        self.network = nn.Sequential(*layers)
        self.to(self._device)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning letter-position Q-values.
        
        Args:
            obs: Observation tensor of shape (batch, 313)
            
        Returns:
            Q-values of shape (batch, 130) representing Q(letter, position)
        """
        return self.network(obs)
    
    def get_letter_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values reshaped as (batch, 5, 26) for position-letter lookup.
        
        Args:
            obs: Observation tensor of shape (batch, 313)
            
        Returns:
            Q-values of shape (batch, 5, 26)
        """
        q_flat = self.forward(obs)  # (batch, 130)
        return q_flat.view(-1, self.NUM_POSITIONS, self.NUM_LETTERS)  # (batch, 5, 26)
    
    @property
    def device(self) -> torch.device:
        return self._device


class WordScorer:
    """
    Utility class for computing word scores from letter Q-values.
    
    Uses precomputed letter matrices for efficient batch scoring.
    """
    
    def __init__(
        self,
        word_list: List[str],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the word scorer.
        
        Args:
            word_list: List of valid words
            device: Device for tensors
        """
        self.word_list = word_list
        self.device = device or get_device()
        
        # Precompute word-letter matrix: (num_words, 5, 26)
        self.word_matrix = self._build_word_matrix()
    
    def _build_word_matrix(self) -> torch.Tensor:
        """Build tensor mapping words to letter positions."""
        num_words = len(self.word_list)
        matrix = torch.zeros(num_words, 5, 26, device=self.device)
        
        for i, word in enumerate(self.word_list):
            for pos, char in enumerate(word.lower()):
                letter_idx = ord(char) - ord('a')
                matrix[i, pos, letter_idx] = 1.0
        
        return matrix
    
    def compute_word_scores(
        self,
        letter_q_values: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scores for all words based on letter Q-values.
        
        The score for a word is the sum of Q-values for its letters:
            score(word) = Σ Q(letter, position) for each (letter, pos) in word
        
        Args:
            letter_q_values: Q-values of shape (batch, 5, 26)
            action_mask: Optional mask of shape (batch, num_words)
            
        Returns:
            Word scores of shape (batch, num_words)
        """
        batch_size = letter_q_values.shape[0]
        
        # Element-wise multiply and sum: (batch, 5, 26) * (num_words, 5, 26) → sum
        # Expand letter_q_values: (batch, 1, 5, 26)
        # word_matrix: (num_words, 5, 26)
        q_expanded = letter_q_values.unsqueeze(1)  # (batch, 1, 5, 26)
        
        # Compute dot product for each word
        # (batch, 1, 5, 26) * (num_words, 5, 26) → (batch, num_words, 5, 26) → sum
        scores = (q_expanded * self.word_matrix.unsqueeze(0)).sum(dim=(2, 3))  # (batch, num_words)
        
        # Apply action mask (set invalid words to very negative value)
        if action_mask is not None:
            scores = scores.masked_fill(action_mask == 0, -1e9)
        
        return scores
    
    def select_best_word(
        self,
        letter_q_values: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Select the best word for each batch element.
        
        Args:
            letter_q_values: Q-values of shape (batch, 5, 26)
            action_mask: Optional mask of shape (batch, num_words)
            
        Returns:
            Tuple of (action_indices, words)
        """
        scores = self.compute_word_scores(letter_q_values, action_mask)
        best_indices = scores.argmax(dim=1)  # (batch,)
        
        words = [self.word_list[idx.item()] for idx in best_indices]
        
        return best_indices, words


class SOTADQNAgent:
    """
    Complete SOTA DQN Agent with Letter Decomposition.
    
    Combines the neural network with word scoring and action selection.
    """
    
    def __init__(
        self,
        word_list: List[str],
        input_dim: int = 313,
        hidden_dims: Tuple[int, ...] = (512, 256),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the SOTA DQN Agent.
        
        Args:
            word_list: List of valid words (actions)
            input_dim: Observation dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            device: Torch device
        """
        self.device = device or get_device()
        self.gamma = gamma
        
        # Networks
        self.policy_net = LetterDecompositionDQN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.target_net = LetterDecompositionDQN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Word scorer
        self.word_scorer = WordScorer(word_list, device=self.device)
        self.word_list = word_list
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )
    
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        epsilon: float = 0.0,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            obs: Observation array of shape (313,)
            action_mask: Optional binary mask for valid actions
            epsilon: Exploration rate
            
        Returns:
            Selected action index
        """
        # Random exploration
        if np.random.random() < epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                return np.random.choice(valid_actions)
            return np.random.randint(len(self.word_list))
        
        # Greedy action
        self.policy_net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            letter_q = self.policy_net.get_letter_q_values(obs_t)  # (1, 5, 26)
            
            mask_t = None
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            best_idx, _ = self.word_scorer.select_best_word(letter_q, mask_t)
        
        return best_idx[0].item()
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DQN loss using Letter Decomposition.
        
        The loss is computed on letter-position Q-values, not word Q-values.
        We compute the "pseudo Q-value" for an action as the sum of letter Qs.
        
        Args:
            states: Batch of states (batch, 313)
            actions: Batch of actions (batch,) - word indices
            rewards: Batch of rewards (batch,)
            next_states: Batch of next states (batch, 313)
            dones: Batch of done flags (batch,)
            next_action_masks: Optional masks for next state actions
            
        Returns:
            Scalar loss value
        """
        batch_size = states.shape[0]
        
        # Get letter Q-values for current states
        letter_q = self.policy_net.get_letter_q_values(states)  # (batch, 5, 26)
        
        # Compute current Q-values: sum of letter Qs for each action
        # Get the word matrix for the taken actions
        action_word_matrix = self.word_scorer.word_matrix[actions]  # (batch, 5, 26)
        current_q = (letter_q * action_word_matrix).sum(dim=(1, 2))  # (batch,)
        
        # Compute target Q-values
        with torch.no_grad():
            next_letter_q = self.target_net.get_letter_q_values(next_states)  # (batch, 5, 26)
            next_word_scores = self.word_scorer.compute_word_scores(
                next_letter_q, next_action_masks
            )  # (batch, num_words)
            next_q_max = next_word_scores.max(dim=1)[0]  # (batch,)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q_max
        
        # MSE loss
        loss = F.mse_loss(current_q, target_q)
        
        return loss
    
    def update_target_network(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
