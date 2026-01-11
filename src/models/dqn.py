"""
Deep Q-Network (DQN) implementation with optional Dueling architecture.

This module provides a modular DQN model compatible with Apple Silicon MPS
acceleration for training Wordle agents.
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch tensors.
    
    Returns:
        torch.device: MPS if available (Apple Silicon), CUDA if available, else CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class DQNEncoder(nn.Module):
    """
    Encoder network for processing Wordle observations.
    
    FIXED: Processes the board (6, 5, 30), alphabet (26,), and turn (1,)
    observations into a fixed-size feature vector.
    """
    
    def __init__(
        self,
        board_shape: Tuple[int, int, int] = (6, 5, 30),  # FIXED: 30 = 26 letters + 4 color
        alphabet_size: int = 26,
        turn_size: int = 1,  # NEW: turn counter
        hidden_dim: int = 256,
        encoder_type: Literal["mlp", "cnn"] = "mlp",
    ) -> None:
        """
        Initialize the encoder.
        
        Args:
            board_shape: Shape of the board observation (turns, positions, features)
                         features = 26 letter one-hot + 4 color state
            alphabet_size: Size of the alphabet status vector
            turn_size: Size of turn indicator (1,)
            hidden_dim: Hidden dimension size
            encoder_type: Type of encoder ('mlp' or 'cnn')
        """
        super().__init__()
        
        self.board_shape = board_shape
        self.alphabet_size = alphabet_size
        self.turn_size = turn_size
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        
        board_flat_size = board_shape[0] * board_shape[1] * board_shape[2]
        
        if encoder_type == "mlp":
            self.board_encoder = nn.Sequential(
                nn.Linear(board_flat_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:  # CNN
            # Treat board as (batch, 6, 5*30) for 1D convolution over turns
            self.board_encoder = nn.Sequential(
                nn.Conv1d(board_shape[1] * board_shape[2], 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * board_shape[0], hidden_dim),
                nn.ReLU(),
            )
        
        self.alphabet_encoder = nn.Sequential(
            nn.Linear(alphabet_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # NEW: Turn encoder
        self.turn_encoder = nn.Sequential(
            nn.Linear(turn_size, 16),
            nn.ReLU(),
        )
        
        # Combined feature dimension: board + alphabet + turn
        self.feature_dim = hidden_dim + 64 + 16
        
        self.combiner = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self, 
        board: torch.Tensor, 
        alphabet: torch.Tensor,
        turn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            board: Board observation tensor of shape (batch, 6, 5, 30)
            alphabet: Alphabet status tensor of shape (batch, 26)
            turn: Turn counter tensor of shape (batch, 1)
            
        Returns:
            Feature tensor of shape (batch, hidden_dim)
        """
        batch_size = board.size(0)
        
        if self.encoder_type == "mlp":
            board_flat = board.view(batch_size, -1)
            board_features = self.board_encoder(board_flat)
        else:  # CNN
            # Reshape to (batch, 5*30, 6) for 1D conv over turns
            board_reshaped = board.permute(0, 2, 3, 1).reshape(
                batch_size, 
                self.board_shape[1] * self.board_shape[2], 
                self.board_shape[0]
            )
            board_features = self.board_encoder(board_reshaped)
        
        alphabet_features = self.alphabet_encoder(alphabet)
        turn_features = self.turn_encoder(turn)
        
        combined = torch.cat([board_features, alphabet_features, turn_features], dim=-1)
        features = self.combiner(combined)
        
        return features


class DuelingHead(nn.Module):
    """
    Dueling network head that separates value and advantage streams.
    
    Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize the Dueling head.
        
        Args:
            input_dim: Input feature dimension
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Dueling head.
        
        Args:
            features: Input features of shape (batch, input_dim)
            
        Returns:
            Q-values of shape (batch, action_dim)
        """
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class StandardHead(nn.Module):
    """Standard Q-network head (no dueling)."""
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize the standard head.
        
        Args:
            input_dim: Input feature dimension
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the standard head.
        
        Args:
            features: Input features of shape (batch, input_dim)
            
        Returns:
            Q-values of shape (batch, action_dim)
        """
        return self.q_network(features)


class DQN(nn.Module):
    """
    Deep Q-Network with optional Dueling architecture.
    
    Supports MPS (Apple Silicon) acceleration and provides
    both standard and dueling variants.
    """
    
    def __init__(
        self,
        action_dim: int,
        board_shape: Tuple[int, int, int] = (6, 5, 30),  # FIXED: 30 = 26 letters + 4 color
        alphabet_size: int = 26,
        hidden_dim: int = 256,
        use_dueling: bool = False,
        encoder_type: Literal["mlp", "cnn"] = "mlp",
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the DQN.
        
        Args:
            action_dim: Number of possible actions (vocabulary size)
            board_shape: Shape of board observation
            alphabet_size: Size of alphabet status vector
            hidden_dim: Hidden layer dimension
            use_dueling: Whether to use Dueling architecture
            encoder_type: Encoder type ('mlp' or 'cnn')
            device: Device to place the model on
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_dueling = use_dueling
        self._device = device if device is not None else get_device()
        
        # Encoder
        self.encoder = DQNEncoder(
            board_shape=board_shape,
            alphabet_size=alphabet_size,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
        )
        
        # Head
        if use_dueling:
            self.head = DuelingHead(
                input_dim=self.encoder.output_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.head = StandardHead(
                input_dim=self.encoder.output_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
        
        # Move to device
        self.to(self._device)
    
    def forward(
        self, 
        board: torch.Tensor, 
        alphabet: torch.Tensor,
        turn: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the DQN.
        
        Args:
            board: Board observation of shape (batch, 6, 5, 30)
            alphabet: Alphabet status of shape (batch, 26)
            turn: Turn counter of shape (batch, 1)
            action_mask: Optional mask of shape (batch, action_dim)
            
        Returns:
            Q-values of shape (batch, action_dim)
        """
        features = self.encoder(board, alphabet, turn)
        q_values = self.head(features)
        
        # Apply action mask
        if action_mask is not None:
            # Set Q-values of invalid actions to large negative value
            invalid_mask = (action_mask == 0)
            q_values = q_values.masked_fill(invalid_mask, float('-inf'))
        
        return q_values
    
    def get_action(
        self,
        board: torch.Tensor,
        alphabet: torch.Tensor,
        turn: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        epsilon: float = 0.0,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            board: Board observation of shape (1, 6, 5, 30)
            alphabet: Alphabet status of shape (1, 26)
            turn: Turn counter of shape (1, 1)
            action_mask: Optional mask of shape (1, action_dim)
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            # Random exploration
            if action_mask is not None:
                valid_actions = torch.where(action_mask[0] > 0)[0]
                if len(valid_actions) > 0:
                    return valid_actions[torch.randint(len(valid_actions), (1,))].item()
            return torch.randint(self.action_dim, (1,)).item()
        
        with torch.no_grad():
            q_values = self.forward(board, alphabet, turn, action_mask)
            return q_values.argmax(dim=-1).item()
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device


def create_dqn(
    action_dim: int,
    use_dueling: bool = False,
    encoder_type: Literal["mlp", "cnn"] = "mlp",
    hidden_dim: int = 256,
    device: Optional[torch.device] = None,
) -> DQN:
    """
    Factory function to create a DQN model.
    
    Args:
        action_dim: Number of possible actions
        use_dueling: Whether to use Dueling architecture
        encoder_type: Type of encoder
        hidden_dim: Hidden dimension size
        device: Device to use
        
    Returns:
        Configured DQN model
    """
    return DQN(
        action_dim=action_dim,
        use_dueling=use_dueling,
        encoder_type=encoder_type,
        hidden_dim=hidden_dim,
        device=device,
    )
