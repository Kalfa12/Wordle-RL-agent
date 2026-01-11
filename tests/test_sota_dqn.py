"""
Unit tests for SOTA DQN with Letter Decomposition (sota_dqn.py).

Run with: python -m pytest tests/test_sota_dqn.py -v
"""

import numpy as np
import pytest
import torch

from src.models.sota_dqn import (
    LetterDecompositionDQN,
    WordScorer,
    SOTADQNAgent,
    get_device,
)


# Force CPU for all tests to avoid MPS issues
TEST_DEVICE = torch.device("cpu")


class TestLetterDecompositionDQN:
    """Tests for the Letter Decomposition DQN architecture."""
    
    def test_output_shape(self):
        """Output should be exactly 130 dimensions (26 letters × 5 positions)."""
        model = LetterDecompositionDQN(input_dim=313, device=TEST_DEVICE)
        
        obs = torch.randn(1, 313)
        output = model(obs)
        
        assert output.shape == (1, 130), f"Expected (1, 130), got {output.shape}"
    
    def test_batch_output_shape(self):
        """Should handle batched inputs correctly."""
        model = LetterDecompositionDQN(input_dim=313, device=TEST_DEVICE)
        
        batch_size = 32
        obs = torch.randn(batch_size, 313)
        output = model(obs)
        
        assert output.shape == (batch_size, 130)
    
    def test_letter_q_values_reshape(self):
        """get_letter_q_values should reshape to (batch, 5, 26)."""
        model = LetterDecompositionDQN(input_dim=313, device=TEST_DEVICE)
        
        obs = torch.randn(4, 313)
        letter_q = model.get_letter_q_values(obs)
        
        assert letter_q.shape == (4, 5, 26), f"Expected (4, 5, 26), got {letter_q.shape}"
    
    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        model = LetterDecompositionDQN(input_dim=313, device=TEST_DEVICE)
        
        obs = torch.randn(1, 313, requires_grad=False)
        output = model(obs)
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model parameters"


class TestWordScorer:
    """Tests for word scoring logic."""
    
    @pytest.fixture
    def simple_word_list(self):
        """A simple word list for testing."""
        return ["crane", "about", "zebra"]
    
    def test_word_matrix_shape(self, simple_word_list):
        """Word matrix should have shape (num_words, 5, 26)."""
        scorer = WordScorer(simple_word_list, device=torch.device("cpu"))
        
        assert scorer.word_matrix.shape == (3, 5, 26)
    
    def test_word_matrix_encoding(self, simple_word_list):
        """Word matrix should correctly encode letters."""
        scorer = WordScorer(simple_word_list, device=torch.device("cpu"))
        
        # "crane" = [c, r, a, n, e]
        # Position 0, letter 'c' (index 2) should be 1
        assert scorer.word_matrix[0, 0, 2] == 1.0  # 'c' at position 0
        assert scorer.word_matrix[0, 1, 17] == 1.0  # 'r' at position 1 (r = 17)
    
    def test_word_score_computation(self, simple_word_list):
        """Word score should equal sum of letter Q-values."""
        scorer = WordScorer(simple_word_list, device=torch.device("cpu"))
        
        # Create known Q-values: all 1.0
        letter_q = torch.ones(1, 5, 26)
        
        scores = scorer.compute_word_scores(letter_q)
        
        # Each word has 5 letters, each with Q=1.0, so score = 5.0
        assert scores.shape == (1, 3)
        assert torch.allclose(scores, torch.tensor([[5.0, 5.0, 5.0]]))
    
    def test_word_score_differentiation(self, simple_word_list):
        """Different Q-values should produce different word scores."""
        scorer = WordScorer(simple_word_list, device=torch.device("cpu"))
        
        # Make 'c' at position 0 very valuable (affects "crane")
        letter_q = torch.zeros(1, 5, 26)
        letter_q[0, 0, 2] = 10.0  # 'c' at position 0
        
        scores = scorer.compute_word_scores(letter_q)
        
        # "crane" should have highest score (has 'c' at position 0)
        crane_score = scores[0, 0].item()
        about_score = scores[0, 1].item()
        zebra_score = scores[0, 2].item()
        
        assert crane_score > about_score
        assert crane_score > zebra_score
    
    def test_action_mask_application(self, simple_word_list):
        """Masked words should have very low scores."""
        scorer = WordScorer(simple_word_list, device=torch.device("cpu"))
        
        letter_q = torch.ones(1, 5, 26)
        
        # Mask out "about" (index 1)
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        
        scores = scorer.compute_word_scores(letter_q, action_mask=mask)
        
        assert scores[0, 0] > -1e6  # "crane" unmasked
        assert scores[0, 1] < -1e6  # "about" masked to very negative
        assert scores[0, 2] > -1e6  # "zebra" unmasked
    
    def test_select_best_word(self, simple_word_list):
        """Should select word with highest score."""
        scorer = WordScorer(simple_word_list, device=torch.device("cpu"))
        
        # Make "zebra" the best choice (boost 'z' at position 0)
        letter_q = torch.zeros(1, 5, 26)
        letter_q[0, 0, 25] = 100.0  # 'z' at position 0
        
        indices, words = scorer.select_best_word(letter_q)
        
        assert words[0] == "zebra"


class TestSOTADQNAgent:
    """Tests for the complete agent."""
    
    @pytest.fixture
    def small_word_list(self):
        """Small word list for fast testing."""
        return ["crane", "about", "beach", "child", "drink"]
    
    def test_action_selection(self, small_word_list):
        """Agent should return valid action indices."""
        agent = SOTADQNAgent(
            word_list=small_word_list,
            input_dim=313,
            hidden_dims=(64,),
            device=torch.device("cpu"),
        )
        
        obs = np.random.randn(313).astype(np.float32)
        action = agent.select_action(obs, epsilon=0.0)
        
        assert 0 <= action < len(small_word_list)
    
    def test_epsilon_exploration(self, small_word_list):
        """With epsilon=1.0, should explore randomly."""
        agent = SOTADQNAgent(
            word_list=small_word_list,
            input_dim=313,
            hidden_dims=(64,),
            device=torch.device("cpu"),
        )
        
        obs = np.random.randn(313).astype(np.float32)
        
        # With epsilon=1.0, all actions should be possible
        actions = [agent.select_action(obs, epsilon=1.0) for _ in range(100)]
        unique_actions = set(actions)
        
        # Should explore multiple actions
        assert len(unique_actions) >= 2
    
    def test_action_mask_respected(self, small_word_list):
        """Agent should only select from valid actions when masked."""
        agent = SOTADQNAgent(
            word_list=small_word_list,
            input_dim=313,
            hidden_dims=(64,),
            device=torch.device("cpu"),
        )
        
        obs = np.random.randn(313).astype(np.float32)
        
        # Only allow action 2
        mask = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        
        # Even with epsilon=1.0, should only select valid action
        for _ in range(10):
            action = agent.select_action(obs, action_mask=mask, epsilon=1.0)
            assert action == 2
    
    def test_loss_computation(self, small_word_list):
        """Loss computation should work without errors."""
        agent = SOTADQNAgent(
            word_list=small_word_list,
            input_dim=313,
            hidden_dims=(64,),
            device=torch.device("cpu"),
        )
        
        batch_size = 4
        states = torch.randn(batch_size, 313)
        actions = torch.randint(0, len(small_word_list), (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 313)
        dones = torch.zeros(batch_size)
        
        loss = agent.compute_loss(states, actions, rewards, next_states, dones)
        
        assert loss.item() >= 0  # MSE loss is non-negative
        assert not torch.isnan(loss)
    
    def test_target_network_update(self, small_word_list):
        """Target network should copy from policy network."""
        agent = SOTADQNAgent(
            word_list=small_word_list,
            input_dim=313,
            hidden_dims=(64,),
            device=torch.device("cpu"),
        )
        
        # Modify policy network
        with torch.no_grad():
            for param in agent.policy_net.parameters():
                param.fill_(42.0)
        
        # Before update, target should be different
        target_sum_before = sum(p.sum().item() for p in agent.target_net.parameters())
        
        # Update target
        agent.update_target_network()
        
        # After update, target should match policy
        for p_param, t_param in zip(
            agent.policy_net.parameters(),
            agent.target_net.parameters()
        ):
            assert torch.allclose(p_param, t_param)
    
    def test_save_load(self, small_word_list, tmp_path):
        """Save and load should preserve weights."""
        agent = SOTADQNAgent(
            word_list=small_word_list,
            input_dim=313,
            hidden_dims=(64,),
            device=torch.device("cpu"),
        )
        
        # Get a reproducible action
        obs = np.ones(313, dtype=np.float32)
        action_before = agent.select_action(obs, epsilon=0.0)
        
        # Save
        path = str(tmp_path / "model.pt")
        agent.save(path)
        
        # Create new agent and load
        agent2 = SOTADQNAgent(
            word_list=small_word_list,
            input_dim=313,
            hidden_dims=(64,),
            device=torch.device("cpu"),
        )
        agent2.load(path)
        
        # Should get same action
        action_after = agent2.select_action(obs, epsilon=0.0)
        
        assert action_before == action_after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
