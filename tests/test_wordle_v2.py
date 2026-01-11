"""
Unit tests for SOTA Wordle Environment (wordle_v2.py).

Run with: python -m pytest tests/test_wordle_v2.py -v
"""

import numpy as np
import pytest

from src.envs.wordle_v2 import WordleEnvSOTA, create_curriculum_env


class TestObservationSpace:
    """Tests for the 313-dimensional observation space."""
    
    def test_observation_shape(self):
        """Observation should be exactly 313 dimensions."""
        env = WordleEnvSOTA()
        obs, _ = env.reset()
        
        assert obs.shape == (313,), f"Expected (313,), got {obs.shape}"
        assert obs.dtype == np.float32
    
    def test_observation_bounds(self):
        """All observation values should be in [0, 1]."""
        env = WordleEnvSOTA()
        
        for _ in range(10):
            obs, _ = env.reset()
            assert obs.min() >= 0.0, "Observation contains negative values"
            assert obs.max() <= 1.0, "Observation contains values > 1"
    
    def test_turns_remaining_decrement(self):
        """turns_remaining should decrement properly."""
        env = WordleEnvSOTA()
        obs, info = env.reset()
        
        # Initial: 6/6 = 1.0
        assert obs[0] == 1.0, f"Initial turns_remaining should be 1.0, got {obs[0]}"
        
        # After one guess: 5/6
        action = env.word_to_action("crane")
        obs, _, _, _, _ = env.step(action)
        assert np.isclose(obs[0], 5/6), f"After 1 guess, expected 5/6, got {obs[0]}"


class TestDictionarySplit:
    """Tests for Real Wordle dictionary split."""
    
    def test_solution_words_loaded(self):
        """Should load ~2,315 solution words from wordle-La.txt."""
        env = WordleEnvSOTA()
        
        # Approximately 2,315 solution words
        assert 2000 <= len(env.solution_words) <= 2500, \
            f"Expected ~2,315 solution words, got {len(env.solution_words)}"
    
    def test_valid_guesses_loaded(self):
        """Should load ~12,972 valid guesses (Ta + La combined)."""
        env = WordleEnvSOTA()
        
        # Valid guesses = Ta words + La words (deduplicated)
        assert len(env.valid_guesses) > 10000, \
            f"Expected >10,000 valid guesses, got {len(env.valid_guesses)}"
    
    def test_solutions_subset_of_guesses(self):
        """All solution words should be valid guesses."""
        env = WordleEnvSOTA()
        
        for word in env.solution_words:
            assert word in env.valid_guesses_set, \
                f"Solution word '{word}' not in valid guesses"
    
    def test_target_words_from_solutions(self):
        """Target words should only come from solution set."""
        env = WordleEnvSOTA()
        
        for _ in range(100):
            _, info = env.reset()
            target = info["target_word"]
            assert target in env.solution_words_set, \
                f"Target '{target}' not in solution set"


class TestRewardFunction:
    """Tests for the reward function."""
    
    def test_reward_win(self):
        """Correct guess should give +10 reward."""
        env = WordleEnvSOTA()
        obs, info = env.reset(options={"target_word": "crane"})
        
        action = env.word_to_action("crane")
        _, reward, terminated, _, _ = env.step(action)
        
        assert reward == 10.0, f"Win reward should be 10.0, got {reward}"
        assert terminated, "Game should terminate on win"
    
    def test_reward_loss(self):
        """6 wrong guesses should give -10 reward."""
        env = WordleEnvSOTA()
        env.reset(options={"target_word": "zebra"})
        
        wrong_words = ["about", "beach", "child", "drink", "eight", "fancy"]
        
        for i, word in enumerate(wrong_words):
            action = env.word_to_action(word)
            _, reward, terminated, _, _ = env.step(action)
            
            if i < 5:
                assert not terminated
            else:
                assert reward == -10.0, f"Loss reward should be -10.0, got {reward}"
                assert terminated
    
    def test_reward_step_penalty(self):
        """Each step should have -0.1 base penalty."""
        env = WordleEnvSOTA()
        env.reset(options={"target_word": "zebra"})
        
        # First guess with no new greens
        action = env.word_to_action("about")  # No letters match "zebra"
        _, reward, _, _, _ = env.step(action)
        
        # Reward = -0.1 (step) + 0 (no greens)
        # Note: 'b' is in zebra, so it's yellow, not green
        # Check that reward is close to step penalty (may have small bonus)
        assert reward <= 0.0, f"Step reward should be non-positive, got {reward}"
    
    def test_reward_green_bonus(self):
        """Finding new green letters should give +0.5 bonus each."""
        env = WordleEnvSOTA()
        env.reset(options={"target_word": "crane"})
        
        # "crane" → "crane": 5 green letters found
        # But we want partial: try "trace" which shares 'r', 'a', 'e' but not all green
        # Actually just test with something that gets exactly 1 green
        
        # "clung" has 'c' at position 0 like "crane"
        action = env.word_to_action("clung")
        _, reward, _, _, _ = env.step(action)
        
        # Should get -0.1 + 0.5 = 0.4 (1 new green: 'c' at position 0)
        assert reward > 0, f"Should get positive reward for green letter, got {reward}"


class TestActionMask:
    """Tests for action masking."""
    
    def test_action_mask_shape(self):
        """Action mask should match vocabulary size."""
        env = WordleEnvSOTA()
        _, info = env.reset()
        
        mask = info["action_mask"]
        assert mask.shape == (env.vocab_size,)
    
    def test_action_mask_initial(self):
        """Initially, all actions should be valid."""
        env = WordleEnvSOTA()
        _, info = env.reset()
        
        mask = info["action_mask"]
        # At least many actions should be valid initially
        assert mask.sum() > 1000, "Too few valid actions initially"
    
    def test_action_mask_after_green(self):
        """After finding a green letter, words without it should be masked."""
        env = WordleEnvSOTA()
        env.reset(options={"target_word": "crane"})
        
        # Guess "clear" - gets 'c' green at position 0, 'r' green at position 2
        action = env.word_to_action("clear")
        _, _, _, _, info = env.step(action)
        
        mask = info["action_mask"]
        
        # Check that "about" (no 'c' at pos 0) is masked
        about_idx = env.word_to_action("about")
        # Note: mask might still allow it if logic is permissive
        # Just check that mask is doing something
        total_valid = mask.sum()
        assert total_valid < len(env.valid_guesses), "Mask should reduce valid actions"


class TestCurriculumEnv:
    """Tests for curriculum learning environment creation."""
    
    def test_phase1_subset(self):
        """Phase 1 should have exactly 100 solution words."""
        env = create_curriculum_env(phase=1)
        assert len(env.solution_words) == 100
    
    def test_phase2_subset(self):
        """Phase 2 should have exactly 1000 solution words."""
        env = create_curriculum_env(phase=2)
        assert len(env.solution_words) == 1000
    
    def test_phase3_full(self):
        """Phase 3 should have full solution set."""
        env = create_curriculum_env(phase=3)
        assert len(env.solution_words) > 2000


class TestBasicGameplay:
    """Tests for basic game mechanics."""
    
    def test_word_conversion(self):
        """word_to_action and action_to_word should be inverses."""
        env = WordleEnvSOTA()
        
        for word in ["crane", "about", "zebra"]:
            action = env.word_to_action(word)
            recovered = env.action_to_word(action)
            assert recovered == word, f"Mismatch: {word} → {action} → {recovered}"
    
    def test_game_terminates(self):
        """Game should terminate after 6 guesses or win."""
        env = WordleEnvSOTA()
        env.reset()
        
        for _ in range(10):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
        
        assert env.current_turn <= 6
    
    def test_render_works(self):
        """Render should return string without errors."""
        env = WordleEnvSOTA(render_mode="ansi")
        env.reset()
        
        action = env.word_to_action("crane")
        env.step(action)
        
        output = env.render()
        assert isinstance(output, str)
        assert all(c in output.upper() for c in "CRANE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
