"""
SOTA Wordle Environment with Letter Decomposition Support.

This environment implements the "Real Wordle" rules with a 313-dimensional
dense observation space optimized for the Letter Decomposition DQN architecture.

Key Features:
- Solutions sampled from ~2,315 common words (wordle-La.txt)
- Valid guesses from full ~10,657 word dictionary (wordle-Ta.txt)
- 313-dim flat observation vector for efficient learning
- Shaped rewards with green letter bonus for exploration guidance
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WordleEnvSOTA(gym.Env):
    """
    SOTA Wordle Environment with 313-dimensional observation space.
    
    Observation Space (313 dimensions):
        [0]:        turns_remaining (normalized 0-1)
        [1:27]:     letters_used (26 binary flags)
        [27:53]:    letters_green_anywhere (26 binary flags)
        [53:183]:   letters_green_at_pos (26×5 = 130 one-hot positions)
        [183:313]:  letters_yellow_or_gray_at_pos (26×5 = 130 binary flags)
    
    Action Space:
        Discrete(vocab_size) where vocab_size = len(valid_guesses)
    
    Reward Function:
        Win:        +10.0
        Loss:       -10.0
        Step:       -0.1
        New Green:  +0.5 per new green letter discovered
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Constants
    WORD_LENGTH = 5
    MAX_TURNS = 6
    NUM_LETTERS = 26
    
    # Reward constants
    REWARD_WIN = 10.0
    REWARD_LOSS = -10.0
    REWARD_STEP = -0.1
    REWARD_GREEN_BONUS = 0.5
    
    def __init__(
        self,
        seed: int = 42,
        render_mode: Optional[str] = None,
        solution_words_path: Optional[str] = None,
        guess_words_path: Optional[str] = None,
    ):
        """
        Initialize the SOTA Wordle environment.
        
        Args:
            seed: Random seed for reproducibility
            render_mode: "human" or "ansi" for text rendering
            solution_words_path: Path to wordle-La.txt (solution words)
            guess_words_path: Path to wordle-Ta.txt (valid guesses)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self._np_random = np.random.default_rng(seed)
        
        # Load dictionaries
        self._load_dictionaries(solution_words_path, guess_words_path)
        
        # Build word-to-index mapping for actions
        self.word_to_idx = {word: i for i, word in enumerate(self.valid_guesses)}
        self.idx_to_word = {i: word for i, word in enumerate(self.valid_guesses)}
        
        # Precompute letter matrices for fast scoring
        self._precompute_word_letters()
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(313,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.valid_guesses))
        
        # Episode state
        self.target_word: str = ""
        self.current_turn: int = 0
        self.guesses: List[str] = []
        self.feedbacks: List[List[int]] = []  # 0=gray, 1=yellow, 2=green
        
        # Tracking state for observations
        self.letters_used = np.zeros(26, dtype=np.float32)
        self.letters_green_anywhere = np.zeros(26, dtype=np.float32)
        self.letters_green_at_pos = np.zeros((5, 26), dtype=np.float32)
        self.letters_yellow_at_pos = np.zeros((5, 26), dtype=np.float32)
        self.letters_gray_at_pos = np.zeros((5, 26), dtype=np.float32)
        
        # Track green letters for bonus calculation
        self.prev_green_count = 0
    
    def _load_dictionaries(
        self,
        solution_path: Optional[str],
        guess_path: Optional[str]
    ) -> None:
        """Load solution and guess word dictionaries."""
        # Find dictionary files
        base_dir = Path(__file__).parent.parent.parent  # Go up from src/envs/
        
        if solution_path is None:
            solution_path = base_dir / "wordle-La.txt"
        if guess_path is None:
            guess_path = base_dir / "wordle-Ta.txt"
        
        # Load solution words (targets)
        with open(solution_path, 'r') as f:
            self.solution_words = [line.strip().lower() for line in f if line.strip()]
        
        # Load valid guesses
        with open(guess_path, 'r') as f:
            guess_words = [line.strip().lower() for line in f if line.strip()]
        
        # Valid guesses = guess words + solution words (deduplicated)
        self.valid_guesses = sorted(set(guess_words + self.solution_words))
        
        # Create set for fast lookup
        self.valid_guesses_set = set(self.valid_guesses)
        self.solution_words_set = set(self.solution_words)
    
    def _precompute_word_letters(self) -> None:
        """Precompute letter-position matrix for all valid words."""
        # Matrix of shape (num_words, 5, 26) for letter at each position
        num_words = len(self.valid_guesses)
        self.word_letter_matrix = np.zeros((num_words, 5, 26), dtype=np.float32)
        
        for i, word in enumerate(self.valid_guesses):
            for pos, char in enumerate(word):
                letter_idx = ord(char) - ord('a')
                self.word_letter_matrix[i, pos, letter_idx] = 1.0
    
    def _get_obs(self) -> np.ndarray:
        """Construct the 313-dimensional observation vector."""
        obs = np.zeros(313, dtype=np.float32)
        
        # [0]: turns_remaining (normalized)
        obs[0] = (self.MAX_TURNS - self.current_turn) / self.MAX_TURNS
        
        # [1:27]: letters_used
        obs[1:27] = self.letters_used
        
        # [27:53]: letters_green_anywhere
        obs[27:53] = self.letters_green_anywhere
        
        # [53:183]: letters_green_at_pos (flattened 5×26)
        obs[53:183] = self.letters_green_at_pos.flatten()
        
        # [183:313]: letters_yellow_or_gray_at_pos (flattened 5×26)
        # Combine yellow and gray to indicate "letter ruled out at this position"
        yellow_or_gray = np.clip(self.letters_yellow_at_pos + self.letters_gray_at_pos, 0, 1)
        obs[183:313] = yellow_or_gray.flatten()
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional info for debugging and action masking."""
        return {
            "target_word": self.target_word,
            "current_turn": self.current_turn,
            "guesses": self.guesses.copy(),
            "action_mask": self._compute_action_mask(),
        }
    
    def _compute_action_mask(self) -> np.ndarray:
        """
        Compute action mask based on known constraints.
        
        A word is invalid if:
        1. It uses a gray letter at any position
        2. It doesn't use a known green letter at its position
        3. It uses a yellow letter at its yellow position
        """
        mask = np.ones(len(self.valid_guesses), dtype=np.float32)
        
        for word_idx in range(len(self.valid_guesses)):
            word_matrix = self.word_letter_matrix[word_idx]  # (5, 26)
            
            # Check each position
            for pos in range(5):
                letter_idx = np.argmax(word_matrix[pos])
                
                # If this position has a known green letter, word must match
                if self.letters_green_at_pos[pos].any():
                    green_letter = np.argmax(self.letters_green_at_pos[pos])
                    if letter_idx != green_letter:
                        mask[word_idx] = 0.0
                        break
                
                # If letter is gray at this position, invalid
                if self.letters_gray_at_pos[pos, letter_idx] > 0:
                    mask[word_idx] = 0.0
                    break
                
                # If letter was yellow at this exact position, invalid
                if self.letters_yellow_at_pos[pos, letter_idx] > 0:
                    mask[word_idx] = 0.0
                    break
        
        # Ensure at least one action is valid (fallback)
        if mask.sum() == 0:
            mask[:] = 1.0
        
        return mask
    
    def _get_feedback(self, guess: str) -> List[int]:
        """
        Get Wordle feedback for a guess.
        
        Returns list of 5 integers:
            0 = gray (letter not in word)
            1 = yellow (letter in word, wrong position)
            2 = green (letter in correct position)
        """
        feedback = [0] * 5
        target_chars = list(self.target_word)
        guess_chars = list(guess)
        
        # First pass: mark greens
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback[i] = 2
                target_chars[i] = None  # Mark as used
                guess_chars[i] = None
        
        # Second pass: mark yellows
        for i in range(5):
            if guess_chars[i] is not None:
                if guess_chars[i] in target_chars:
                    feedback[i] = 1
                    # Mark first occurrence as used
                    target_chars[target_chars.index(guess_chars[i])] = None
        
        return feedback
    
    def _update_state(self, guess: str, feedback: List[int]) -> None:
        """Update tracking state based on guess and feedback."""
        for pos, (char, fb) in enumerate(zip(guess, feedback)):
            letter_idx = ord(char) - ord('a')
            
            # Mark letter as used
            self.letters_used[letter_idx] = 1.0
            
            if fb == 2:  # Green
                self.letters_green_anywhere[letter_idx] = 1.0
                self.letters_green_at_pos[pos, letter_idx] = 1.0
            elif fb == 1:  # Yellow
                self.letters_yellow_at_pos[pos, letter_idx] = 1.0
            else:  # Gray
                self.letters_gray_at_pos[pos, letter_idx] = 1.0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        
        # Select target word from solutions
        if options and "target_word" in options:
            self.target_word = options["target_word"].lower()
        else:
            self.target_word = self._np_random.choice(self.solution_words)
        
        # Reset episode state
        self.current_turn = 0
        self.guesses = []
        self.feedbacks = []
        
        # Reset tracking arrays
        self.letters_used.fill(0)
        self.letters_green_anywhere.fill(0)
        self.letters_green_at_pos.fill(0)
        self.letters_yellow_at_pos.fill(0)
        self.letters_gray_at_pos.fill(0)
        self.prev_green_count = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index of the word to guess
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get the guessed word
        guess = self.idx_to_word[action]
        
        # Get feedback
        feedback = self._get_feedback(guess)
        
        # Update state
        self._update_state(guess, feedback)
        
        # Record guess
        self.guesses.append(guess)
        self.feedbacks.append(feedback)
        self.current_turn += 1
        
        # Calculate reward
        won = (guess == self.target_word)
        lost = (self.current_turn >= self.MAX_TURNS and not won)
        
        # Count new green letters for bonus
        current_green_count = int(self.letters_green_at_pos.sum())
        new_greens = current_green_count - self.prev_green_count
        self.prev_green_count = current_green_count
        
        # Compute reward
        if won:
            reward = self.REWARD_WIN
        elif lost:
            reward = self.REWARD_LOSS
        else:
            reward = self.REWARD_STEP + (new_greens * self.REWARD_GREEN_BONUS)
        
        terminated = won or lost
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self) -> Optional[str]:
        """Render the current game state."""
        if self.render_mode not in ["human", "ansi"]:
            return None
        
        output = []
        output.append(f"\n{'='*30}")
        output.append(f"Target: {'*****' if self.current_turn < self.MAX_TURNS else self.target_word}")
        output.append(f"Turn: {self.current_turn}/{self.MAX_TURNS}")
        output.append("-" * 30)
        
        color_map = {0: "⬛", 1: "🟨", 2: "🟩"}
        
        for guess, feedback in zip(self.guesses, self.feedbacks):
            colored = " ".join(
                f"{color_map[fb]}{c.upper()}" for c, fb in zip(guess, feedback)
            )
            output.append(colored)
        
        output.append("=" * 30)
        
        result = "\n".join(output)
        if self.render_mode == "human":
            print(result)
        return result
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size (action space size)."""
        return len(self.valid_guesses)
    
    def word_to_action(self, word: str) -> int:
        """Convert a word to its action index."""
        return self.word_to_idx[word.lower()]
    
    def action_to_word(self, action: int) -> str:
        """Convert an action index to its word."""
        return self.idx_to_word[action]


# Convenience function to create environment with curriculum learning subset
def create_curriculum_env(
    phase: int,
    seed: int = 42,
    solution_words: Optional[List[str]] = None,
) -> WordleEnvSOTA:
    """
    Create environment with curriculum learning subset.
    
    Args:
        phase: Curriculum phase (1, 2, or 3)
        seed: Random seed
        solution_words: Optional explicit list of solution words
        
    Returns:
        WordleEnvSOTA configured for the curriculum phase
    """
    env = WordleEnvSOTA(seed=seed)
    
    if solution_words is not None:
        env.solution_words = solution_words
    elif phase == 1:
        # Phase 1: First 100 solution words
        env.solution_words = env.solution_words[:100]
    elif phase == 2:
        # Phase 2: First 1000 solution words
        env.solution_words = env.solution_words[:1000]
    # Phase 3: Full set (default)
    
    return env
