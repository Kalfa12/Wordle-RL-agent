#!/usr/bin/env python3
"""
SOTA Wordle DQN Training with Curriculum Learning.

This script implements the training pipeline for the Letter Decomposition
DQN architecture with 3-phase curriculum learning:

    Phase 1: 100 solution words (verify logic, fast iteration)
    Phase 2: 1,000 solution words (scale up, transfer learning)  
    Phase 3: 2,315 solution words (full training)

Each phase loads weights from the previous phase for transfer learning.

Usage:
    python train_curriculum.py --phase 1 --episodes 10000
    python train_curriculum.py --phase 2 --episodes 50000 --load-weights results/sota/phase1_model.pt
    python train_curriculum.py --phase 3 --episodes 100000 --load-weights results/sota/phase2_model.pt
"""

import argparse
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.envs.wordle_v2 import WordleEnvSOTA, create_curriculum_env
from src.models.sota_dqn import SOTADQNAgent, get_device
from src.utils.sota_replay import ReplayBuffer


@dataclass
class TrainingConfig:
    """Configuration for curriculum training."""
    
    # Curriculum phase (1, 2, or 3)
    phase: int = 1
    
    # Episodes per phase
    episodes: int = 10_000
    
    # DQN hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 5_000
    
    # Training
    warmup_episodes: int = 100
    target_update_freq: int = 100
    train_freq: int = 4  # Train every N steps
    
    # Logging
    log_freq: int = 100
    save_freq: int = 1_000
    eval_freq: int = 500
    eval_episodes: int = 100
    
    # Paths
    results_dir: str = "results/sota"
    load_weights: Optional[str] = None
    
    # Misc
    seed: int = 42
    smoke_test: bool = False
    continue_training: bool = False  # When True, skip epsilon decay (use low epsilon)
    initial_best_win_rate: float = 0.0  # Starting threshold for best model saving


class CurriculumTrainer:
    """
    Trainer for SOTA DQN with Curriculum Learning.
    """
    
    # Phase configurations
    PHASE_CONFIG = {
        1: {"num_words": 100, "desc": "Phase 1: 100 solution words"},
        2: {"num_words": 1000, "desc": "Phase 2: 1,000 solution words"},
        3: {"num_words": None, "desc": "Phase 3: Full 2,315 solution words"},  # None = all
    }
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer."""
        self.config = config
        self.device = get_device()
        
        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment with curriculum subset
        self._setup_environment()
        
        # Create agent
        self.agent = SOTADQNAgent(
            word_list=self.env.valid_guesses,
            input_dim=313,
            hidden_dims=(512, 256),
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            device=self.device,
        )
        
        # Load weights if specified
        if config.load_weights:
            print(f"Loading weights from {config.load_weights}")
            self.agent.load(config.load_weights)
        
        # Create replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            obs_dim=313,
            action_space_size=len(self.env.valid_guesses),
        )
        
        # Training state
        # When continuing, start with low epsilon to preserve learned behavior
        if config.continue_training:
            self.epsilon = config.epsilon_end
            print(f"  Continuing training with ε={self.epsilon} (skipping exploration warmup)")
        else:
            self.epsilon = config.epsilon_start
        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.wins: List[bool] = []
        self.losses: List[float] = []
        self.current_episode = 0
        self.interrupted = False
        
        # Setup graceful interrupt handling
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # CSV logging
        self._init_csv()
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully by saving checkpoint."""
        print("\n\n⚠️  Interrupt received! Saving checkpoint...")
        self._save_checkpoint(f"phase{self.config.phase}_interrupted_ep{self.current_episode}")
        print(f"✓ Checkpoint saved: phase{self.config.phase}_interrupted_ep{self.current_episode}_model.pt")
        self.interrupted = True
        sys.exit(0)
    
    def _setup_environment(self) -> None:
        """Setup environment with curriculum subset."""
        phase_info = self.PHASE_CONFIG[self.config.phase]
        
        # Create base environment
        self.env = WordleEnvSOTA(seed=self.config.seed)
        
        # Apply curriculum subset
        if phase_info["num_words"] is not None:
            self.env.solution_words = self.env.solution_words[:phase_info["num_words"]]
        
        print(f"\n{phase_info['desc']}")
        print(f"  Solution words: {len(self.env.solution_words)}")
        print(f"  Valid guesses: {len(self.env.valid_guesses)}")
    
    def _init_csv(self) -> None:
        """Initialize CSV logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.results_dir / f"phase{self.config.phase}_{timestamp}.csv"
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'win_rate', 'avg_reward', 'avg_length',
                'loss', 'epsilon', 'time_elapsed'
            ])
    
    def _log_to_csv(
        self,
        episode: int,
        win_rate: float,
        avg_reward: float,
        avg_length: float,
        loss: float,
        time_elapsed: float,
    ) -> None:
        """Log metrics to CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, f"{win_rate:.4f}", f"{avg_reward:.4f}",
                f"{avg_length:.2f}", f"{loss:.6f}", f"{self.epsilon:.4f}",
                f"{time_elapsed:.1f}"
            ])
    
    def _update_epsilon(self, episode: int) -> None:
        """Update epsilon with linear decay."""
        # Skip decay if continuing training (already at low epsilon)
        if self.config.continue_training:
            return
        
        if episode < self.config.epsilon_decay_episodes:
            decay_progress = episode / self.config.epsilon_decay_episodes
            self.epsilon = self.config.epsilon_start - decay_progress * (
                self.config.epsilon_start - self.config.epsilon_end
            )
        else:
            self.epsilon = self.config.epsilon_end
    
    def _train_step(self) -> Optional[float]:
        """Perform a single training step."""
        if not self.buffer.is_ready(self.config.batch_size):
            return None
        
        # Sample batch
        (states, actions, rewards, next_states, dones,
         action_masks, next_action_masks) = self.buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        next_masks_t = None
        if next_action_masks is not None:
            next_masks_t = torch.tensor(
                next_action_masks, dtype=torch.float32, device=self.device
            )
        
        # Compute loss and update
        self.agent.policy_net.train()
        self.agent.optimizer.zero_grad()
        
        loss = self.agent.compute_loss(
            states_t, actions_t, rewards_t, next_states_t, dones_t, next_masks_t
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), max_norm=10.0)
        self.agent.optimizer.step()
        
        return loss.item()
    
    def _run_episode(self) -> tuple:
        """Run a single episode."""
        obs, info = self.env.reset()
        action_mask = info.get("action_mask")
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Select action
            action = self.agent.select_action(obs, action_mask, self.epsilon)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated
            next_action_mask = next_info.get("action_mask")
            
            # Store transition
            self.buffer.push(
                obs, action, reward, next_obs, done,
                action_mask, next_action_mask
            )
            
            # Train
            self.total_steps += 1
            if self.total_steps % self.config.train_freq == 0:
                loss = self._train_step()
                if loss is not None:
                    self.losses.append(loss)
            
            # Update state
            obs = next_obs
            action_mask = next_action_mask
            total_reward += reward
            steps += 1
        
        # Check if won
        won = (self.env.guesses[-1] == self.env.target_word) if self.env.guesses else False
        
        return total_reward, steps, won
    
    def _evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate current policy."""
        wins = 0
        total_guesses = 0
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            action_mask = info.get("action_mask")
            done = False
            
            while not done:
                action = self.agent.select_action(obs, action_mask, epsilon=0.0)
                obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                action_mask = info.get("action_mask")
            
            if self.env.guesses and self.env.guesses[-1] == self.env.target_word:
                wins += 1
                total_guesses += len(self.env.guesses)
        
        return {
            "win_rate": wins / num_episodes,
            "avg_guesses": total_guesses / max(wins, 1),
        }
    
    def train(self) -> Dict[str, List]:
        """Run the full training loop."""
        print(f"\nStarting training:")
        print(f"  Episodes: {self.config.episodes}")
        print(f"  Device: {self.device}")
        print(f"  Epsilon decay: {self.config.epsilon_start} → {self.config.epsilon_end}")
        print(f"  over {self.config.epsilon_decay_episodes} episodes")
        if self.config.initial_best_win_rate > 0:
            print(f"  Best win rate threshold: {self.config.initial_best_win_rate:.1%}")
        print()
        
        start_time = time.time()
        best_win_rate = self.config.initial_best_win_rate
        
        for episode in range(1, self.config.episodes + 1):
            self.current_episode = episode
            
            # Update epsilon
            self._update_epsilon(episode)
            
            # Run episode
            reward, length, won = self._run_episode()
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.wins.append(won)
            
            # Update target network
            if episode % self.config.target_update_freq == 0:
                self.agent.update_target_network()
            
            # Logging
            if episode % self.config.log_freq == 0:
                recent_wins = self.wins[-self.config.log_freq:]
                recent_rewards = self.episode_rewards[-self.config.log_freq:]
                recent_lengths = self.episode_lengths[-self.config.log_freq:]
                recent_losses = self.losses[-100:] if self.losses else [0.0]
                
                win_rate = sum(recent_wins) / len(recent_wins)
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                avg_loss = np.mean(recent_losses)
                elapsed = time.time() - start_time
                
                print(
                    f"Episode {episode:6d} | "
                    f"Win: {win_rate:5.1%} | "
                    f"Reward: {avg_reward:6.2f} | "
                    f"Guesses: {avg_length:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"ε: {self.epsilon:.3f} | "
                    f"Time: {elapsed:.0f}s"
                )
                
                self._log_to_csv(
                    episode, win_rate, avg_reward, avg_length, avg_loss, elapsed
                )
                
                # Save best model based on training win rate
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self._save_checkpoint(f"phase{self.config.phase}_best")
                    print(f"  → New best win rate: {best_win_rate:.1%}")
            
            # Save checkpoint (phase-specific naming)
            if episode % self.config.save_freq == 0:
                self._save_checkpoint(f"phase{self.config.phase}_ep{episode}")
            
            # Smoke test: exit early
            if self.config.smoke_test and episode >= 100:
                print("\nSmoke test complete!")
                break
        
        # Final save
        self._save_checkpoint(f"phase{self.config.phase}_final")
        
        print(f"\nTraining complete!")
        print(f"  Best win rate: {best_win_rate:.1%}")
        print(f"  Model saved to: {self.results_dir}")
        
        return {
            "rewards": self.episode_rewards,
            "lengths": self.episode_lengths,
            "wins": self.wins,
            "losses": self.losses,
        }
    
    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = self.results_dir / f"{name}_model.pt"
        self.agent.save(str(path))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SOTA Wordle DQN Training with Curriculum Learning"
    )
    
    # Phase
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3],
        help="Curriculum phase (1=100 words, 2=1000 words, 3=full)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10_000,
        help="Number of training episodes"
    )
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size")
    
    # Exploration
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=int, default=5_000, help="Episodes to decay epsilon")
    
    # Paths
    parser.add_argument("--load-weights", type=str, help="Path to pretrained weights")
    parser.add_argument("--results-dir", type=str, default="results/sota")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Run quick test (100 episodes)")
    parser.add_argument("--continue", dest="continue_training", action="store_true", 
                        help="Continue training from loaded weights (skip epsilon warmup)")
    parser.add_argument("--best-win-rate", type=float, default=0.0,
                        help="Initial best win rate threshold (prevents saving worse models)")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        phase=args.phase,
        episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay,
        load_weights=args.load_weights,
        results_dir=args.results_dir,
        seed=args.seed,
        smoke_test=args.smoke_test,
        continue_training=args.continue_training,
        initial_best_win_rate=args.best_win_rate,
    )
    
    # Train
    trainer = CurriculumTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
