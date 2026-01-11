#!/usr/bin/env python3
"""
Interactive Wordle Agent Evaluation Script.

Visualize the SOTA DQN agent guessing words step-by-step to understand
the model's behavior and decision-making process.

Usage:
    python evaluate_agent.py --model results/sota/phase2_final_model.pt
    python evaluate_agent.py --model results/sota/phase2_final_model.pt --word crane
    python evaluate_agent.py --model results/sota/phase2_final_model.pt --games 10
"""

import argparse
from typing import Optional
import numpy as np
import torch

from src.envs.wordle_v2 import WordleEnvSOTA
from src.models.sota_dqn import SOTADQNAgent, get_device


def format_feedback(guess: str, feedback: list) -> str:
    """Format guess with colored emoji feedback."""
    colors = {0: "⬛", 1: "🟨", 2: "🟩"}
    return " ".join(f"{colors[fb]}{c.upper()}" for c, fb in zip(guess, feedback))


def get_top_words(agent, obs, action_mask, top_k: int = 5):
    """Get top-k words by Q-value score."""
    agent.policy_net.eval()
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        letter_q = agent.policy_net.get_letter_q_values(obs_t)  # (1, 5, 26)
        
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=agent.device).unsqueeze(0)
        scores = agent.word_scorer.compute_word_scores(letter_q, mask_t)  # (1, num_words)
        
        top_indices = scores[0].argsort(descending=True)[:top_k]
        top_words = [(agent.word_list[idx.item()], scores[0, idx].item()) for idx in top_indices]
    
    return top_words


def print_letter_q_values(agent, obs):
    """Print a heatmap of letter Q-values by position."""
    agent.policy_net.eval()
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        letter_q = agent.policy_net.get_letter_q_values(obs_t)[0]  # (5, 26)
    
    print("\n  Letter Q-values by position (top 5 per position):")
    print("  " + "-" * 60)
    
    for pos in range(5):
        pos_q = letter_q[pos].cpu().numpy()
        top_indices = np.argsort(pos_q)[-5:][::-1]
        top_letters = [(chr(ord('A') + i), pos_q[i]) for i in top_indices]
        formatted = " ".join(f"{l}:{v:+.1f}" for l, v in top_letters)
        print(f"  Pos {pos}: {formatted}")


def play_game(agent, env, target_word: Optional[str] = None, verbose: bool = True):
    """Play one game and show the agent's decision process."""
    options = {"target_word": target_word} if target_word else None
    obs, info = env.reset(options=options)
    action_mask = info["action_mask"]
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"🎯 TARGET WORD: {'?' * 5} (hidden)")
        print("=" * 70)
    
    done = False
    turn = 0
    
    while not done:
        turn += 1
        
        # Get agent's decision
        action = agent.select_action(obs, action_mask, epsilon=0.0)
        guess = env.action_to_word(action)
        
        if verbose:
            print(f"\n--- Turn {turn}/6 ---")
            print(f"\n  Top 5 candidate words:")
            top_words = get_top_words(agent, obs, action_mask, top_k=5)
            for i, (word, score) in enumerate(top_words):
                marker = "👉" if word == guess else "  "
                print(f"  {marker} {i+1}. {word.upper()} (score: {score:.2f})")
        
        # Take action
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        
        if verbose:
            feedback = env.feedbacks[-1]
            print(f"\n  Guess: {format_feedback(guess, feedback)}")
            print(f"  Reward: {reward:+.2f}")
            
            # Show letter Q-values for understanding
            if not done and turn < 3:  # Only show for first few turns
                print_letter_q_values(agent, obs)
        
        obs = next_obs
        action_mask = next_info["action_mask"]
    
    # Game result
    won = env.guesses[-1] == env.target_word
    
    if verbose:
        print("\n" + "=" * 70)
        if won:
            print(f"✅ WON in {turn} guesses! Target: {env.target_word.upper()}")
        else:
            print(f"❌ LOST! Target was: {env.target_word.upper()}")
        print("=" * 70)
    
    return won, turn


def run_evaluation(model_path: str, target_word: Optional[str] = None, 
                   num_games: int = 1, phase: int = 2):
    """Run evaluation games."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Create environment
    env = WordleEnvSOTA()
    if phase == 1:
        env.solution_words = env.solution_words[:100]
    elif phase == 2:
        env.solution_words = env.solution_words[:1000]
    
    # Create agent and load model
    agent = SOTADQNAgent(
        word_list=env.valid_guesses,
        input_dim=313,
        hidden_dims=(512, 256),
        device=device,
    )
    
    print(f"Loading model from: {model_path}")
    agent.load(model_path)
    print("Model loaded successfully!\n")
    
    if num_games == 1:
        # Single game with detailed output
        play_game(agent, env, target_word, verbose=True)
    else:
        # Multiple games with summary
        wins = 0
        total_turns = 0
        
        print(f"Running {num_games} evaluation games...")
        print("-" * 40)
        
        for i in range(num_games):
            won, turns = play_game(agent, env, target_word=None, verbose=False)
            wins += int(won)
            if won:
                total_turns += turns
            
            # Progress
            if (i + 1) % 10 == 0 or i == num_games - 1:
                current_win_rate = wins / (i + 1)
                avg_turns = total_turns / max(wins, 1)
                print(f"  Game {i+1:4d}: Win rate = {current_win_rate:.1%}, Avg guesses = {avg_turns:.2f}")
        
        print("-" * 40)
        print(f"Final: {wins}/{num_games} wins ({wins/num_games:.1%})")
        print(f"Average guesses (wins only): {total_turns/max(wins,1):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wordle Agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--word", type=str, default=None, help="Specific target word to guess")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3],
                        help="Curriculum phase (affects solution word set)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model,
        target_word=args.word,
        num_games=args.games,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
