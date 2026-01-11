#!/usr/bin/env python3
"""
🎓 PROFESSOR DEMO SCRIPT 🎓

An impressive, colorful terminal demo for presenting the Wordle RL Agent.

Features:
- ANSI color-coded Wordle feedback (Green/Yellow/Gray)
- Top-5 word candidates with Q-value scores
- Step-by-step commentary on agent decisions
- Interactive mode: let the teacher pick custom words

Usage:
    python presentation/demo_teacher.py                    # Random word demo
    python presentation/demo_teacher.py --word crane       # Specific word
    python presentation/demo_teacher.py --interactive      # Teacher picks words
    python presentation/demo_teacher.py --model path.pt    # Custom model

Requirements: Trained model in results/sota/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.wordle_v2 import WordleEnvSOTA
from src.models.sota_dqn import SOTADQNAgent, get_device


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLOR CODES
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Wordle colors
    GREEN_BG = "\033[42m\033[30m"   # Green background, black text
    YELLOW_BG = "\033[43m\033[30m"  # Yellow background, black text
    GRAY_BG = "\033[100m\033[37m"   # Gray background, white text
    
    # Text colors  
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"


def colored_letter(char: str, feedback: int) -> str:
    """Return a colored letter based on Wordle feedback."""
    c = char.upper()
    if feedback == 2:   # Green
        return f"{Colors.GREEN_BG} {c} {Colors.RESET}"
    elif feedback == 1: # Yellow
        return f"{Colors.YELLOW_BG} {c} {Colors.RESET}"
    else:               # Gray
        return f"{Colors.GRAY_BG} {c} {Colors.RESET}"


def format_guess(guess: str, feedback: List[int]) -> str:
    """Format a guess with colored feedback."""
    return " ".join(colored_letter(c, fb) for c, fb in zip(guess, feedback))


def print_header():
    """Print fancy header."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║        🎯 WORDLE RL AGENT - LIVE DEMONSTRATION 🎯            ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║                                                              ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║   Letter Decomposition DQN with Curriculum Learning         ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()


def print_divider(char="─", width=60):
    """Print a divider line."""
    print(f"{Colors.DIM}{char * width}{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class DemoAgent:
    """Wrapper for the SOTA agent with demo-friendly methods."""
    
    def __init__(self, model_path: str, phase: int = 2):
        self.device = get_device()
        print(f"{Colors.DIM}Loading model from: {model_path}{Colors.RESET}")
        print(f"{Colors.DIM}Using device: {self.device}{Colors.RESET}")
        
        # Create environment
        self.env = WordleEnvSOTA()
        if phase == 1:
            self.env.solution_words = self.env.solution_words[:100]
        elif phase == 2:
            self.env.solution_words = self.env.solution_words[:1000]
        
        # Create and load agent
        self.agent = SOTADQNAgent(
            word_list=self.env.valid_guesses,
            input_dim=313,
            hidden_dims=(512, 256),
            device=self.device,
        )
        self.agent.load(model_path)
        self.agent.policy_net.eval()
        
        print(f"{Colors.GREEN}✓ Model loaded successfully!{Colors.RESET}")
        print(f"{Colors.DIM}Solution words: {len(self.env.solution_words)}, "
              f"Valid guesses: {len(self.env.valid_guesses)}{Colors.RESET}")
    
    def get_top_candidates(self, obs: np.ndarray, mask: np.ndarray, 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k word candidates with their scores."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, 
                                device=self.device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            
            letter_q = self.agent.policy_net.get_letter_q_values(obs_t)
            scores = self.agent.word_scorer.compute_word_scores(letter_q, mask_t)
            
            top_indices = scores[0].argsort(descending=True)[:top_k]
            candidates = [
                (self.agent.word_list[idx.item()], scores[0, idx].item())
                for idx in top_indices
            ]
        
        return candidates
    
    def get_letter_insights(self, obs: np.ndarray) -> np.ndarray:
        """Get letter Q-values for insight display."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            letter_q = self.agent.policy_net.get_letter_q_values(obs_t)
        return letter_q[0].cpu().numpy()  # (5, 26)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO GAME
# ═══════════════════════════════════════════════════════════════════════════════

def play_demo_game(demo: DemoAgent, target_word: Optional[str] = None, 
                   slow_mode: bool = True):
    """Play one game with verbose commentary."""
    
    # Reset environment
    if target_word:
        if target_word.lower() not in demo.env.valid_guesses:
            print(f"{Colors.RED}Error: '{target_word}' is not a valid Wordle word!{Colors.RESET}")
            return False, 0
        obs, info = demo.env.reset()
        demo.env.target_word = target_word.lower()
    else:
        obs, info = demo.env.reset()
    
    mask = info["action_mask"]
    
    # Game header
    print_divider("═")
    print(f"{Colors.BOLD}🎯 TARGET WORD: ", end="")
    if target_word:
        print(f"{Colors.MAGENTA}{target_word.upper()}{Colors.RESET}")
    else:
        print(f"{Colors.DIM}[HIDDEN]{Colors.RESET}")
    print_divider("═")
    print()
    
    turn = 0
    guesses_display = []
    
    while True:
        turn += 1
        
        # Get agent's analysis
        candidates = demo.get_top_candidates(obs, mask, top_k=5)
        best_word = candidates[0][0]
        
        # Commentary before guess
        print(f"{Colors.BOLD}{Colors.CYAN}━━━ Turn {turn}/6 ━━━{Colors.RESET}")
        print()
        
        # Show top candidates
        print(f"  {Colors.BOLD}Agent's Top 5 Candidates:{Colors.RESET}")
        for i, (word, score) in enumerate(candidates):
            marker = f"{Colors.GREEN}▶{Colors.RESET}" if i == 0 else " "
            bar_len = int(max(0, min(20, (score + 5) * 2)))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {marker} {Colors.BOLD}{word.upper()}{Colors.RESET} "
                  f"{Colors.DIM}[{bar}]{Colors.RESET} "
                  f"score: {Colors.YELLOW}{score:+.2f}{Colors.RESET}")
        print()
        
        if slow_mode:
            time.sleep(0.8)
        
        # Make the guess
        action = demo.env.word_to_idx[best_word]
        next_obs, reward, terminated, truncated, next_info = demo.env.step(action)
        done = terminated or truncated
        feedback = demo.env.feedbacks[-1]
        
        # Show the guess result
        print(f"  {Colors.BOLD}Guess:{Colors.RESET} {format_guess(best_word, feedback)}")
        
        # Show reward
        reward_color = Colors.GREEN if reward > 0 else (Colors.RED if reward < 0 else Colors.WHITE)
        print(f"  {Colors.DIM}Reward:{Colors.RESET} {reward_color}{reward:+.2f}{Colors.RESET}")
        
        guesses_display.append(format_guess(best_word, feedback))
        print()
        
        if slow_mode:
            time.sleep(0.5)
        
        if done:
            break
        
        obs = next_obs
        mask = next_info["action_mask"]
    
    # Game result
    won = demo.env.guesses[-1] == demo.env.target_word
    
    print_divider("═")
    print()
    print(f"  {Colors.BOLD}Final Board:{Colors.RESET}")
    for g in guesses_display:
        print(f"    {g}")
    print()
    
    if won:
        print(f"  {Colors.GREEN}{Colors.BOLD}✅ SOLVED in {turn} guesses!{Colors.RESET}")
        emoji_row = "🟩" * 5
    else:
        print(f"  {Colors.RED}{Colors.BOLD}❌ FAILED! The word was: "
              f"{demo.env.target_word.upper()}{Colors.RESET}")
        emoji_row = "⬛" * 5
    
    print(f"  {Colors.DIM}Target: {demo.env.target_word.upper()}{Colors.RESET}")
    print_divider("═")
    print()
    
    return won, turn


def interactive_mode(demo: DemoAgent):
    """Let the teacher pick words interactively."""
    print_header()
    print(f"{Colors.BOLD}🎮 INTERACTIVE MODE{Colors.RESET}")
    print(f"{Colors.DIM}Type a 5-letter word to challenge the AI, or:{Colors.RESET}")
    print(f"  • 'random' - pick a random word")
    print(f"  • 'quit' or 'exit' - end demo")
    print()
    
    games_won = 0
    games_played = 0
    
    while True:
        print_divider()
        user_input = input(f"{Colors.CYAN}Enter target word: {Colors.RESET}").strip().lower()
        
        if user_input in ('quit', 'exit', 'q'):
            break
        
        if user_input == 'random':
            target = None
        elif len(user_input) != 5 or not user_input.isalpha():
            print(f"{Colors.RED}Please enter exactly 5 letters!{Colors.RESET}")
            continue
        else:
            target = user_input
        
        won, turns = play_demo_game(demo, target_word=target, slow_mode=True)
        games_played += 1
        if won:
            games_won += 1
        
        print(f"{Colors.DIM}Session stats: {games_won}/{games_played} won "
              f"({100*games_won/games_played:.0f}%){Colors.RESET}")
        print()
    
    print(f"\n{Colors.BOLD}Thanks for the demo! Final: {games_won}/{games_played} won.{Colors.RESET}\n")


def quick_benchmark(demo: DemoAgent, num_games: int = 50):
    """Run a quick benchmark."""
    print(f"\n{Colors.BOLD}📊 Quick Benchmark ({num_games} games){Colors.RESET}\n")
    
    wins = 0
    total_turns = 0
    
    for i in range(num_games):
        obs, info = demo.env.reset()
        mask = info["action_mask"]
        done = False
        
        while not done:
            candidates = demo.get_top_candidates(obs, mask, 1)
            action = demo.env.word_to_idx[candidates[0][0]]
            obs, _, terminated, truncated, info = demo.env.step(action)
            done = terminated or truncated
            mask = info["action_mask"]
        
        won = demo.env.guesses[-1] == demo.env.target_word
        if won:
            wins += 1
            total_turns += len(demo.env.guesses)
        
        # Progress
        if (i + 1) % 10 == 0:
            wr = wins / (i + 1)
            print(f"  {i+1}/{num_games}: Win Rate = {wr:.1%}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Results:{Colors.RESET}")
    print(f"  Win Rate: {wins}/{num_games} = {100*wins/num_games:.1f}%")
    print(f"  Avg Guesses (wins): {total_turns/max(wins,1):.2f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🎓 Wordle RL Agent - Professor Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_teacher.py                      # Random word demo
  python demo_teacher.py --word piano         # Try specific word
  python demo_teacher.py --interactive        # Teacher picks words
  python demo_teacher.py --benchmark 100      # Run 100-game benchmark
        """
    )
    
    parser.add_argument("--model", type=str, 
                        default=str(PROJECT_ROOT / "results/sota/phase2_final_model.pt"),
                        help="Path to trained model")
    parser.add_argument("--word", type=str, default=None,
                        help="Specific target word to guess")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: teacher picks words")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Run N-game benchmark")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3],
                        help="Curriculum phase (affects word set)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip delays for fast output")
    
    args = parser.parse_args()
    
    print_header()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"{Colors.RED}Error: Model not found at {args.model}{Colors.RESET}")
        print(f"{Colors.DIM}Available models:{Colors.RESET}")
        sota_dir = PROJECT_ROOT / "results/sota"
        if sota_dir.exists():
            for f in sorted(sota_dir.glob("*.pt"))[:5]:
                print(f"  - {f}")
        sys.exit(1)
    
    # Load agent
    demo = DemoAgent(args.model, phase=args.phase)
    print()
    
    # Run appropriate mode
    if args.interactive:
        interactive_mode(demo)
    elif args.benchmark > 0:
        quick_benchmark(demo, args.benchmark)
    else:
        play_demo_game(demo, target_word=args.word, slow_mode=not args.fast)


if __name__ == "__main__":
    main()
