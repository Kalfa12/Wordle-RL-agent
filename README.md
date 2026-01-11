# Wordle RL Agent

**State-of-the-Art Wordle Solver using Deep Reinforcement Learning**

A reinforcement learning project demonstrating the **Letter Decomposition** architecture to solve Wordle, achieving up to 92% win rate on training subsets.

## 🎯 Key Innovation

Instead of 13,000 Q-values (one per word), we output **130 Q-values** (26 letters × 5 positions) and compute word scores as the sum of letter Q-values:

```
Score(CRANE) = Q(C, pos=0) + Q(R, pos=1) + Q(A, pos=2) + Q(N, pos=3) + Q(E, pos=4)
```

This reduces output complexity by 99% and enables compositional learning.

## 📊 Results

| Model | Win Rate | Peak | Description |
|-------|----------|------|-------------|
| Naive DQN | 0% | 1% | 13,000 outputs - failed to learn |
| SOTA Phase 1 | 73% | **92%** | 100 solution words |
| SOTA Phase 2 | 45% | **65%** | 1,000 solution words |
| SOTA Phase 3 | 35% | 36% | Full 2,315 words |

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Demo
```bash
# Interactive demo
python presentation/demo_teacher.py --interactive

# Specific word
python presentation/demo_teacher.py --word crane

# Benchmark
python presentation/demo_teacher.py --benchmark 50
```

### Train Model
```bash
# Phase 1: 100 words
python train_curriculum.py --phase 1 --episodes 10000

# Phase 2: 1000 words (with transfer)
python train_curriculum.py --phase 2 --episodes 10000 \
    --load-weights results/sota/phase1_final_model.pt

# Phase 3: Full vocabulary
python train_curriculum.py --phase 3 --episodes 100000 \
    --load-weights results/sota/phase2_final_model.pt --continue
```

## 📁 Project Structure

```
├── src/
│   ├── envs/
│   │   └── wordle_v2.py          # 313-dim observation environment
│   ├── models/
│   │   └── sota_dqn.py           # Letter Decomposition DQN
│   └── utils/
│       └── sota_replay.py        # Experience replay buffer
├── train_curriculum.py           # Main training script
├── evaluate_agent.py             # Model evaluation
├── presentation/
│   ├── demo_teacher.py           # Interactive demo
├── results/sota/                 # Trained models & logs
├── wordle-La.txt                 # 2,315 solution words
└── wordle-Ta.txt                 # 10,657 valid guess words
```

## 🔬 Technical Details

### Observation Space (313 dimensions)
- `turns_remaining`: 1 dim
- `letters_used`: 26 dims (binary)
- `letters_green_anywhere`: 26 dims
- `letters_green_at_pos`: 130 dims (5 × 26)
- `letters_yellow_or_gray_at_pos`: 130 dims (5 × 26)

### Reward Function
| Event | Reward |
|-------|--------|
| Win | +10.0 |
| Loss | -10.0 |
| Step | -0.1 |
| New green letter | +0.5 |

### Algorithm
- DQN with Experience Replay
- Target network (soft updates)
- ε-greedy exploration with decay
- Action masking for constraint satisfaction

## 📚 References

- Mnih et al. (2015) - *Human-level control through deep reinforcement learning*
- Bengio et al. (2009) - *Curriculum learning*

## 👤 Author

Taha El Younsi - December 2024
