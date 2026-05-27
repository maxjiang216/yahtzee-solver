# Yahtzee Solver

Optimal solo Yahtzee solver in C++ with a 2-player neural network extension.

## What This Does

**Solo solver (C++):** Computes the exact optimal strategy via reverse-induction dynamic programming over ~536K reachable board states. Produces two precomputed tables:

- `yahtzee_optimal_v3.bin` (1.5 GB) — expected value and optimal keep/score decisions for every board state
- `yahtzee_pmf_v1.bin` (4.7 GB) — full probability distribution over remaining scores for every board state

**2-player solver (Python + PyTorch):** Neural network that estimates P(current player wins | both play optimally) for any pair of board states and accumulated score difference. Trained via supervised bootstrap on the PMF table, then refined with TD self-play RL.

## Project Structure

```
cpp/
  yahtzee_strategy.{h,cpp}         shared board state, scoring, dice transitions
  yahtzee_optimized_solver.cpp     solo EV solver → yahtzee_optimal_v3.bin
  yahtzee_pmf_solver.cpp           PMF table solver → yahtzee_pmf_v1.bin
  yahtzee_guide_v3.cpp             interactive solo guide
  test_{strategy,probabilities}.cpp

python/
  yahtzee_2player_env.py           game environment, transitions, PMF loader, P_baseline
  yahtzee_2player_nn.py            PyTorch model (3×256 LayerNorm MLP)
  yahtzee_2player_train.py         supervised bootstrap + TD self-play training
  yahtzee_2player_eval.py          evaluation and diagnostics
  analyze_reachable.py             state space analysis

docs/
  PMF_SOLVER_NOTES.md              PMF solver algorithm and performance notes
```

The `.bin` tables are not in the repo (too large); build them locally with the steps below.

## Prerequisites

**C++ solver:** g++ with C++17 and OpenMP.

**Python (2-player NN):**
```bash
pip install -r requirements.txt
```

## Quickstart

### 1. Build and run the solo EV solver

```bash
make solver && make solve
# Produces yahtzee_optimal_v3.bin (~1.5 GB, takes ~5–30 min)
```

### 2. Generate the PMF table

```bash
make pmf && make pmf-solve
# Produces yahtzee_pmf_v1.bin (~4.7 GB, takes ~9 min on 15 cores)
```

### 3. Interactive solo guide

```bash
make guide
./yahtzee_guide yahtzee_optimal_v3.bin
```

### 4. Train the 2-player NN

```bash
# Phase 1: supervised bootstrap (P_baseline labels)
python python/yahtzee_2player_train.py --phase 2 --games 50000

# Phase 2: TD self-play RL
python python/yahtzee_2player_train.py --phase 3 --load checkpoints_2player/phase2_bootstrap.pt
```

## Results

| Metric | Value |
|--------|-------|
| Solo EV at start | **254.588** |
| PMF total probability mass | **1.000000** (exact, no truncation) |
| PMF score range | 0–1575 (covers all 13-Yahtzee games) |
| PMF solver runtime | ~9 min on 15 cores |
| PMF solver speedup | ~700× vs naive triple-loop |

The PMF solver uses a three-pass matrix factorization that factors the 252³ nested loop into three sequential DAXPY passes — see [docs/PMF_SOLVER_NOTES.md](docs/PMF_SOLVER_NOTES.md) for details.

## 2-Player Solver Design

The 2-player state space (~95 billion board-pair states × ~1000 score differences) is too large for exact DP. The NN approximates P(win) using:

- Both players' board state encoded as 36-dimensional feature vectors
- **P_baseline** — an O(1576) PMF convolution giving the win probability under independent solo-optimal play, used as both a training label and model input feature

The symmetry `V(cur, opp, diff) = 1 − V(opp, cur, −diff)` is enforced at training time.
