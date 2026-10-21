# Yahtzee Solver

Optimal solo Yahtzee solver in C++ with a 2-player neural network extension.

## What This Does

**Solo EV solver (C++):** Computes the optimal strategy by reverse-induction dynamic programming over all board states, and stores only the expected value at the start of every turn:

- `yahtzee_optimal_v4.bin` (3 MB): every mid-turn decision (which dice to keep, where to score) is recomputed from it on demand in well under a millisecond, so nothing else is stored.

**Exact rational solvers (C++):** No floating point anywhere in the DP:

- `yahtzee_rational_solver` computes every board EV exactly (~5 s), writing `yahtzee_rational_v1.bin` (54 MB) and printing the optimal EV as a reduced fraction.
- `yahtzee_pmf_solver` computes the exact distribution of the remaining score for every reachable board state under the EV-optimal policy. It writes `yahtzee_pmf_v2.bin` (each state's score range as floats, read by the Python code) and `yahtzee_pmf_exact_start.txt` (the exact distribution of a whole game).

**2-player solver (Python + PyTorch):** Neural network that estimates P(current player wins | both play optimally) for any pair of board states and accumulated score difference. Trained via supervised bootstrap on the PMF table, then refined with TD self-play RL.

## Project Structure

```
cpp/
  yahtzee_rules.h                  constants, dice indexing, scoring rules (shared)
  yahtzee_keep_lattice.h           within-turn keep expectations (shared)
  yahtzee_bigint.h                 fixed-width big integers (exact solvers)
  yahtzee_exact_turn.h             exact evaluation of one turn (exact solvers)
  yahtzee_optimized_solver.cpp     float EV solver → yahtzee_optimal_v4.bin
  yahtzee_rational_solver.cpp      exact EV solver → yahtzee_rational_v1.bin
  yahtzee_pmf_solver.cpp           exact PMF solver → yahtzee_pmf_v2.bin
  yahtzee_strategy.{h,cpp}         strategy queries (keep / category) from board EVs
  yahtzee_guide.cpp                interactive solo guide
  test_strategy.cpp                strategy table tests
  test_keep_lattice.cpp            keep lattice vs brute-force enumeration

python/
  yahtzee_2player_env.py           game environment, PMF loader, P_baseline
  yahtzee_2player_nn.py            PyTorch model (3×256 LayerNorm MLP)
  yahtzee_2player_train.py         supervised bootstrap + TD self-play training
  yahtzee_2player_eval.py          evaluation and diagnostics

docs/
  PMF_SOLVER_NOTES.md              PMF solver algorithm and performance notes
```

The data files are not in the repo; build them locally with the steps below.

## Prerequisites

**C++:** g++ with C++17 and OpenMP.

**Python (2-player NN):**
```bash
pip install -r requirements.txt
```

## Quickstart

```bash
make all          # build every program
make solve        # yahtzee_optimal_v4.bin (3 MB, ~4 s)
make test         # unit tests
make pmf-solve    # yahtzee_pmf_v2.bin + yahtzee_pmf_exact_start.txt (~40 s; pmf-solve-low for ~2 GB RAM)
make rational-solve   # optional: exact EV table and fraction

./yahtzee_guide yahtzee_optimal_v4.bin   # interactive guide
```

Train the 2-player NN:

```bash
# Phase 1: supervised bootstrap (P_baseline labels)
python python/yahtzee_2player_train.py --phase 2 --games 50000

# Phase 2: TD self-play RL
python python/yahtzee_2player_train.py --phase 3 --load checkpoints_2player/phase2_bootstrap.pt
```

## Results

| Metric | Value |
|--------|-------|
| Optimal solo EV | **254.58772873449589…** = N / (2^169 · 3^181), N has 140 digits |
| Most likely final score | 266 (P ≈ 1.26%) |
| P(score ≥ 200 / 250 / 300) | 86.36% / 48.37% / 14.33% |
| Final score range | 12 to 1575 |
| EV solver | ~4 s, 11 MB RAM, 3 MB output |
| Exact PMF solver | ~37 s on 4 cores, 4.0 GB peak RSS (2.2 GB with `--low-memory`), 1.1 GB output |

## How It Works

**Board-only storage.** The EV of a mid-turn position follows from the board EVs of the possible next states in two lattice passes, so only the 786K start-of-turn values are stored.

**Keep lattice.** Rerolling the non-kept dice is the same as rolling them one at a time, so `E[K] = (1/6) Σ_f E[K ∪ {f}]` over the 462 kept multisets. One pass gives the value of every keep (1,260 additions) instead of a 462×252 matrix product.

**Exact arithmetic.** Every probability in a turn is a product of 1/6 factors, and `max` only selects one of its arguments, so with *k* turns left every EV and every probability is an integer over the fixed denominator 6^(15k). The exact solvers work on integer numerators only: big-integer addition, multiplication by small constants, and integer comparison. They need no gcd and no fraction comparisons. Numerators fit in at most 9 × 64-bit limbs, and each layer is compiled with the fewest limbs it needs.

**Exact PMF.** With the turn's policy fixed by exact EVs, each final roll's probability is an integer over 6^15 found by pushing weights through the keep lattice, and a state's distribution is a weighted sum of its successors' distributions. Every state is checked exactly (probabilities sum to 1, mean equals the exact EV). See [docs/PMF_SOLVER_NOTES.md](docs/PMF_SOLVER_NOTES.md).

## 2-Player Solver Design

The 2-player state space (~95 billion board-pair states × ~1000 score differences) is too large for exact DP. The NN approximates P(win) using:

- Both players' board state encoded as 36-dimensional feature vectors
- **P_baseline** — an O(1576) PMF convolution giving the win probability under independent solo-optimal play, used as both a training label and model input feature

The symmetry `V(cur, opp, diff) = 1 − V(opp, cur, −diff)` is enforced at training time.
