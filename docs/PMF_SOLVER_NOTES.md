# Yahtzee PMF Solver — Technical Notes

## What It Does

`yahtzee_pmf_solver.cpp` computes, for every reachable Yahtzee board state, the full probability distribution of remaining score under optimal solo play. The output is `yahtzee_pmf_v1.bin`, which is used by the 2-player NN as a near-optimal win-probability feature (P_baseline).

Formally: for each state `(mask, upper, y_state)`, the PMF gives `P(future_score = s)` for all `s` from 0 to 1575 (the exact theoretical maximum), where score is measured from this state forward, and optimal play is defined by `yahtzee_optimal_v4.bin`.

---

## Output File

| Property | Value |
|----------|-------|
| File | `yahtzee_pmf_v1.bin` |
| Size | 4.7 GB |
| States | 786,432 (mask × upper × y_state) |
| Scores per state | 1,576 (0–1575 inclusive) |
| Format | 16-byte header + raw float32 array |
| Header | `magic=0x504D4659`, `version=1`, `num_states`, `dense_size` |

**Verification at starting state (mask=0, upper=0, y_state=0):**
- PMF expected value: **254.588** (exact match to EV solver)
- Total probability mass: **1.000000** (exact — no truncation)

---

## Score Ceiling

`DENSE_MAX = 1575` covers every possible game outcome. The maximum occurs when all 13 turns produce a Yahtzee and joker scoring is optimal:

| Component | Points |
|-----------|--------|
| First Yahtzee (fills box) | 50 |
| 6 jokers → upper section (5×face each) | 105 |
| Upper bonus (sum ≥ 63) | 35 |
| 6 jokers → lower section (3ok/4ok/FH/SS/LS/Chance) | 185 |
| 12 × Yahtzee bonus (100 each) | 1,200 |
| **Total** | **1,575** |

P(all 13 Yahtzees) ≈ (4.6%)^13 ≈ 10⁻²⁴ — astronomically rare, but covering it costs nothing vs. DENSE_MAX = 800.

---

## Algorithm

Standard reverse induction. States are processed from `total_filled = 13` down to `0`. For each board state, optimal keep decisions and scoring choices are derived from the preloaded start-of-turn EVs (`board_ev_` from `yahtzee_optimal_v4.bin`; a version 3 file also works, and its mid-turn tables are ignored).

### Per-state computation — three optimizations

**1. Keep-decision precomputation (dot products over subsets)**

The naive implementation called `GetOptimalScoringMove` inside `GetOptimalKeep(..., rolls_left=1)`, resulting in ~800K scoring evaluations per state. The fix: precompute `term_ev[d] = pts[d] + board_ev[next[d]]` for all 252 terminal outcomes, then compute expected value for every subset as a single dot product:

```
subset_ev[k] = Σ_d  trans_probs[k][d] × term_ev[d]      (for keep2)
roll2_ev[d] = max_{k ∈ valid_subsets(d)} subset_ev[k]
subset_r2_ev[k] = Σ_d  trans_probs[k][d] × roll2_ev[d]  (for keep1)
```

Then `keep_for[d] = argmax_{k ∈ valid_subsets(d)} subset_ev[k]`. `roll2_ev` is recomputed per state this way rather than loaded, so the EV file only needs start-of-turn values. Total: 462 × 252 = 116K multiplications vs. ~800K scoring calls. Same pattern as the booty-dice solver's backward EV propagation.

**2. Three-pass PMF accumulation (matrix products)**

The original accumulation used a 252³ = 16M iteration triple loop with a 1,576-element inner loop per iteration — O(16M × 1576) ≈ 25 billion operations per state.

Factoring out the three rolls into sequential passes:

```
Pass 0:  shifted[r3][s] = board_pmf[next[r3]].dense[s − pts[r3]]
Pass 1:  pmf2[r2][s]   = Σ_r3  trans[keep2[r2]][r3] × shifted[r3][s]
Pass 2:  pmf1[r1][s]   = Σ_r2  trans[keep1[r1]][r2] × pmf2[r2][s]
Pass 3:  result[s]     = Σ_r1  init_probs[r1]        × pmf1[r1][s]
```

Each pass is O(252 × ~10_nonzero × 1576) — DAXPY-style, SIMD-vectorizable, cache-friendly. Per-state work: ~4M multiply-adds vs. 25B iterations. Thread-private 404 KB heap buffers per pass avoid stack overflow under OpenMP.

This is the same factorization the booty-dice solver uses for its within-turn EV propagation (Steps 2–6 in its `compute_turn_value`), applied here to full PMF vectors instead of scalars.

**3. OpenMP parallelization**

States at each fill level are fully independent (the PMF of a state depends only on already-computed deeper states). `#pragma omp parallel` with `schedule(dynamic, 4)` spreads work across all available cores.

---

## Runtime

Hardware: 15-thread x86-64 Linux, g++ -O3 -march=native -fopenmp

| DENSE_MAX | Memory (PMF table) | Runtime | PMF total prob |
|-----------|-------------------|---------|----------------|
| 400 (original) | 1.2 GB | ~17 hours (estimated, killed) | 0.962 (truncated) |
| 400 (with dot-product fix only) | 1.2 GB | ~126s/level-12 → ~17h | 0.962 |
| 400 (three-pass) | 1.2 GB | 87s | 0.962 |
| 800 (three-pass) | 2.4 GB | 195s | 0.999997 |
| **1575 (three-pass, final)** | **4.7 GB** | **536s (~9 min)** | **1.000000** |

The three-pass restructuring gives a ~700× wall-clock speedup over the naive triple loop. DENSE_MAX = 1575 adds ~2.7× more time than DENSE_MAX = 400 (linear in DENSE_SIZE), which is acceptable since the computation runs once.

Per-level breakdown (DENSE_MAX = 1575):

| Categories filled | States | Time (ms) |
|-------------------|--------|-----------|
| 13 | 128 | 14 |
| 12 | 1,600 | 1,965 |
| 11 | 9,216 | 8,385 |
| 10 | 32,384 | 25,289 |
| 9 | 77,440 | 55,688 |
| 8 | 133,056 | 90,620 |
| **7 (peak)** | **168,960** | **113,773** |
| 6 | 160,512 | 105,211 |
| 5 | 114,048 | 73,812 |
| 4 | 59,840 | 37,689 |
| 3 | 22,528 | 13,734 |
| 2 | 5,760 | 3,577 |
| 1 | 896 | 589 |
| 0 | 64 | 60 |

---

## Loading (Python)

```python
from yahtzee_2player_env import load_pmf_table, compute_p_baseline

pmf_dense = load_pmf_table('yahtzee_pmf_v1.bin')
# pmf_dense.shape = (786432, 1576), dtype float32, memory-mapped

p_win = compute_p_baseline(mask_cur, upper_cur, y_cur,
                            mask_opp, upper_opp, y_opp,
                            score_diff, pmf_dense)
```

`compute_p_baseline` computes `Σ_r PMF_cur[r] × CDF_opp[r + diff]` in O(dense_size) time. `dense_size` is read from the file header — no hardcoded constant.
