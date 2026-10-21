# Yahtzee PMF Solver — Technical Notes

## What It Does

`yahtzee_pmf_solver.cpp` computes, for every reachable board state, the exact probability distribution of the remaining score under the EV-optimal solo policy. Everything is done in exact rational arithmetic; floats appear only when results are written out.

Formally: for each reachable state `(mask, upper, y_state)`, the PMF gives `P(future_score = s)` for every `s` from 0 to 1575 (the theoretical maximum), measured from this state forward.

The 2-player NN uses the table as a near-optimal win-probability feature (P_baseline).

---

## Output Files

**`yahtzee_pmf_v2.bin`** (1.1 GB), read by `python/yahtzee_2player_env.py` (`load_pmf_table`):

| Part | Contents |
|------|----------|
| Header (16 bytes) | `magic=0x504D4659` ("PMFY"), `version=2`, `num_states=786432`, `dense_size=1576` |
| Index (8 bytes per state) | `uint32 offset`, `int16 lo`, `int16 width`: the state's probabilities for scores `lo .. lo+width-1` are floats `offset .. offset+width-1` of the data section. `width = 0` means the state is unreachable. |
| Data | float32 probabilities, each the exact value rounded to float |

Only each state's nonzero score range is stored, and unreachable states take no data. The old version 1 layout, with a dense 1576-float row for every state (4.7 GB), is still readable by `load_pmf_table`.

**`yahtzee_pmf_exact_start.txt`**: the exact distribution of a whole game (start state), one `score numerator` line per score, with every probability equal to `numerator / 6^195`.

**Start state summary:**

| | |
|---|---|
| Mean | 254.587728734495891231825246064916… (equals the exact EV) |
| Score range | 12 to 1575 |
| Most likely score | 266 (P ≈ 0.0125799) |
| P(score ≥ 200 / 250 / 300 / 400 / 500) | 0.86357 / 0.48369 / 0.14326 / 0.03834 / 0.00719 |

---

## Score Ceiling

1575 covers every possible game outcome. The maximum occurs when all 13 turns produce a Yahtzee and joker scoring is optimal:

| Component | Points |
|-----------|--------|
| First Yahtzee (fills box) | 50 |
| 6 jokers → upper section (5×face each) | 105 |
| Upper bonus (sum ≥ 63) | 35 |
| 6 jokers → lower section (3ok/4ok/FH/SS/LS/Chance) | 185 |
| 12 × Yahtzee bonus (100 each) | 1,200 |
| **Total** | **1,575** |

---

## Algorithm

Standard reverse induction over layers of equal turns remaining: every turn fills exactly one box, so a state's successors are all in the layer solved just before it.

### 1. Exact policy

With *k* turns left, every EV and every probability is an integer over `D_k = 6^(15k)`: each turn rolls at most 15 dice, each contributing a factor of 1/6, and `max` only selects one of its arguments. The solver keeps integer numerators in fixed-width limbs (`yahtzee_bigint.h`). For each state, `ExactTurnSolver` (`yahtzee_exact_turn.h`) evaluates the turn exactly: the best scoring choice for each final roll, then the best keep with one and with two rolls left, using the keep lattice. It records the policy.

Because EVs are exact, near-ties are never misjudged by rounding. Exact ties leave the EV unchanged but can change the distribution, so they are broken by fixed rules:
- **Scoring:** take the first best option in `ScoringRules` order.
- **Keeps:** keep more dice, then take the lowest subset id.

### 2. Final-roll probabilities

With the policy fixed, the probability of each final roll is an integer over `6^15 < 2^39`. It is found by pushing integer weights through the keep lattice twice: deposit each outcome's weight on its chosen keep, scaled by `6^|K|`, then add every subset's weight into its six children. Each push takes 1,260 additions.

### 3. Distribution

Final rolls leading to the same (points `r`, next board `n`) are merged. Each state has about 35 such pairs on average. Then

```
PMF_k[s] = Σ over pairs (r, n, w) of  w × PMF_{k-1}[n][s − r]
```

This is one big-integer multiply-add by a 64-bit weight per (pair, score), with no matrix products and no per-roll passes over distributions.

### 4. Exact checks

For every state, the probabilities must sum to exactly `D_k` (probability 1), and `Σ s × numerator(s)` must equal the exact EV numerator. Any mismatch aborts the run.

### Memory

Only the previous layer's exact distributions are needed. Each is stored over its score range only, and each probability with only the limbs it needs: one byte per score holds the limb count, followed by the limbs.

- **Default:** both live layers stay in RAM.
- **`--low-memory`:** each finished layer goes to a temporary file (`<output>.layer0/1`), and the previous layer is read through a read-only memory map. Resident memory then stays around one layer, and the OS can drop those clean pages under pressure.

The float output is written straight to the output file as each state finishes.

---

## Performance

Measured on 4 cores, `g++ -O3 -march=native -fopenmp`:

| | Old float solver | Exact solver | Exact, `--low-memory` |
|---|---|---|---|
| Wall time | 2,259 s | **~37 s** | ~43 s |
| Peak RSS | 4.8 GB | 4.0 GB | **2.2 GB** |
| Output | 4.7 GB (dense rows) | 1.1 GB (v2) | 1.1 GB (v2) |
| Arithmetic | float32 | exact | exact |

The old solver's time was measured on the same machine, with other runs sharing the CPU for part of it.

Both exact modes produce byte-identical output, and repeated runs are byte-for-byte reproducible. Compared with the old float table, the mean absolute difference per state is 1.3e-5. About 2% of states differ by more than 1e-4, up to 2.7e-3. There the means agree to within float precision (≤ 9e-7 relative), so these are ties or near-ties that the float solver decided differently.

For comparison, the previous float PMF solver held the dense 4.7 GB table in RAM, and its documented runtime was ~9 minutes on 15 threads. It decided keeps from float EVs, so near-ties could be misjudged, and it computed every state, including unreachable ones.

Per-layer breakdown (default mode):

| Categories filled | States | Time (ms) | Exact layer size |
|-------------------|--------|-----------|------------------|
| 12 | 1,598 | 14 | <1 MB |
| 11 | 9,135 | 102 | 7 MB |
| 10 | 31,322 | 496 | 61 MB |
| 9 | 71,237 | 1,342 | 377 MB |
| 8 | 112,596 | 3,544 | 923 MB |
| 7 | 126,219 | 6,516 | 1,856 MB |
| **6 (peak)** | **100,619** | **10,021** | **2,084 MB** |
| 5 | 56,283 | 5,523 | 1,599 MB |
| 4 | 21,377 | 2,514 | 827 MB |
| 3 | 5,178 | 825 | 248 MB |
| 2 | 711 | 144 | 44 MB |
| 1 | 44 | 16 | 3 MB |
| 0 | 1 | 0 | <1 MB |

Profile (thread time): the exact turn solve and the multiply-adds into distributions take roughly equal shares; the rest is packing, float conversion and output writes.

---

## Loading (Python)

```python
from yahtzee_2player_env import load_pmf_table, compute_p_baseline

pmf = load_pmf_table('yahtzee_pmf_v2.bin')
# pmf.shape == (786432, 1576); pmf[board_index] is a dense float32 row

p_win = compute_p_baseline(mask_cur, upper_cur, y_cur,
                            mask_opp, upper_opp, y_opp,
                            score_diff, pmf)
```

`compute_p_baseline` computes `Σ_r PMF_cur[r] × CDF_opp[r + diff]` in O(dense_size) time.
