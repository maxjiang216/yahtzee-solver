#!/usr/bin/env python3
"""
Analyze reachable (mask, upper) combinations in Yahtzee

Upper category score ranges:
  Ones (bit 0):   0-5  (0,1,2,3,4,5 ones × 1)
  Twos (bit 1):   0-10 (0,1,2,3,4,5 twos × 2)
  Threes (bit 2): 0-15 (0,1,2,3,4,5 threes × 3)
  Fours (bit 3):  0-20 (0,1,2,3,4,5 fours × 4)
  Fives (bit 4):  0-25 (0,1,2,3,4,5 fives × 5)
  Sixes (bit 5):  0-30 (0,1,2,3,4,5 sixes × 6)
"""

from itertools import product

# Possible scores for each upper category (0-5)
possible_scores = [
    [0, 1, 2, 3, 4, 5],       # Ones: 0-5
    [0, 2, 4, 6, 8, 10],      # Twos: 0-10
    [0, 3, 6, 9, 12, 15],     # Threes: 0-15
    [0, 4, 8, 12, 16, 20],    # Fours: 0-20
    [0, 5, 10, 15, 20, 25],   # Fives: 0-25
    [0, 6, 12, 18, 24, 30],   # Sixes: 0-30
]

print("=== Analyzing Reachable (mask, upper) Combinations ===\n")

for c in range(6):
    print(f"Category {c} (×{c+1}): {possible_scores[c]}")
print()

# For each mask, find all possible upper values (capped at 63)
total_reachable = 0
total_reachable_with_ystate = 0
mask_to_uppers = {}

for mask in range(4096):
    # Find which upper categories (bits 0-5) are filled
    filled = [c for c in range(6) if (mask >> c) & 1]
    
    if not filled:
        # No upper categories filled -> upper = 0
        reachable = {0}
    else:
        # Enumerate all combinations of scores for filled categories
        score_options = [possible_scores[c] for c in filled]
        reachable = set()
        for combo in product(*score_options):
            upper = min(63, sum(combo))  # Cap at 63
            reachable.add(upper)
    
    mask_to_uppers[mask] = reachable
    total_reachable += len(reachable)
    total_reachable_with_ystate += len(reachable) * 3  # × 3 for y_state

# Stats
theoretical = 4096 * 64 * 3
print(f"Theoretical states (mask × upper × y_state): {theoretical:,}")
print(f"Reachable states: {total_reachable_with_ystate:,}")
print(f"Reduction ratio: {total_reachable_with_ystate / theoretical:.2%}")
print(f"Wasted space in current storage: {(1 - total_reachable_with_ystate / theoretical):.2%}")
print()

# Distribution of reachable upper values per mask
upper_counts = [len(v) for v in mask_to_uppers.values()]
print(f"Reachable upper values per mask:")
print(f"  Min: {min(upper_counts)}")
print(f"  Max: {max(upper_counts)}")
print(f"  Avg: {sum(upper_counts)/len(upper_counts):.1f}")
print()

# Show some examples
print("Examples:")
for mask in [0, 1, 3, 7, 15, 31, 63, 0b111111]:  # Various upper fill patterns
    uppers = sorted(mask_to_uppers[mask])
    upper_bits = mask & 0b111111
    print(f"  mask={mask:4d} (upper bits: {upper_bits:06b}): {len(uppers):2d} values, range [{min(uppers)}-{max(uppers)}]")

print()

# Memory analysis
print("=== Memory Analysis ===")
print()

# Current storage
current_board = 4096 * 64 * 3
current_roll = current_board * 252
print(f"Current storage (all mask × upper × y_state):")
print(f"  Board states: {current_board:,} ({current_board * 4 / 1024 / 1024:.1f} MB)")
print(f"  Roll states:  {current_roll:,} ({current_roll * 4 / 1024 / 1024:.1f} MB × 2)")
print()

# Optimized storage
opt_board = total_reachable_with_ystate
opt_roll = opt_board * 252
print(f"Optimized storage (only reachable states):")
print(f"  Board states: {opt_board:,} ({opt_board * 4 / 1024 / 1024:.1f} MB)")
print(f"  Roll states:  {opt_roll:,} ({opt_roll * 4 / 1024 / 1024:.1f} MB × 2)")
print()

savings = (current_board + 2 * current_roll - opt_board - 2 * opt_roll) * 4 / 1024 / 1024
print(f"Potential savings: {savings:.1f} MB")
print()

# For PMF storage
print("=== PMF Storage Analysis ===")
SCORE_RANGE = 501  # 0-500
print(f"Score range: 0-{SCORE_RANGE-1}")
print()

pmf_current = current_board * SCORE_RANGE
pmf_optimized = opt_board * SCORE_RANGE
print(f"PMF storage (current indexing):   {pmf_current:,} ({pmf_current * 4 / 1024 / 1024 / 1024:.2f} GB)")
print(f"PMF storage (optimized indexing): {pmf_optimized:,} ({pmf_optimized * 4 / 1024 / 1024 / 1024:.2f} GB)")
print(f"PMF savings: {(pmf_current - pmf_optimized) * 4 / 1024 / 1024:.1f} MB")