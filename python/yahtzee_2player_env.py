"""
2-player Yahtzee environment.

Mirrors the C++ solver's dice indexer, transition matrix, and scoring logic exactly
so that Python training matches C++ inference.

All board states are (mask, upper, y_state):
  mask   : 12-bit int, bit i=1 means category i is filled (categories 0-11)
  upper  : int 0-63, upper section sum capped for bonus tracking
  y_state: 0=yahtzee open, 1=filled with 0, 2=filled with 50
"""

import struct
import numpy as np
from functools import lru_cache
from itertools import combinations_with_replacement, product

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_DICE_COMBOS = 252
NUM_SUBSETS = 462      # all multisets of 0-5 dice from {1..6}
BOARD_STATE_SIZE = 4096 * 64 * 3  # 786,432
DENSE_MAX = 1575
DENSE_SIZE = 1576
CAT_YAHTZEE = 12
MAX_UPPER = 64

# ---------------------------------------------------------------------------
# Dice / subset indexing  (matches C++ DiceIndexer exactly)
# ---------------------------------------------------------------------------

# 252 sorted 5-dice multisets from {1..6}, lexicographic order
DICE_COMBOS = list(combinations_with_replacement(range(1, 7), 5))
DICE_TO_ID = {combo: i for i, combo in enumerate(DICE_COMBOS)}

# 462 subsets: all multisets of 0-5 dice, grouped by size
ALL_SUBSETS = []
for k in range(6):
    for combo in combinations_with_replacement(range(1, 7), k):
        ALL_SUBSETS.append(combo)
SUBSET_TO_ID = {s: i for i, s in enumerate(ALL_SUBSETS)}

# For each of 252 dice outcomes, which subset_ids can be kept?
VALID_SUBSETS_FOR_OUTCOME = []
for d, dice in enumerate(DICE_COMBOS):
    sids = set()
    for m in range(32):   # 2^5 keep bitmasks
        kept = tuple(sorted(dice[b] for b in range(5) if (m >> b) & 1))
        sids.add(SUBSET_TO_ID[kept])
    VALID_SUBSETS_FOR_OUTCOME.append(sorted(sids))

# Same lists as a (252, 32) array, each row padded with its first entry, so the
# best keep of every outcome can be taken with one vectorized max
VALID_SUBSETS_PADDED = np.array(
    [s + [s[0]] * (32 - len(s)) for s in VALID_SUBSETS_FOR_OUTCOME], dtype=np.int64)


def get_dice_id(dice):
    return DICE_TO_ID[tuple(sorted(dice))]


def get_subset_id(dice):
    return SUBSET_TO_ID[tuple(sorted(dice))]


def board_index(mask, upper, y_state):
    return (mask * MAX_UPPER + upper) * 3 + y_state


def cats_filled(mask, y_state):
    return bin(mask).count('1') + (1 if y_state > 0 else 0)

# ---------------------------------------------------------------------------
# Transition matrix  (matches C++ BuildTransitionMatrix)
# ---------------------------------------------------------------------------

def build_transition_matrix():
    """
    trans[subset_id, dice_id] = P(roll yields dice_id | kept subset).
    Shape: (462, 252), float32.
    """
    trans = np.zeros((NUM_SUBSETS, NUM_DICE_COMBOS), dtype=np.float32)
    for i, subset in enumerate(ALL_SUBSETS):
        needed = 5 - len(subset)
        rolls = list(product(range(1, 7), repeat=needed))
        for roll in rolls:
            combo = tuple(sorted(subset + roll))
            trans[i, DICE_TO_ID[combo]] += 1.0
        if rolls:
            trans[i] /= len(rolls)
        else:
            # Keep all 5 dice: deterministic outcome
            trans[i, DICE_TO_ID[subset]] = 1.0
    return trans


# Built once at import time (~0.1 seconds)
TRANS_PROBS = build_transition_matrix()

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_score(category, dice):
    """Score for a given category and dice tuple (sorted)."""
    if category < 6:
        return sum(d for d in dice if d == category + 1)
    counts = [dice.count(v) for v in range(1, 7)]
    max_count = max(counts)
    s = sum(dice)
    if category == 6:   return s if max_count >= 3 else 0      # 3-of-a-kind
    if category == 7:   return s if max_count >= 4 else 0      # 4-of-a-kind
    if category == 8:                                           # full house
        c = sorted(x for x in counts if x)
        return 25 if c in ([2, 3], [5]) else 0
    if category == 9:   # small straight
        u = set(dice)
        return 30 if any(all(v in u for v in range(r, r+4)) for r in [1,2,3]) else 0
    if category == 10:  # large straight
        u = set(dice)
        return 40 if u >= {1,2,3,4,5} or u >= {2,3,4,5,6} else 0
    if category == 11:  return s                                # chance
    if category == 12:  return 50 if max_count == 5 else 0     # yahtzee
    return 0


def get_best_scoring_move(dice_id, mask, upper, y_state, board_ev):
    """
    Given dice_id and board state, return the EV-maximizing scoring choice.
    Returns (category, points_scored, next_mask, next_upper, next_y_state).
    points_scored includes upper bonus and yahtzee bonus if triggered.
    """
    dice = DICE_COMBOS[dice_id]
    is_ytz = len(set(dice)) == 1
    ytz_num = dice[0] if is_ytz else 0

    best_val = -1e30
    best = None

    def try_move(cat, pts, nm, nu, ny):
        nonlocal best_val, best
        val = pts + board_ev[board_index(nm, nu, ny)]
        if val > best_val:
            best_val = val
            best = (cat, pts, nm, nu, ny)

    if is_ytz:
        if y_state == 0:
            # Mandatory: fill yahtzee box with 50
            return (CAT_YAHTZEE, 50, mask, upper, 2)

        bonus = 100 if y_state == 2 else 0
        upper_idx = ytz_num - 1
        upper_open = not (mask >> upper_idx & 1)

        if upper_open:
            # Mandatory: play in corresponding upper section
            score = calculate_score(upper_idx, dice)
            nu = min(63, upper + score)
            u_bonus = 35 if upper < 63 and nu >= 63 else 0
            pts = bonus + score + u_bonus
            return (upper_idx, pts, mask | (1 << upper_idx), nu, y_state)

        # Try lower categories (joker: FH/SS/LS guaranteed)
        found_lower = False
        for c in range(6, 12):
            if not (mask >> c & 1):
                score = calculate_score(c, dice)
                if c == 8: score = 25
                elif c == 9: score = 30
                elif c == 10: score = 40
                pts = bonus + score
                try_move(c, pts, mask | (1 << c), upper, y_state)
                found_lower = True

        if not found_lower:
            # Force: use any remaining upper for 0
            for c in range(6):
                if not (mask >> c & 1):
                    try_move(c, bonus, mask | (1 << c), upper, y_state)

        return best

    # Normal (non-yahtzee) roll: try all open categories 0-11
    for c in range(12):
        if not (mask >> c & 1):
            score = calculate_score(c, dice)
            nu = upper
            u_bonus = 0
            if c < 6:
                nu = min(63, upper + score)
                u_bonus = 35 if upper < 63 and nu >= 63 else 0
            pts = score + u_bonus
            try_move(c, pts, mask | (1 << c), nu, y_state)

    # Also option to fill yahtzee box with 0 (sacrifice yahtzee for flexibility)
    if y_state == 0:
        try_move(CAT_YAHTZEE, 0, mask, upper, 1)

    return best

# ---------------------------------------------------------------------------
# Load binary strategy files
# ---------------------------------------------------------------------------

def load_ev_table(filename='yahtzee_optimal_v4.bin'):
    """
    Load the precomputed start-of-turn EV table from binary.
    Returns board_ev (shape: BOARD_STATE_SIZE,) as a numpy array (float32).

    Accepts version 4 files (board EVs only) and version 3 files (board EVs
    followed by mid-turn tables, which are ignored: every mid-turn value is
    recomputed from board_ev, see simulate_turn_solo_optimal).
    """
    with open(filename, 'rb') as f:
        magic, version, board_size, roll_size = struct.unpack('<IIII', f.read(16))
    if magic != 0x59414854 or version not in (3, 4):
        raise ValueError(f'Invalid EV file: magic={hex(magic)} version={version}')
    board_ev = np.memmap(filename, dtype='float32', mode='r',
                         offset=16, shape=(board_size,))
    return board_ev


class PMFTable:
    """
    Read-only view of the PMF table: pmf[board_index] is a dense float32 row
    of P(future score = s) for s = 0 .. dense_size-1, and pmf.shape is
    (num_states, dense_size), like a 2-D array.

    Version 2 files (yahtzee_pmf_v2.bin) store each state's nonzero score
    range only: an index of {uint32 offset, int16 lo, int16 width} entries,
    then the float data. Version 1 files store full dense rows.
    """

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            magic, version, num_states, dense_size = struct.unpack('<IIII', f.read(16))
        if magic != 0x504D4659 or version not in (1, 2):
            raise ValueError(f'Invalid PMF file: magic={hex(magic)} version={version}')
        self.version = version
        self.shape = (num_states, dense_size)
        if version == 1:
            self._dense = np.memmap(filename, dtype='float32', mode='r',
                                    offset=16, shape=self.shape)
        else:
            index_dtype = np.dtype([('offset', '<u4'), ('lo', '<i2'), ('width', '<i2')])
            self._index = np.memmap(filename, dtype=index_dtype, mode='r',
                                    offset=16, shape=(num_states,))
            self._data = np.memmap(filename, dtype='float32', mode='r',
                                   offset=16 + index_dtype.itemsize * num_states)

    def __getitem__(self, idx):
        if self.version == 1:
            return self._dense[idx]
        offset, lo, width = self._index[idx]
        row = np.zeros(self.shape[1], dtype=np.float32)
        row[lo:lo + width] = self._data[offset:offset + width]
        return row


def load_pmf_table(filename='yahtzee_pmf_v2.bin'):
    """
    Load the precomputed PMF table (memory-mapped).
    Returns a PMFTable: pmf[board_index] -> dense row of length dense_size.
    dense_size is read from the file header — no hardcoded constant needed.
    """
    return PMFTable(filename)

# ---------------------------------------------------------------------------
# P_baseline: win probability under both-play-solo-optimally
# ---------------------------------------------------------------------------

def compute_p_baseline(mask_c, upper_c, y_c, mask_o, upper_o, y_o, diff, pmf_dense):
    """
    P(current player wins | both continue with solo-optimal strategy).

    Uses: P_win = sum_r  PMF_cur[r] * CDF_opp[r + diff]
    where CDF_opp[k] = P(opp future score <= k).

    diff = accumulated_current - accumulated_opponent (positive = cur is winning).
    O(dense_size) computation.
    """
    idx_c = board_index(mask_c, upper_c, y_c)
    idx_o = board_index(mask_o, upper_o, y_o)

    pmf_c = pmf_dense[idx_c]
    pmf_o = pmf_dense[idx_o]
    dense_size = pmf_dense.shape[1]

    cdf_o = np.cumsum(pmf_o)

    indices = np.arange(dense_size) + int(diff)
    cdf_vals = np.where(indices < 0, 0.0,
               np.where(indices >= dense_size, 1.0,
                        cdf_o[np.clip(indices, 0, dense_size - 1)]))
    return float(np.dot(pmf_c, cdf_vals))

# ---------------------------------------------------------------------------
# Optimal keep selection  (recomputed from board_ev)
# ---------------------------------------------------------------------------

def turn_values(mask, upper, y_state, board_ev):
    """
    Mid-turn values for one board state, derived from board_ev.
    Returns (term_vals, roll2_vals), each shape (252,):
      term_vals[d]  = value of final dice d (best scoring move)
      roll2_vals[d] = value of dice d with one roll left (best keep)
    """
    term_vals = np.empty(NUM_DICE_COMBOS, dtype=np.float32)
    for d in range(NUM_DICE_COMBOS):
        res = get_best_scoring_move(d, mask, upper, y_state, board_ev)
        if res is None:
            term_vals[d] = 0.0
        else:
            _, pts, nm, nu, ny = res
            term_vals[d] = pts + board_ev[board_index(nm, nu, ny)]
    keep_ev = TRANS_PROBS @ term_vals                       # (462,)
    roll2_vals = keep_ev[VALID_SUBSETS_PADDED].max(axis=1)  # (252,)
    return term_vals, roll2_vals


def simulate_turn_solo_optimal(mask, upper, y_state, board_ev):
    """
    Simulate one turn of solo-optimal play via dice rolling.
    Returns (points_scored, next_mask, next_upper, next_y_state).

    Keep decisions are recomputed from board_ev (same logic as C++ guide).
    """
    term_vals, roll2_vals = turn_values(mask, upper, y_state, board_ev)

    # Roll 1
    dice1 = tuple(sorted(int(x) for x in np.random.randint(1, 7, 5)))
    d1 = DICE_TO_ID[dice1]

    # Keep after roll 1: maximise E[roll2 value over next roll]
    best_ev, best_sid = -1e30, 0
    for sid in VALID_SUBSETS_FOR_OUTCOME[d1]:
        ev = float(TRANS_PROBS[sid] @ roll2_vals)
        if ev > best_ev:
            best_ev, best_sid = ev, sid
    kept1 = list(ALL_SUBSETS[best_sid])

    # Roll 2
    dice2 = tuple(sorted(kept1 + [int(x) for x in np.random.randint(1, 7, 5 - len(kept1))]))
    d2 = DICE_TO_ID[dice2]

    # Keep after roll 2: maximise E[terminal scoring]
    best_ev2, best_sid2 = -1e30, 0
    for sid in VALID_SUBSETS_FOR_OUTCOME[d2]:
        ev = float(TRANS_PROBS[sid] @ term_vals)
        if ev > best_ev2:
            best_ev2, best_sid2 = ev, sid
    kept2 = list(ALL_SUBSETS[best_sid2])

    # Roll 3 and score
    dice3 = tuple(sorted(kept2 + [int(x) for x in np.random.randint(1, 7, 5 - len(kept2))]))
    d3 = DICE_TO_ID[dice3]
    res = get_best_scoring_move(d3, mask, upper, y_state, board_ev)
    if res is None:
        return (0, mask, upper, y_state)
    _, pts, nm, nu, ny = res
    return (pts, nm, nu, ny)



# Feature extraction for NN
# ---------------------------------------------------------------------------

def extract_features(mask_c, upper_c, y_c, mask_o, upper_o, y_o, diff, p_baseline):
    """
    Build 36-dim feature vector for V(cur, opp, diff).

    Per player (17 features):
      - mask bits 0-11 (12)
      - upper deficit: (upper - 3 * filled_upper_cats) / 63  (1)
      - y_state one-hot: [open, filled0, filled50]  (3)
      - turns_remaining / 13  (1)

    Global (2 features):
      - diff / 500, clamped  (1)
      - P_baseline  (1)

    Total: 17 + 17 + 2 = 36 features.
    """
    def player_features(mask, upper, y_state):
        f = np.zeros(17, dtype=np.float32)
        # Mask bits
        for b in range(12):
            f[b] = (mask >> b) & 1
        # Upper deficit (3 = average dice per upper cat to hit 63)
        filled_upper = bin(mask & 0x3F).count('1')
        f[12] = (upper - 3.0 * filled_upper) / 63.0
        # y_state one-hot
        f[13 + y_state] = 1.0
        # turns remaining
        f[16] = (13 - cats_filled(mask, y_state)) / 13.0
        return f

    features = np.concatenate([
        player_features(mask_c, upper_c, y_c),
        player_features(mask_o, upper_o, y_o),
        np.float32([np.clip(diff / 500.0, -1.0, 1.0)]),
        np.float32([p_baseline]),
    ])
    return features   # shape (36,)


# Vectorized version for batches: arrays of shape (N,)
def extract_features_batch(masks_c, uppers_c, y_states_c,
                           masks_o, uppers_o, y_states_o,
                           diffs, p_baselines):
    """
    N-element batch → (N, 36) float32 array.
    All inputs are 1-D numpy integer/float arrays of length N.
    """
    N = len(masks_c)
    out = np.zeros((N, 36), dtype=np.float32)

    def fill_player(offset, masks, uppers, y_states):
        for b in range(12):
            out[:, offset + b] = (masks >> b) & 1
        filled_upper = np.vectorize(lambda m: bin(m & 0x3F).count('1'))(masks)
        out[:, offset + 12] = (uppers - 3.0 * filled_upper) / 63.0
        for ys in range(3):
            out[:, offset + 13 + ys] = (y_states == ys).astype(np.float32)
        turns = np.vectorize(cats_filled)(masks, y_states)
        out[:, offset + 16] = (13 - turns) / 13.0

    fill_player(0, masks_c, uppers_c, y_states_c)
    fill_player(17, masks_o, uppers_o, y_states_o)
    out[:, 34] = np.clip(diffs / 500.0, -1.0, 1.0)
    out[:, 35] = p_baselines
    return out
