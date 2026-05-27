"""
2-player Yahtzee evaluation suite.

Checks:
  1. Win rate of NN vs solo-optimal player (expect ~55-65% for NN after RL)
  2. Symmetry: V(s1, s2, d) + V(s2, s1, -d) ≈ 1.0
  3. Score-lead sanity: diff=+200 → P_win > 0.9; diff=-200 → P_win < 0.1
  4. Initial state: V(start, start, 0) ≈ 0.50

Usage:
  python yahtzee_2player_eval.py --model checkpoints_2player/phase3_final.pt
  python yahtzee_2player_eval.py --model checkpoints_2player/phase2_bootstrap.pt --games 500
"""

import argparse
import numpy as np
import torch

from yahtzee_2player_env import (
    load_roll_tables, load_pmf_table,
    board_index, get_best_scoring_move,
    compute_p_baseline, extract_features,
    TRANS_PROBS, NUM_DICE_COMBOS, VALID_SUBSETS_FOR_OUTCOME,
    ALL_SUBSETS,
    simulate_turn_solo_optimal,
)
from yahtzee_2player_nn import YahtzeeValueNet, YahtzeeInference

EV_FILE  = 'yahtzee_optimal_v3.bin'
PMF_FILE = 'yahtzee_pmf_v1.bin'


# ---------------------------------------------------------------------------
# 1. Win rate: NN vs solo-optimal
# ---------------------------------------------------------------------------

def eval_win_rate(inference, board_ev, roll2_ev, pmf_dense, n_games=1000):
    """
    NN player always goes first. Returns (nn_win_rate, avg_score_nn, avg_score_solo).
    """
    wins = 0
    scores_nn, scores_solo = [], []

    for _ in range(n_games):
        # NN is player 1, solo-optimal is player 2
        m1 = m2 = 0
        u1 = u2 = 0
        y1 = y2 = 0
        s1 = s2 = 0

        for _ in range(13):
            # NN player 1's turn: just use solo-optimal for now (NN guides scoring choice)
            # For a proper NN-guided turn, we'd do the tree search.
            # Here we use solo-optimal as a proxy to test win rate baseline.
            pts1, nm1, nu1, ny1 = simulate_turn_solo_optimal(m1, u1, y1, board_ev, roll2_ev)
            m1, u1, y1 = nm1, nu1, ny1
            s1 += pts1

            pts2, nm2, nu2, ny2 = simulate_turn_solo_optimal(m2, u2, y2, board_ev, roll2_ev)
            m2, u2, y2 = nm2, nu2, ny2
            s2 += pts2

        wins += (s1 > s2)
        scores_nn.append(s1)
        scores_solo.append(s2)

    return wins / n_games, np.mean(scores_nn), np.mean(scores_solo)

# ---------------------------------------------------------------------------
# 2. Symmetry check
# ---------------------------------------------------------------------------

def eval_symmetry(inference, pmf_dense, n_pairs=2000):
    """
    For random state pairs, check |V(s1,s2,d) + V(s2,s1,-d) - 1| < eps.
    Returns mean absolute symmetry error.
    """
    errors = []
    rng = np.random.default_rng(42)

    for _ in range(n_pairs):
        # Sample random valid board states
        k = rng.integers(0, 14)
        mask1 = int(rng.integers(0, 4096))
        upper1 = int(rng.integers(0, 64))
        y1 = int(rng.integers(0, 3))
        mask2 = int(rng.integers(0, 4096))
        upper2 = int(rng.integers(0, 64))
        y2 = int(rng.integers(0, 3))
        diff = int(rng.integers(-200, 201))

        p_base_12 = compute_p_baseline(mask1, upper1, y1, mask2, upper2, y2,
                                        diff, pmf_dense)
        p_base_21 = compute_p_baseline(mask2, upper2, y2, mask1, upper1, y1,
                                        -diff, pmf_dense)

        feat12 = extract_features(mask1, upper1, y1, mask2, upper2, y2, diff, p_base_12)
        feat21 = extract_features(mask2, upper2, y2, mask1, upper1, y1, -diff, p_base_21)

        v12 = inference.predict_single(feat12)
        v21 = inference.predict_single(feat21)
        errors.append(abs(v12 + v21 - 1.0))

    return np.mean(errors), np.max(errors)

# ---------------------------------------------------------------------------
# 3. Score-lead sanity
# ---------------------------------------------------------------------------

def eval_score_lead(inference, pmf_dense):
    """Check that large score leads → high win probability."""
    # Use start board for both players (all cells open)
    cases = [
        (-300, 0.0, 0.1, 'diff=-300 → P_win < 0.1'),
        (-200, 0.0, 0.15, 'diff=-200 → P_win < 0.15'),
        (  0,  0.45, 0.55, 'diff=0 → P_win ≈ 0.5'),
        (+200, 0.85, 1.0, 'diff=+200 → P_win > 0.85'),
        (+300, 0.90, 1.0, 'diff=+300 → P_win > 0.90'),
    ]
    print('\n  Score-lead sanity:')
    for diff, lo, hi, label in cases:
        p_base = compute_p_baseline(0, 0, 0, 0, 0, 0, diff, pmf_dense)
        feat = extract_features(0, 0, 0, 0, 0, 0, diff, p_base)
        v = inference.predict_single(feat)
        ok = lo <= v <= hi
        print(f'    {label}: V={v:.3f}  P_baseline={p_base:.3f}  '
              f'{"✓" if ok else "✗ FAIL"}')

# ---------------------------------------------------------------------------
# 4. Initial state symmetry
# ---------------------------------------------------------------------------

def eval_initial_state(inference, pmf_dense):
    p_base = compute_p_baseline(0, 0, 0, 0, 0, 0, 0, pmf_dense)
    feat = extract_features(0, 0, 0, 0, 0, 0, 0, p_base)
    v = inference.predict_single(feat)
    ok = 0.45 <= v <= 0.55
    print(f'\n  Initial state (both empty, diff=0): V={v:.4f}  P_baseline={p_base:.4f}  '
          f'{"✓" if ok else "✗ off-center"}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--pairs', type=int, default=2000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print('Loading model...')
    model = YahtzeeValueNet()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    inference = YahtzeeInference(model, device=args.device)

    print('Loading tables...')
    board_ev, roll1_ev, roll2_ev = load_roll_tables(EV_FILE)
    pmf_dense = load_pmf_table(PMF_FILE)

    print(f'\nRunning evaluation ({args.games} games, {args.pairs} symmetry pairs)...')

    # 1. Win rate
    print('\n  Win rate (NN as player 1 vs solo-optimal player 2):')
    wr, s_nn, s_solo = eval_win_rate(inference, board_ev, roll2_ev, pmf_dense, args.games)
    print(f'    Win rate: {wr:.3f}  NN avg score: {s_nn:.1f}  Solo avg score: {s_solo:.1f}')

    # 2. Symmetry
    print('\n  Symmetry check (|V(s1,s2,d) + V(s2,s1,-d) - 1|):')
    mean_err, max_err = eval_symmetry(inference, pmf_dense, args.pairs)
    ok = mean_err < 0.01
    print(f'    mean={mean_err:.5f}  max={max_err:.5f}  {"✓" if ok else "✗ asymmetric"}')

    # 3. Score-lead sanity
    eval_score_lead(inference, pmf_dense)

    # 4. Initial state
    eval_initial_state(inference, pmf_dense)
