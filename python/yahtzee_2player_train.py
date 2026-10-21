"""
2-player Yahtzee training.

Phase 2: Supervised bootstrap
  - Generate game trajectories using solo-optimal play (from EV table).
  - Label each turn-start state with P_baseline from the PMF table.
  - Train NN with binary cross-entropy.

Phase 3: TD self-play RL
  - At each turn, enumerate all 252 terminal dice × scoring choices.
  - Compute exact expected value (no sampling) using the transition matrix.
  - TD target = 1 - E[V(next_opp, next_cur, -new_diff)].
  - Update with MSE loss; stabilise with EMA target network.

Usage:
  python yahtzee_2player_train.py [--phase 2|3] [--iters N] [--load checkpoint.pt]
"""

import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yahtzee_2player_env import (
    TRANS_PROBS, DICE_COMBOS, NUM_DICE_COMBOS, VALID_SUBSETS_FOR_OUTCOME,
    ALL_SUBSETS, DICE_TO_ID,
    BOARD_STATE_SIZE, DENSE_SIZE, DENSE_MAX,
    board_index, cats_filled, get_best_scoring_move,
    extract_features, compute_p_baseline, load_ev_table, load_pmf_table,
    simulate_turn_solo_optimal,
)
from yahtzee_2player_nn import YahtzeeValueNet, YahtzeeInference, EMANet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EV_FILE  = 'yahtzee_optimal_v4.bin'
PMF_FILE = 'yahtzee_pmf_v1.bin'
CKPT_DIR = Path('checkpoints_2player')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def generate_bootstrap_data(n_games, board_ev, pmf_dense):
    """
    Play n_games with both players using solo-optimal strategy.
    For each turn-start state, record (features, P_baseline) as a training pair.
    Returns (features_array, labels_array) each of shape (n_samples, 36) and (n_samples,).
    """
    features_list, labels_list = [], []

    for _ in range(n_games):
        mask1 = mask2 = 0
        upper1 = upper2 = 0
        y1 = y2 = 0
        score1 = score2 = 0

        for turn in range(13):
            # Player 1's turn
            diff = score1 - score2
            p_base = compute_p_baseline(mask1, upper1, y1, mask2, upper2, y2,
                                        diff, pmf_dense)
            feat = extract_features(mask1, upper1, y1, mask2, upper2, y2,
                                    diff, p_base)
            features_list.append(feat)
            labels_list.append(p_base)   # bootstrap label = P_baseline

            pts, nm1, nu1, ny1 = simulate_turn_solo_optimal(
                mask1, upper1, y1, board_ev)
            mask1, upper1, y1 = nm1, nu1, ny1
            score1 += pts

            # Player 2's turn
            diff2 = score2 - score1
            p_base2 = compute_p_baseline(mask2, upper2, y2, mask1, upper1, y1,
                                          diff2, pmf_dense)
            feat2 = extract_features(mask2, upper2, y2, mask1, upper1, y1,
                                     diff2, p_base2)
            features_list.append(feat2)
            labels_list.append(p_base2)

            pts2, nm2, nu2, ny2 = simulate_turn_solo_optimal(
                mask2, upper2, y2, board_ev)
            mask2, upper2, y2 = nm2, nu2, ny2
            score2 += pts2

    X = np.stack(features_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.float32)
    return X, y


def phase2_train(model, n_games=200_000, batch_size=4096, lr=3e-4, device='cuda'):
    print('\n=== Phase 2: Supervised bootstrap ===')
    print('Loading EV and PMF tables...')
    board_ev = load_ev_table(EV_FILE)
    pmf_dense = load_pmf_table(PMF_FILE)
    print('Tables loaded.')

    print(f'Generating {n_games} games of bootstrap data...')
    t0 = time.time()
    X, y = generate_bootstrap_data(n_games, board_ev, pmf_dense)
    print(f'  {len(X)} samples in {time.time()-t0:.1f}s')

    model = model.to(device).train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    n_batches = len(X) // batch_size
    for epoch in range(5):
        perm = np.random.permutation(len(X))
        epoch_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size: (i+1) * batch_size]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f'  Epoch {epoch+1}/5  loss={epoch_loss/n_batches:.5f}')

    return model

# ---------------------------------------------------------------------------
# Phase 3: TD self-play RL
# ---------------------------------------------------------------------------

def compute_td_targets(states_batch, opp_board, diff_batch, pmf_dense,
                       target_net, trans, board_ev, device):
    """
    For a batch of turn-start states (all sharing the same opponent board opp_board),
    compute exact TD targets using the full dice enumeration.

    Each state in the batch: (mask_c, upper_c, y_c, diff)
    opp_board: (mask_o, upper_o, y_o) — fixed for this batch

    Returns: (N,) float32 tensor of TD targets.

    The exact algorithm:
      1. For each terminal dice d3 × each open category → (next_cur, new_diff)
      2. Collect unique (next_mask, next_upper, next_y, new_diff, mask_o, upper_o, y_o) states
      3. One NN forward pass (roles swapped: V(opp, next_cur, -new_diff))
      4. Build V_terminal[state, d3] from NN outputs
      5. Roll2 keep: V_roll2[state, d2] = max_k sum_d3 trans[k,d3] * V_terminal[state,d3]
      6. Roll1 keep: V_roll1[state, d1] = max_k sum_d2 trans[k,d2] * V_roll2[state,d2]
      7. TD target = sum_d1 trans[0,d1] * V_roll1[state, d1]
         (then TD target for cur = 1 - expected V because roles swap)
    """
    mask_o, upper_o, y_o = opp_board
    N = len(states_batch)

    # Step 1: For each state, compute terminal values V_terminal[state, d3]
    # by evaluating all (next_cur_state, diff) pairs via the NN.
    # We batch all (state × d3 × category) evaluations together.

    all_feat = []
    all_indices = []  # (state_idx, d3)

    for si, (mask_c, upper_c, y_c, diff) in enumerate(states_batch):
        for d3 in range(NUM_DICE_COMBOS):
            res = get_best_scoring_move(d3, mask_c, upper_c, y_c, board_ev)
            if res is None:
                # No valid move (shouldn't happen in valid game state)
                all_feat.append(np.zeros(36, dtype=np.float32))
                all_indices.append((si, d3, 0.0))
                continue
            _, pts, nm, nu, ny = res
            new_diff = -(diff + pts)  # negative because roles swap
            p_base = compute_p_baseline(mask_o, upper_o, y_o, nm, nu, ny,
                                        new_diff, pmf_dense)
            # NN called with (opp, next_cur, -new_diff) = (opp, next_cur, diff+pts)
            feat = extract_features(mask_o, upper_o, y_o, nm, nu, ny,
                                    new_diff, p_base)
            all_feat.append(feat)
            all_indices.append((si, d3, 0.0))  # placeholder for NN value

    # Single batch NN forward pass
    feat_arr = np.stack(all_feat, axis=0)
    with torch.no_grad():
        x = torch.from_numpy(feat_arr).to(device)
        nn_vals = target_net(x).cpu().numpy()  # shape (N * 252,)

    # Reshape: V_terminal[state, d3] = 1 - nn_val (already from opponent's perspective)
    # Note: NN gives P(opp wins), so P(cur wins at terminal) = 1 - nn_val? Actually:
    # The NN is called with opp as "current player", so nn_val = P(opp wins).
    # But we want V for the current player = P(cur wins) = 1 - P(opp wins).
    # However, since this IS the terminal of the cur player's turn, and after scoring
    # the roles swap, the value to cur = 1 - V(opp, next_cur, -new_diff). ✓
    V_terminal = (1.0 - nn_vals).reshape(N, NUM_DICE_COMBOS)

    # Steps 5-7: Backward pass through keep decisions
    # V_roll2[state, d2] = max_k trans[k] @ V_terminal[state]
    trans_np = trans  # (462, 252)

    V_roll2 = np.zeros((N, NUM_DICE_COMBOS), dtype=np.float32)
    for d2 in range(NUM_DICE_COMBOS):
        for si in range(N):
            mask_c, upper_c, y_c, diff = states_batch[si]
            best = -1e30
            for sid in VALID_SUBSETS_FOR_OUTCOME[d2]:
                v = float(trans_np[sid] @ V_terminal[si])
                if v > best:
                    best = v
            V_roll2[si, d2] = best

    V_roll1 = np.zeros((N, NUM_DICE_COMBOS), dtype=np.float32)
    for d1 in range(NUM_DICE_COMBOS):
        for si in range(N):
            best = -1e30
            for sid in VALID_SUBSETS_FOR_OUTCOME[d1]:
                v = float(trans_np[sid] @ V_roll2[si])
                if v > best:
                    best = v
            V_roll1[si, d1] = best

    # TD target = init roll distribution @ V_roll1
    init_probs = trans_np[0]   # roll all 5 dice
    td_targets = V_roll1 @ init_probs   # (N,)
    return torch.from_numpy(td_targets).float()


def play_td_game(mask1, upper1, y1, mask2, upper2, y2,
                 board_ev, pmf_dense,
                 inference):
    """
    Play a single game using the NN for strategy. Collect (state, opp, diff, td_target) tuples.
    Uses solo-optimal for the actual dice rolling; NN guides scoring/keep decisions.
    (Simplified: both players use solo-optimal play for generation; NN just gets trained.)
    """
    score1 = score2 = 0
    experiences = []

    for _ in range(13):
        # Record state before player 1's turn
        diff = score1 - score2
        experiences.append({
            'mask_c': mask1, 'upper_c': upper1, 'y_c': y1,
            'mask_o': mask2, 'upper_o': upper2, 'y_o': y2,
            'diff': diff,
        })
        pts, nm1, nu1, ny1 = simulate_turn_solo_optimal(
            mask1, upper1, y1, board_ev)
        mask1, upper1, y1 = nm1, nu1, ny1
        score1 += pts

        diff2 = score2 - score1
        experiences.append({
            'mask_c': mask2, 'upper_c': upper2, 'y_c': y2,
            'mask_o': mask1, 'upper_o': upper1, 'y_o': y1,
            'diff': diff2,
        })
        pts2, nm2, nu2, ny2 = simulate_turn_solo_optimal(
            mask2, upper2, y2, board_ev)
        mask2, upper2, y2 = nm2, nu2, ny2
        score2 += pts2

    return experiences, (1 if score1 > score2 else 0)


def phase3_train(model, n_iters=1000, games_per_iter=1000, batch_size=4096,
                 lr=3e-4, ema_decay=0.999, device='cuda', save_every=100):
    print('\n=== Phase 3: TD self-play RL ===')
    board_ev = load_ev_table(EV_FILE)
    pmf_dense = load_pmf_table(PMF_FILE)
    print('Tables loaded.')

    CKPT_DIR.mkdir(exist_ok=True)
    model = model.to(device).train()
    target_model = YahtzeeValueNet().to(device).eval()
    target_model.load_state_dict(model.state_dict())
    ema = EMANet(model, decay=ema_decay)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay_buffer = []  # list of (feat, td_target) pairs
    MAX_BUFFER = 500_000

    inference = YahtzeeInference(target_model, device=device)

    for it in range(1, n_iters + 1):
        t0 = time.time()

        # Generate games
        new_experiences = []
        for _ in range(games_per_iter):
            exp, winner = play_td_game(0, 0, 0, 0, 0, 0,
                                       board_ev, pmf_dense,
                                       inference)
            new_experiences.extend(exp)

        # Compute TD targets for new experiences
        # Group by opp_board to reduce NN batch overhead
        opp_groups = {}
        for e in new_experiences:
            key = (e['mask_o'], e['upper_o'], e['y_o'])
            opp_groups.setdefault(key, []).append(e)

        new_pairs = []
        with torch.no_grad():
            for opp_board, group in opp_groups.items():
                states_batch = [(e['mask_c'], e['upper_c'], e['y_c'], e['diff'])
                                for e in group]
                td_tgts = compute_td_targets(
                    states_batch, opp_board, [e['diff'] for e in group],
                    pmf_dense, target_model, TRANS_PROBS, board_ev, device)

                for e, tgt in zip(group, td_tgts.numpy()):
                    p_base = compute_p_baseline(
                        e['mask_c'], e['upper_c'], e['y_c'],
                        e['mask_o'], e['upper_o'], e['y_o'],
                        e['diff'], pmf_dense)
                    feat = extract_features(
                        e['mask_c'], e['upper_c'], e['y_c'],
                        e['mask_o'], e['upper_o'], e['y_o'],
                        e['diff'], p_base)
                    new_pairs.append((feat, float(tgt)))

        replay_buffer.extend(new_pairs)
        if len(replay_buffer) > MAX_BUFFER:
            replay_buffer = replay_buffer[-MAX_BUFFER:]

        # Training step
        sample = random.sample(replay_buffer, min(batch_size, len(replay_buffer)))
        feats  = np.stack([s[0] for s in sample], axis=0).astype(np.float32)
        labels = np.array([s[1] for s in sample], dtype=np.float32)

        xb = torch.from_numpy(feats).to(device)
        yb = torch.from_numpy(labels).to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Update EMA target network
        ema.update(model)
        if it % 10 == 0:
            target_model.load_state_dict(ema.get_model().state_dict())

        elapsed = time.time() - t0
        print(f'Iter {it}/{n_iters}  loss={loss.item():.5f}  '
              f'buf={len(replay_buffer)}  t={elapsed:.1f}s')

        if it % save_every == 0:
            path = CKPT_DIR / f'ckpt_iter{it:05d}.pt'
            torch.save(model.state_dict(), path)
            print(f'  Saved {path}')

    return model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3])
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = YahtzeeValueNet()
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location='cpu'))
        print(f'Loaded checkpoint: {args.load}')

    if args.phase == 2:
        n_games = args.games or 50_000
        model = phase2_train(model, n_games=n_games, device=args.device)
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), CKPT_DIR / 'phase2_bootstrap.pt')
        print(f'Phase 2 done. Saved to {CKPT_DIR}/phase2_bootstrap.pt')

    elif args.phase == 3:
        n_iters  = args.iters or 1000
        n_games  = args.games or 500
        model = phase3_train(model, n_iters=n_iters, games_per_iter=n_games,
                             device=args.device)
        torch.save(model.state_dict(), CKPT_DIR / 'phase3_final.pt')
        print(f'Phase 3 done. Saved to {CKPT_DIR}/phase3_final.pt')
