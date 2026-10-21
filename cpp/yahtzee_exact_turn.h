#ifndef YAHTZEE_EXACT_TURN_H
#define YAHTZEE_EXACT_TURN_H

/**
 * Exact evaluation of one turn, shared by the rational EV and PMF solvers.
 *
 * Every probability in a turn is a product of 1/6 factors and max() only
 * selects one of its arguments, so with k turns left every board EV is an
 * integer divided by D_k = 6^(15k). Within a turn:
 *
 *   final-roll value   : numerator / D_{k-1}
 *   one roll left      : numerator / (D_{k-1} * 6^5)
 *   two rolls left     : numerator / (D_{k-1} * 6^10)
 *   start of turn      : numerator / (D_{k-1} * 6^15) = D_k
 *
 * Values in one stage share a denominator, so the whole turn runs on integer
 * numerators (UInt<L>): addition, small multiplications and comparison.
 *
 * Rerolls use the keep lattice: with S[K] = 6^(5-|K|) * E[K],
 * S[K] = sum_f S[K + {f}] (pure additions), and T[K] = 6^|K| * S[K] =
 * 6^5 * E[K] puts every keep on the common denominator.
 *
 * TIE-BREAKS (exact ties do occur and change the score distribution, though
 * not the EV): scoring takes the first best option in ScoringRules order;
 * keeps prefer more dice kept, then the lowest subset id.
 */

#include <cstdint>
#include <vector>

#include "yahtzee_bigint.h"
#include "yahtzee_keep_lattice.h"
#include "yahtzee_rules.h"

// EV numerators are stored with this many limbs (2048 * 6^195 < 2^517)
const int EV_LIMBS = 9;
// Numerators below 2^11 * 6^(15k) (scores are at most 1575 < 2^11)
inline int EvLimbs(int turns_left) { return LimbsFor(turns_left, 11); }

// Largest single-turn reward: 100 (Yahtzee bonus) + 30 (sixes) + 35 (bonus)
const int MAX_REWARD = 200;

// The optimal policy for one turn
struct TurnPolicy {
  int16_t reward[NUM_DICE_COMBOS];     // points scored for each final roll
  uint32_t next_idx[NUM_DICE_COMBOS];  // board index after scoring it
  int16_t keep2[NUM_DICE_COMBOS];      // subset kept with one roll left
  int16_t keep1[NUM_DICE_COMBOS];      // subset kept with two rolls left
};

template <int L>
struct ExactTurnScratch {
  UInt<L> term[NUM_DICE_COMBOS];
  UInt<L> roll2[NUM_DICE_COMBOS];
  UInt<L> roll1[NUM_DICE_COMBOS];
  UInt<L> S[KeepLattice::kNumSubsets];
};

template <int L>
class ExactTurnSolver {
 public:
  // ev_table: EV numerators of the next layer (EV_LIMBS per board index,
  // over D_{k-1}); terminal boards must be zero.
  ExactTurnSolver(const ScoringRules& rules, const KeepLattice& lattice,
                  int turns_left, const u64* ev_table)
      : rules_(rules), lattice_(lattice), ev_(ev_table), reward_scaled_(MAX_REWARD + 1) {
    // reward_scaled[r] = r * D_{k-1}
    UInt<L> d;
    SetSmall(d, 1);
    for (int t = 0; t < 15 * (turns_left - 1); ++t) MulSmall(d, d, 6);
    for (int r = 0; r <= MAX_REWARD; ++r) MulSmall(reward_scaled_[r], d, r);
  }

  // board = EV numerator over D_k; fills policy when non-null
  void Solve(int mask, int upper, int y_state, ExactTurnScratch<L>& w,
             UInt<L>& board, TurnPolicy* policy) const {
    // Stage 1: best scoring move per final roll (over D_{k-1})
    ScoringOption opts[MAX_SCORING_OPTIONS];
    for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
      int n = rules_.Options(d, mask, upper, y_state, opts);
      int best_i = 0;
      if (n == 0) SetZero(w.term[d]);  // terminal board: nothing to score
      for (int i = 0; i < n; ++i) {
        const ScoringOption& o = opts[i];
        UInt<L> v, next;
        Load(next, ev_ + EV_LIMBS * BoardStateIndex(o.next_mask, o.next_upper, o.next_y_state));
        Add(v, reward_scaled_[o.reward], next);
        if (i == 0 || Less(w.term[d], v)) {
          w.term[d] = v;
          best_i = i;
        }
      }
      if (policy) {
        const ScoringOption& o = opts[best_i];
        policy->reward[d] = static_cast<int16_t>(o.reward);
        policy->next_idx[d] = static_cast<uint32_t>(
            BoardStateIndex(o.next_mask, o.next_upper, o.next_y_state));
      }
    }

    // Stages 2 and 3: the two reroll decisions
    Reroll(w.term, w.roll2, w.S, policy ? policy->keep2 : nullptr);
    Reroll(w.roll2, w.roll1, w.S, policy ? policy->keep1 : nullptr);

    // Stage 4: initial roll of all five dice; S[0] = 6^5 * E over D_{k-1}*6^10
    Sums(w.roll1, w.S);
    board = w.S[0];
  }

 private:
  const ScoringRules& rules_;
  const KeepLattice& lattice_;
  const u64* ev_;
  std::vector<UInt<L>> reward_scaled_;

  // S[K] = sum over rerolls of the five-dice values (6^(5-|K|) * E[K])
  void Sums(const UInt<L>* in, UInt<L>* S) const {
    const int kFirst5 = KeepLattice::kFirst5;
    for (int d = 0; d < NUM_DICE_COMBOS; ++d) S[kFirst5 + d] = in[d];
    for (int i = kFirst5 - 1; i >= 0; --i) {
      const auto& ch = lattice_.Children(i);
      Add(S[i], S[ch[0]], S[ch[1]]);
      for (int f = 2; f < 6; ++f) AddTo(S[i], S[ch[f]]);
    }
  }

  // One reroll decision: in[d] over X, out[d] over X * 6^5
  void Reroll(const UInt<L>* in, UInt<L>* out, UInt<L>* S, int16_t* keep) const {
    static const u64 kPow6[6] = {1, 6, 36, 216, 1296, 7776};
    Sums(in, S);
    // T[K] = 6^|K| * S[K] (in place)
    for (int i = 1; i < KeepLattice::kNumSubsets; ++i) {
      MulSmall(S[i], S[i], kPow6[lattice_.Size(i)]);
    }
    for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
      const auto& keeps = lattice_.Keeps(d);  // ascending id = ascending size
      int best = keeps[0];
      for (size_t j = 1; j < keeps.size(); ++j) {
        int sid = keeps[j];
        if (Less(S[best], S[sid]) ||
            (Equal(S[best], S[sid]) && lattice_.Size(sid) > lattice_.Size(best))) {
          best = sid;
        }
      }
      out[d] = S[best];
      if (keep) keep[d] = static_cast<int16_t>(best);
    }
  }
};

#endif  // YAHTZEE_EXACT_TURN_H
