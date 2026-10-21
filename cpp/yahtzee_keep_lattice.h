#ifndef YAHTZEE_KEEP_LATTICE_H
#define YAHTZEE_KEEP_LATTICE_H

/**
 * KeepLattice: within-turn expectations over kept dice.
 *
 * Rerolling the non-kept dice is the same as rolling them one at a time, so
 * for a kept multiset K with fewer than 5 dice:
 *
 *   E[K] = (1/6) * sum_{f=1..6} E[K + {f}]
 *
 * One pass down the lattice of all 462 multisets of 0-5 dice gives the
 * expected value of every possible keep, given values for the 252 five-dice
 * outcomes. That is all the mid-turn computation there is: with the board
 * EVs of the next states known, a whole turn takes two passes (plus one for
 * the initial roll), so mid-turn values never need to be stored.
 *
 * Subset ids are ordered by size, then lexicographically, matching the
 * DiceIndexer ordering used elsewhere: id 0 is the empty keep, and the 252
 * five-dice outcomes occupy ids [kFirst5, 462) in DiceIndexer(5) order, so
 * outcome d has subset id kFirst5 + d.
 */

#include <algorithm>
#include <array>
#include <map>
#include <vector>

class KeepLattice {
 public:
  static const int kNumSubsets = 462;
  static const int kNumOutcomes = 252;
  static const int kFirst5 = kNumSubsets - kNumOutcomes;  // 210

  KeepLattice() {
    std::vector<int> current;
    for (int n = 0; n <= 5; ++n) Generate(current, 1, n);
    for (int i = 0; i < kFirst5; ++i) {
      for (int f = 1; f <= 6; ++f) {
        std::vector<int> c = dice_[i];
        c.push_back(f);
        std::sort(c.begin(), c.end());
        children_[i][f - 1] = id_.at(c);
      }
    }
    keeps_.resize(kNumOutcomes);
    for (int d = 0; d < kNumOutcomes; ++d) {
      const std::vector<int>& dice = dice_[kFirst5 + d];
      for (int m = 0; m < 32; ++m) keeps_[d].push_back(KeptSubset(dice, m));
      std::sort(keeps_[d].begin(), keeps_[d].end());
      keeps_[d].erase(std::unique(keeps_[d].begin(), keeps_[d].end()), keeps_[d].end());
    }
  }

  // Sorted dice of a subset id
  const std::vector<int>& Dice(int sid) const { return dice_[sid]; }

  // Subset id of any multiset of 0-5 dice (order does not matter)
  int SubsetId(std::vector<int> dice) const {
    std::sort(dice.begin(), dice.end());
    return id_.at(dice);
  }

  // Subset id of the dice selected by keep_mask (bit i keeps dice[i])
  int KeptSubset(const std::vector<int>& dice, int keep_mask) const {
    std::vector<int> kept;
    for (int i = 0; i < static_cast<int>(dice.size()); ++i) {
      if ((keep_mask >> i) & 1) kept.push_back(dice[i]);
    }
    return SubsetId(kept);
  }

  // Distinct subset ids that can be kept from outcome d
  const std::vector<int>& Keeps(int d) const { return keeps_[d]; }

  // exp[sid] = expected value of keeping subset sid and rerolling the rest,
  // given values[d] for each five-dice outcome. exp has kNumSubsets entries.
  void Expectations(const float* values, double* exp) const {
    for (int d = 0; d < kNumOutcomes; ++d) exp[kFirst5 + d] = values[d];
    for (int i = kFirst5 - 1; i >= 0; --i) {
      const auto& ch = children_[i];
      exp[i] = (exp[ch[0]] + exp[ch[1]] + exp[ch[2]] +
                exp[ch[3]] + exp[ch[4]] + exp[ch[5]]) * (1.0 / 6.0);
    }
  }

  // One reroll decision: out[d] = best keep from outcome d. exp is scratch.
  void Reroll(const float* in, float* out, double* exp) const {
    Expectations(in, exp);
    for (int d = 0; d < kNumOutcomes; ++d) {
      double best = exp[keeps_[d][0]];
      for (int sid : keeps_[d]) best = std::max(best, exp[sid]);
      out[d] = static_cast<float>(best);
    }
  }

  // Expected value of rolling all five dice. exp is scratch.
  double InitialRoll(const float* values, double* exp) const {
    Expectations(values, exp);
    return exp[0];
  }

 private:
  std::vector<std::vector<int>> dice_;
  std::map<std::vector<int>, int> id_;
  std::array<std::array<int, 6>, kFirst5> children_;
  std::vector<std::vector<int>> keeps_;

  void Generate(std::vector<int>& current, int start_face, int size) {
    if (static_cast<int>(current.size()) == size) {
      id_[current] = dice_.size();
      dice_.push_back(current);
      return;
    }
    for (int f = start_face; f <= 6; ++f) {
      current.push_back(f);
      Generate(current, f, size);
      current.pop_back();
    }
  }
};

#endif  // YAHTZEE_KEEP_LATTICE_H
