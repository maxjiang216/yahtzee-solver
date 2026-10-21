#ifndef YAHTZEE_RULES_H
#define YAHTZEE_RULES_H

/**
 * Yahtzee rules shared by every solver and the strategy queries.
 *
 * BOARD STATE: (mask, upper, y_state)
 * - mask: 12-bit bitmask of filled non-Yahtzee categories 0-11
 * - upper: upper section total, capped at 63 (63+ all earn the same bonus)
 * - y_state: 0 = Yahtzee box open, 1 = scored 0, 2 = scored 50
 * board_index = (mask * 64 + upper) * 3 + y_state
 *
 * CATEGORIES: 0-5 Ones..Sixes, 6 Three of a kind, 7 Four of a kind,
 * 8 Full house, 9 Small straight, 10 Large straight, 11 Chance, 12 Yahtzee
 *
 * SCORING CHOICES (ScoringOptions): with the Yahtzee box open, a Yahtzee may
 * be scored in any open box (or the Yahtzee box for 50). With the box filled,
 * joker rules apply: +100 bonus if it holds 50, and the matching upper box
 * must be used if open; otherwise any open lower box scores full value
 * (25/30/40 for FH/SS/LS); otherwise an open upper box scores 0. A non-Yahtzee
 * roll may also scratch the open Yahtzee box for 0. Options are listed in a
 * fixed order, and solvers take the first best one, so ties break the same
 * way everywhere.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

// -------------------------------------------------------------------------
// CONSTANTS
// -------------------------------------------------------------------------

const int NUM_CATS = 13;
const int NUM_UPPER = 6;  // Categories 0-5
const int CAT_YAHTZEE = 12;

// Non-Yahtzee categories: 2^12 = 4,096
const int NUM_BOARD_MASKS = (1 << (NUM_CATS - 1));
const int FULL_MASK = NUM_BOARD_MASKS - 1;

// Upper score capped at 63 because 63+ yields the same bonus
const int MAX_UPPER_SCORE = 64;

// Yahtzee State: 0 = Open, 1 = Filled(0), 2 = Filled(50)
const int NUM_Y_STATES = 3;

// Dice combinations (stars and bars: C(10,5) = 252)
const int NUM_DICE_COMBOS = 252;

const int BOARD_STATE_SIZE = NUM_BOARD_MASKS * MAX_UPPER_SCORE * NUM_Y_STATES;

// Largest score from any state onward (13 Yahtzees with all bonuses)
const int MAX_FUTURE_SCORE = 1575;

inline size_t BoardStateIndex(int mask, int upper, int y_state) {
  return (static_cast<size_t>(mask) * MAX_UPPER_SCORE + upper) * NUM_Y_STATES + y_state;
}

inline bool IsTerminal(int mask, int y_state) {
  return mask == FULL_MASK && y_state > 0;
}

// Turns remaining = categories still open (including Yahtzee)
inline int TurnsLeft(int mask, int y_state) {
  return 13 - __builtin_popcount(mask) - (y_state > 0 ? 1 : 0);
}

// -------------------------------------------------------------------------
// DICE INDEXER
// -------------------------------------------------------------------------

/**
 * DiceIndexer: Maps sorted dice combinations to unique integer IDs
 *
 * For 5 dice, there are 252 unique sorted combinations (multisets),
 * numbered in lexicographic order.
 */
class DiceIndexer {
 public:
  std::map<std::vector<int>, int> dice_to_id;
  std::vector<std::vector<int>> id_to_dice;

  explicit DiceIndexer(int num_dice) {
    std::vector<int> current;
    Generate(current, 1, num_dice);
  }

  // Convert dice vector to unique ID (sorts first)
  int GetId(std::vector<int> dice) const {
    std::sort(dice.begin(), dice.end());
    return dice_to_id.at(dice);
  }

  // Get dice combination for a given ID
  const std::vector<int>& GetDice(int id) const { return id_to_dice[id]; }

 private:
  // Recursively generate all non-decreasing sequences
  void Generate(std::vector<int>& current, int start_face, int max_depth) {
    if (static_cast<int>(current.size()) == max_depth) {
      dice_to_id[current] = id_to_dice.size();
      id_to_dice.push_back(current);
      return;
    }
    for (int i = start_face; i <= 6; ++i) {
      current.push_back(i);
      Generate(current, i, max_depth);
      current.pop_back();
    }
  }
};

// -------------------------------------------------------------------------
// SCORING
// -------------------------------------------------------------------------

inline int GetSum(const std::vector<int>& dice) {
  return std::accumulate(dice.begin(), dice.end(), 0);
}

inline bool HasNOfAKind(const std::vector<int>& dice, int n) {
  int counts[7] = {0};
  for (int d : dice) counts[d]++;
  for (int c : counts) if (c >= n) return true;
  return false;
}

inline bool IsFullHouse(const std::vector<int>& dice) {
  int counts[7] = {0};
  for (int d : dice) counts[d]++;
  bool has3 = false, has2 = false;
  for (int c : counts) {
    if (c == 5) return true;  // Five of a kind counts
    if (c == 3) has3 = true;
    if (c == 2) has2 = true;
  }
  return has3 && has2;
}

inline bool IsStraight(const std::vector<int>& dice, int len) {
  int unique[7] = {0};
  for (int d : dice) unique[d] = 1;
  int consec = 0;
  for (int i = 1; i <= 6; ++i) {
    if (unique[i]) consec++;
    else consec = 0;
    if (consec >= len) return true;
  }
  return false;
}

// Score for placing dice in a category (no joker adjustments)
inline int CalculateScore(int category, const std::vector<int>& dice) {
  if (category < 6) {
    return std::count(dice.begin(), dice.end(), category + 1) * (category + 1);
  }
  switch (category) {
    case 6: return HasNOfAKind(dice, 3) ? GetSum(dice) : 0;
    case 7: return HasNOfAKind(dice, 4) ? GetSum(dice) : 0;
    case 8: return IsFullHouse(dice) ? 25 : 0;
    case 9: return IsStraight(dice, 4) ? 30 : 0;
    case 10: return IsStraight(dice, 5) ? 40 : 0;
    case 11: return GetSum(dice);
    case 12: return HasNOfAKind(dice, 5) ? 50 : 0;
  }
  return 0;
}

// -------------------------------------------------------------------------
// SCORING OPTIONS
// -------------------------------------------------------------------------

// One way to score a final roll: points earned now and the resulting board
struct ScoringOption {
  int category;
  int reward;  // score + upper bonus + Yahtzee bonus
  int next_mask;
  int next_upper;
  int next_y_state;
};

// At most 13 options (one per open category)
const int MAX_SCORING_OPTIONS = 13;

class ScoringRules {
 public:
  ScoringRules() {
    DiceIndexer indexer(5);
    for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
      const std::vector<int>& dice = indexer.GetDice(d);
      for (int c = 0; c < NUM_CATS; ++c) score_[d][c] = CalculateScore(c, dice);
      yahtzee_face_[d] = HasNOfAKind(dice, 5) ? dice[0] : 0;
    }
    reachable_[0] = 1;  // no upper category filled -> upper = 0
    for (int m = 1; m < 64; ++m) {
      int i = __builtin_ctz(m);
      uint64_t prev = reachable_[m & (m - 1)];
      uint64_t r = 0;
      for (int u = 0; u < MAX_UPPER_SCORE; ++u) {
        if (!((prev >> u) & 1)) continue;
        for (int c = 0; c <= 5; ++c) r |= 1ULL << std::min(63, u + c * (i + 1));
      }
      reachable_[m] = r;
    }
  }

  int Score(int dice_id, int category) const { return score_[dice_id][category]; }

  // Face of a five-of-a-kind roll, or 0
  int YahtzeeFace(int dice_id) const { return yahtzee_face_[dice_id]; }

  // Whether upper total `upper` can occur with this mask's upper boxes filled
  bool Reachable(int mask, int upper) const {
    return (reachable_[mask & 63] >> upper) & 1;
  }

  // Fills out[] with every legal way to score dice_id, in tie-break order.
  // Returns the count (0 only for a terminal board).
  int Options(int dice_id, int mask, int upper, int y_state,
              ScoringOption* out) const {
    if (IsTerminal(mask, y_state)) return 0;
    const auto& sc = score_[dice_id];
    const int yface = yahtzee_face_[dice_id];
    int n = 0;

    if (yface && y_state > 0) {
      // Joker rules
      int bonus = (y_state == 2) ? 100 : 0;
      int ui = yface - 1;
      if (!((mask >> ui) & 1)) {
        int s = sc[ui];
        int nu = std::min(63, upper + s);
        int ub = (upper < 63 && nu >= 63) ? 35 : 0;
        out[n++] = {ui, bonus + s + ub, mask | (1 << ui), nu, y_state};
        return n;
      }
      for (int c = 6; c < 12; ++c) {
        if ((mask >> c) & 1) continue;
        int s = sc[c];
        if (c == 8) s = 25;
        else if (c == 9) s = 30;
        else if (c == 10) s = 40;
        out[n++] = {c, bonus + s, mask | (1 << c), upper, y_state};
      }
      if (n == 0) {
        for (int c = 0; c < 6; ++c) {
          if ((mask >> c) & 1) continue;
          out[n++] = {c, bonus, mask | (1 << c), upper, y_state};
        }
      }
      return n;
    }

    if (yface) out[n++] = {CAT_YAHTZEE, 50, mask, upper, 2};
    for (int c = 0; c < 12; ++c) {
      if ((mask >> c) & 1) continue;
      int s = sc[c];
      int nu = upper, ub = 0;
      if (c < 6) {
        nu = std::min(63, upper + s);
        if (upper < 63 && nu >= 63) ub = 35;
      }
      out[n++] = {c, s + ub, mask | (1 << c), nu, y_state};
    }
    if (y_state == 0 && !yface) out[n++] = {CAT_YAHTZEE, 0, mask, upper, 1};
    return n;
  }

 private:
  std::array<std::array<int, NUM_CATS>, NUM_DICE_COMBOS> score_;
  std::array<int, NUM_DICE_COMBOS> yahtzee_face_;
  std::array<uint64_t, 64> reachable_;
};

#endif  // YAHTZEE_RULES_H
