/**
 * Yahtzee Exact (Rational) EV Solver
 *
 * Computes the optimal solo Yahtzee expected value with exact rational
 * arithmetic. Same game model as yahtzee_optimized_solver.cpp (same scoring,
 * joker and bonus rules), but no floating point anywhere in the DP.
 *
 * WHY NO GENERAL FRACTIONS ARE NEEDED:
 * ===================================
 * Every probability in a turn is a product of 1/6 factors (one per die
 * rolled), and max() just selects one of its arguments. So with k turns
 * remaining, every board EV is an integer divided by D_k = 6^(15k):
 *
 *   term value  (after scoring)   : numerator / D_{k-1}
 *   roll2 value (1 roll left)     : numerator / (D_{k-1} * 6^5)
 *   roll1 value (2 rolls left)    : numerator / (D_{k-1} * 6^10)
 *   board value (start of turn)   : numerator / (D_{k-1} * 6^15) = D_k
 *
 * Within a layer all values share one denominator, so the DP runs purely on
 * unsigned integer numerators: addition, multiplication by small constants
 * and integer comparison. No gcd, no cross-multiplied fraction compares, no
 * heap allocation. The numerator is bounded by 2048 * 6^195 (< 2^517), so
 * fixed-width limb arrays of at most 9 x 64 bits suffice; each layer is
 * instantiated with the minimum number of limbs it needs.
 *
 * FAST REROLL STAGE (keep lattice):
 * =================================
 * Rerolling the non-kept dice is the same as rolling them one at a time, so
 * for a kept multiset K with |K| < 5:
 *
 *   E[K] = (1/6) * sum_{f=1..6} E[K + {f}]
 *
 * With S[K] = 6^(5-|K|) * E[K] this is S[K] = sum_f S[K + {f}] (pure
 * additions, 210 x 6 of them), and T[K] = 6^|K| * S[K] = 6^5 * E[K] puts all
 * keeps on the common denominator for comparison.
 *
 * OUTPUT:
 * =======
 * yahtzee_rational_v1.bin: header + for every board state
 * (mask, upper, y_state), indexed as in the float solver, a 9-limb
 * little-endian numerator. The denominator is implied: 6^(15k) where
 * k = 13 - popcount(mask) - (y_state > 0). Unreachable upper scores are 0.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// -------------------------------------------------------------------------
// CONSTANTS
// -------------------------------------------------------------------------

const int NUM_CATS = 13;
const int CAT_YAHTZEE = 12;
const int NUM_BOARD_MASKS = (1 << (NUM_CATS - 1));
const int MAX_UPPER_SCORE = 64;
const int NUM_Y_STATES = 3;
const int NUM_DICE_COMBOS = 252;
const int BOARD_STATE_SIZE = NUM_BOARD_MASKS * MAX_UPPER_SCORE * NUM_Y_STATES;

// Max numerator width: 2048 * 6^195 < 2^517 -> 9 limbs
const int MAX_LIMBS = 9;
// Largest single-turn reward: 100 (Yahtzee bonus) + 30 (sixes) + 35 (bonus)
const int MAX_REWARD = 200;

// -------------------------------------------------------------------------
// FIXED-WIDTH UNSIGNED INTEGER
// -------------------------------------------------------------------------

using u64 = uint64_t;
using u128 = unsigned __int128;

template <int L>
struct UInt {
  u64 w[L];
};

template <int L>
inline void SetZero(UInt<L>& a) {
  for (int i = 0; i < L; ++i) a.w[i] = 0;
}

// a += b
template <int L>
inline void AddTo(UInt<L>& a, const UInt<L>& b) {
  u64 c = 0;
  for (int i = 0; i < L; ++i) {
    u128 s = static_cast<u128>(a.w[i]) + b.w[i] + c;
    a.w[i] = static_cast<u64>(s);
    c = static_cast<u64>(s >> 64);
  }
}

// r = a + b
template <int L>
inline void Add(UInt<L>& r, const UInt<L>& a, const UInt<L>& b) {
  u64 c = 0;
  for (int i = 0; i < L; ++i) {
    u128 s = static_cast<u128>(a.w[i]) + b.w[i] + c;
    r.w[i] = static_cast<u64>(s);
    c = static_cast<u64>(s >> 64);
  }
}

// r = a * m  (m small)
template <int L>
inline void MulSmall(UInt<L>& r, const UInt<L>& a, u64 m) {
  u64 carry = 0;
  for (int i = 0; i < L; ++i) {
    u128 p = static_cast<u128>(a.w[i]) * m + carry;
    r.w[i] = static_cast<u64>(p);
    carry = static_cast<u64>(p >> 64);
  }
}

template <int L>
inline bool Less(const UInt<L>& a, const UInt<L>& b) {
  for (int i = L - 1; i >= 0; --i) {
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
  }
  return false;
}

template <int L>
inline void Load(UInt<L>& a, const u64* src) {
  std::memcpy(a.w, src, sizeof(u64) * L);
}

template <int L>
inline void Store(u64* dst, const UInt<L>& a) {
  std::memcpy(dst, a.w, sizeof(u64) * L);
  for (int i = L; i < MAX_LIMBS; ++i) dst[i] = 0;
}

// -------------------------------------------------------------------------
// ARBITRARY-WIDTH UNSIGNED INTEGER (output / reporting only)
// -------------------------------------------------------------------------

struct BigUInt {
  std::vector<u64> w;  // little-endian, no trailing zeros

  explicit BigUInt(u64 v = 0) {
    if (v) w.push_back(v);
  }
  BigUInt(const u64* limbs, int n) : w(limbs, limbs + n) { Trim(); }

  void Trim() {
    while (!w.empty() && w.back() == 0) w.pop_back();
  }
  bool IsZero() const { return w.empty(); }

  void MulSmall(u64 m) {
    u64 carry = 0;
    for (auto& x : w) {
      u128 p = static_cast<u128>(x) * m + carry;
      x = static_cast<u64>(p);
      carry = static_cast<u64>(p >> 64);
    }
    if (carry) w.push_back(carry);
    Trim();
  }

  // Divides in place, returns remainder
  u64 DivSmall(u64 d) {
    u128 rem = 0;
    for (int i = static_cast<int>(w.size()) - 1; i >= 0; --i) {
      u128 cur = (rem << 64) | w[i];
      w[i] = static_cast<u64>(cur / d);
      rem = cur % d;
    }
    Trim();
    return static_cast<u64>(rem);
  }

  u64 ModSmall(u64 d) const {
    u128 rem = 0;
    for (int i = static_cast<int>(w.size()) - 1; i >= 0; --i) {
      rem = ((rem << 64) | w[i]) % d;
    }
    return static_cast<u64>(rem);
  }

  std::string ToDecimal() const {
    if (IsZero()) return "0";
    BigUInt t = *this;
    std::vector<u64> chunks;
    const u64 kBase = 1000000000000000000ULL;  // 10^18
    while (!t.IsZero()) chunks.push_back(t.DivSmall(kBase));
    std::string s = std::to_string(chunks.back());
    for (int i = static_cast<int>(chunks.size()) - 2; i >= 0; --i) {
      std::string part = std::to_string(chunks[i]);
      s += std::string(18 - part.size(), '0') + part;
    }
    return s;
  }

  long double ToLongDouble() const {
    long double r = 0;
    for (int i = static_cast<int>(w.size()) - 1; i >= 0; --i) {
      r = r * 18446744073709551616.0L + static_cast<long double>(w[i]);
    }
    return r;
  }
};

// -------------------------------------------------------------------------
// DICE & SCORING (identical rules to yahtzee_optimized_solver.cpp)
// -------------------------------------------------------------------------

class DiceIndexer {
 public:
  std::map<std::vector<int>, int> dice_to_id;
  std::vector<std::vector<int>> id_to_dice;

  explicit DiceIndexer(int num_dice) {
    std::vector<int> current;
    Generate(current, 1, num_dice);
  }

  int GetId(std::vector<int> dice) const {
    std::sort(dice.begin(), dice.end());
    return dice_to_id.at(dice);
  }

 private:
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

bool HasNOfAKind(const std::vector<int>& dice, int n) {
  int counts[7] = {0};
  for (int d : dice) counts[d]++;
  for (int c : counts) if (c >= n) return true;
  return false;
}

bool IsFullHouse(const std::vector<int>& dice) {
  int counts[7] = {0};
  for (int d : dice) counts[d]++;
  bool has3 = false, has2 = false;
  for (int c : counts) {
    if (c == 5) return true;
    if (c == 3) has3 = true;
    if (c == 2) has2 = true;
  }
  return has3 && has2;
}

bool IsStraight(const std::vector<int>& dice, int len) {
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

int CalculateScore(int category, const std::vector<int>& dice) {
  int sum = std::accumulate(dice.begin(), dice.end(), 0);
  if (category < 6) {
    return std::count(dice.begin(), dice.end(), category + 1) * (category + 1);
  }
  switch (category) {
    case 6: return HasNOfAKind(dice, 3) ? sum : 0;
    case 7: return HasNOfAKind(dice, 4) ? sum : 0;
    case 8: return IsFullHouse(dice) ? 25 : 0;
    case 9: return IsStraight(dice, 4) ? 30 : 0;
    case 10: return IsStraight(dice, 5) ? 40 : 0;
    case 11: return sum;
    case 12: return HasNOfAKind(dice, 5) ? 50 : 0;
  }
  return 0;
}

// -------------------------------------------------------------------------
// SOLVER
// -------------------------------------------------------------------------

class RationalSolver {
 public:
  RationalSolver();
  void Solve();
  void SaveToFile(const std::string& filename) const;
  void Report() const;

 private:
  DiceIndexer indexer_5_;

  // Keep lattice over all 462 multisets of 0..5 dice. Ids are ordered by
  // size; the 252 five-dice multisets take ids [kFirst5, 462) in the same
  // order as indexer_5_ (outcome d has lattice id kFirst5 + d).
  static const int kNumSubsets = 462;
  static const int kFirst5 = kNumSubsets - NUM_DICE_COMBOS;  // 210
  std::array<int, kNumSubsets> subset_size_;
  std::array<std::array<int, 6>, kFirst5> children_;  // K + {f}
  std::vector<std::vector<int>> keeps_;  // per outcome: distinct kept ids

  // Per-outcome scoring tables
  std::array<std::array<int, NUM_CATS>, NUM_DICE_COMBOS> score_;
  std::array<int, NUM_DICE_COMBOS> yahtzee_face_;  // 0 if not a Yahtzee

  // reachable_[upper_mask6] bit u set if upper total u (capped) is reachable
  std::array<u64, 64> reachable_;

  // Board EV numerators, MAX_LIMBS per state
  std::vector<u64> board_;

  static int TurnsLeft(int mask, int y_state) {
    return 13 - __builtin_popcount(mask) - (y_state > 0 ? 1 : 0);
  }
  static int LimbsFor(int turns_left) {
    // numerator < 2^11 * 6^(15k); +1 bit slack
    int bits = static_cast<int>(std::ceil(11 + 15.0 * turns_left * std::log2(6.0))) + 1;
    return (bits + 63) / 64;
  }
  static size_t BoardIndex(int mask, int upper, int y_state) {
    return (static_cast<size_t>(mask) * MAX_UPPER_SCORE + upper) * NUM_Y_STATES + y_state;
  }
  const u64* BoardPtr(int mask, int upper, int y_state) const {
    return &board_[BoardIndex(mask, upper, y_state) * MAX_LIMBS];
  }

  void BuildLattice();
  void BuildScores();
  void BuildReachable();

  template <int L>
  void SolveLayer(const std::vector<std::pair<int, int>>& states, int turns_left);

  template <int L>
  void RerollStage(const UInt<L>* in, UInt<L>* out, UInt<L>* S) const;
};

RationalSolver::RationalSolver() : indexer_5_(5) {
  BuildLattice();
  BuildScores();
  BuildReachable();
  board_.assign(static_cast<size_t>(BOARD_STATE_SIZE) * MAX_LIMBS, 0);
  std::cout << "Board table: "
            << board_.size() * sizeof(u64) / (1024 * 1024) << " MB" << std::endl;
}

void RationalSolver::BuildLattice() {
  std::map<std::vector<int>, int> id_of;
  std::vector<std::vector<int>> all;
  for (int n = 0; n <= 5; ++n) {
    DiceIndexer tmp(n);
    for (const auto& d : tmp.id_to_dice) {
      id_of[d] = all.size();
      all.push_back(d);
    }
  }
  for (int i = 0; i < kNumSubsets; ++i) subset_size_[i] = all[i].size();
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
    if (id_of.at(indexer_5_.id_to_dice[d]) != kFirst5 + d) {
      std::cerr << "FATAL: lattice ids do not match dice ids" << std::endl;
      std::exit(1);
    }
  }
  for (int i = 0; i < kFirst5; ++i) {
    for (int f = 1; f <= 6; ++f) {
      std::vector<int> c = all[i];
      c.push_back(f);
      std::sort(c.begin(), c.end());
      children_[i][f - 1] = id_of.at(c);
    }
  }
  keeps_.resize(NUM_DICE_COMBOS);
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
    const auto& dice = indexer_5_.id_to_dice[d];
    for (int m = 0; m < 32; ++m) {
      std::vector<int> sub;
      for (int b = 0; b < 5; ++b) {
        if ((m >> b) & 1) sub.push_back(dice[b]);
      }
      keeps_[d].push_back(id_of.at(sub));
    }
    std::sort(keeps_[d].begin(), keeps_[d].end());
    keeps_[d].erase(std::unique(keeps_[d].begin(), keeps_[d].end()), keeps_[d].end());
  }
}

void RationalSolver::BuildScores() {
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
    const auto& dice = indexer_5_.id_to_dice[d];
    for (int c = 0; c < NUM_CATS; ++c) score_[d][c] = CalculateScore(c, dice);
    yahtzee_face_[d] = HasNOfAKind(dice, 5) ? dice[0] : 0;
  }
}

void RationalSolver::BuildReachable() {
  reachable_[0] = 1;  // only upper = 0
  for (int m = 1; m < 64; ++m) {
    int i = __builtin_ctz(m);
    u64 prev = reachable_[m & (m - 1)];
    u64 r = 0;
    for (int u = 0; u < MAX_UPPER_SCORE; ++u) {
      if (!((prev >> u) & 1)) continue;
      for (int c = 0; c <= 5; ++c) r |= 1ULL << std::min(63, u + c * (i + 1));
    }
    reachable_[m] = r;
  }
}

// One reroll decision: in[d] over denominator X, out[d] over X * 6^5.
template <int L>
void RationalSolver::RerollStage(const UInt<L>* in, UInt<L>* out, UInt<L>* S) const {
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) S[kFirst5 + d] = in[d];
  for (int i = kFirst5 - 1; i >= 0; --i) {
    const auto& ch = children_[i];
    Add(S[i], S[ch[0]], S[ch[1]]);
    AddTo(S[i], S[ch[2]]);
    AddTo(S[i], S[ch[3]]);
    AddTo(S[i], S[ch[4]]);
    AddTo(S[i], S[ch[5]]);
  }
  // T[K] = 6^|K| * S[K] (in place)
  static const u64 kPow6[6] = {1, 6, 36, 216, 1296, 7776};
  for (int i = 1; i < kNumSubsets; ++i) MulSmall(S[i], S[i], kPow6[subset_size_[i]]);
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
    const UInt<L>* best = &S[keeps_[d][0]];
    for (size_t j = 1; j < keeps_[d].size(); ++j) {
      const UInt<L>* c = &S[keeps_[d][j]];
      if (Less(*best, *c)) best = c;
    }
    out[d] = *best;
  }
}

template <int L>
void RationalSolver::SolveLayer(const std::vector<std::pair<int, int>>& states,
                                int turns_left) {
  // reward_scaled[r] = r * D_{k-1}, D_{k-1} = 6^(15(k-1))
  std::vector<UInt<L>> reward_scaled(MAX_REWARD + 1);
  {
    UInt<L> one;
    SetZero(one);
    one.w[0] = 1;
    for (int t = 0; t < 15 * (turns_left - 1); ++t) MulSmall(one, one, 6);
    for (int r = 0; r <= MAX_REWARD; ++r) MulSmall(reward_scaled[r], one, r);
  }

  #pragma omp parallel
  {
    std::vector<UInt<L>> term(NUM_DICE_COMBOS), r2(NUM_DICE_COMBOS),
        r1(NUM_DICE_COMBOS), S(kNumSubsets);

    #pragma omp for schedule(dynamic, 1)
    for (size_t si = 0; si < states.size(); ++si) {
      const int mask = states[si].first;
      const int y_state = states[si].second;
      const u64 reach = reachable_[mask & 63];

      for (int upper = 0; upper < MAX_UPPER_SCORE; ++upper) {
        if (!((reach >> upper) & 1)) continue;

        // value of placing reward r and moving to (nm, nu, ny)
        auto option = [&](int r, int nm, int nu, int ny, UInt<L>& v) {
          UInt<L> next;
          Load(next, BoardPtr(nm, nu, ny));
          Add(v, reward_scaled[r], next);
        };

        // Stage 1: best scoring move per outcome (over D_{k-1})
        for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
          UInt<L>& best = term[d];
          bool have = false;
          UInt<L> v;
          auto consider = [&](int r, int nm, int nu, int ny) {
            option(r, nm, nu, ny, v);
            if (!have || Less(best, v)) {
              best = v;
              have = true;
            }
          };
          const auto& sc = score_[d];
          const int yface = yahtzee_face_[d];

          if (yface && y_state > 0) {
            // Joker rules
            int bonus = (y_state == 2) ? 100 : 0;
            int ui = yface - 1;
            if (!((mask >> ui) & 1)) {
              int s = sc[ui];
              int nu = std::min(63, upper + s);
              int ub = (upper < 63 && nu >= 63) ? 35 : 0;
              consider(bonus + s + ub, mask | (1 << ui), nu, y_state);
            } else {
              bool lower = false;
              for (int c = 6; c < 12; ++c) {
                if ((mask >> c) & 1) continue;
                lower = true;
                int s = sc[c];
                if (c == 8) s = 25;
                else if (c == 9) s = 30;
                else if (c == 10) s = 40;
                consider(bonus + s, mask | (1 << c), upper, y_state);
              }
              if (!lower) {
                for (int c = 0; c < 6; ++c) {
                  if ((mask >> c) & 1) continue;
                  consider(bonus, mask | (1 << c), upper, y_state);
                }
              }
            }
          } else {
            // Free choice; a Yahtzee with the box open may also go elsewhere
            if (yface) consider(50, mask, upper, 2);
            for (int c = 0; c < 12; ++c) {
              if ((mask >> c) & 1) continue;
              int s = sc[c];
              int nu = upper, ub = 0;
              if (c < 6) {
                nu = std::min(63, upper + s);
                if (upper < 63 && nu >= 63) ub = 35;
              }
              consider(s + ub, mask | (1 << c), nu, y_state);
            }
            if (y_state == 0 && !yface) consider(0, mask, upper, 1);
          }
        }

        // Stage 2/3: two reroll decisions
        RerollStage<L>(term.data(), r2.data(), S.data());
        RerollStage<L>(r2.data(), r1.data(), S.data());

        // Stage 4: initial roll of 5 dice; S[0] = 6^5 * E over D_{k-1}*6^10
        for (int d = 0; d < NUM_DICE_COMBOS; ++d) S[kFirst5 + d] = r1[d];
        for (int i = kFirst5 - 1; i >= 0; --i) {
          const auto& ch = children_[i];
          Add(S[i], S[ch[0]], S[ch[1]]);
          for (int f = 2; f < 6; ++f) AddTo(S[i], S[ch[f]]);
        }
        Store(&board_[BoardIndex(mask, upper, y_state) * MAX_LIMBS], S[0]);
      }
    }
  }
}

void RationalSolver::Solve() {
  std::cout << "Starting exact optimization (reverse induction)...\n" << std::endl;

  // Terminal layer (13 filled) is all zero already.
  for (int total_filled = 12; total_filled >= 0; --total_filled) {
    auto start = std::chrono::high_resolution_clock::now();
    int turns_left = 13 - total_filled;
    int limbs = LimbsFor(turns_left);

    std::vector<std::pair<int, int>> states;
    for (int m = 0; m < NUM_BOARD_MASKS; ++m) {
      int pc = __builtin_popcount(m);
      if (pc == total_filled) states.push_back({m, 0});
      if (pc + 1 == total_filled) {
        states.push_back({m, 1});
        states.push_back({m, 2});
      }
    }

    std::cout << "Layer " << std::setw(2) << total_filled << " filled ("
              << std::setw(4) << states.size() << " boards, " << limbs
              << " limbs)..." << std::flush;

    switch (limbs) {
      case 1: SolveLayer<1>(states, turns_left); break;
      case 2: SolveLayer<2>(states, turns_left); break;
      case 3: SolveLayer<3>(states, turns_left); break;
      case 4: SolveLayer<4>(states, turns_left); break;
      case 5: SolveLayer<5>(states, turns_left); break;
      case 6: SolveLayer<6>(states, turns_left); break;
      case 7: SolveLayer<7>(states, turns_left); break;
      case 8: SolveLayer<8>(states, turns_left); break;
      case 9: SolveLayer<9>(states, turns_left); break;
      default:
        std::cerr << "FATAL: unsupported limb count " << limbs << std::endl;
        std::exit(1);
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    std::cout << " (" << ms << " ms)" << std::endl;
  }
  std::cout << "\nOptimization complete!" << std::endl;
}

void RationalSolver::Report() const {
  // Start state: numerator / 6^195 = numerator / (2^195 * 3^195)
  BigUInt num(BoardPtr(0, 0, 0), MAX_LIMBS);
  int e2 = 195, e3 = 195;
  while (e2 > 0 && num.ModSmall(2) == 0) { num.DivSmall(2); --e2; }
  while (e3 > 0 && num.ModSmall(3) == 0) { num.DivSmall(3); --e3; }
  BigUInt den(1);
  for (int i = 0; i < e2; ++i) den.MulSmall(2);
  for (int i = 0; i < e3; ++i) den.MulSmall(3);

  // Decimal expansion: floor(num * 10^40 / den) via small divisions
  const int kDigits = 40;
  BigUInt scaled = num;
  for (int i = 0; i < kDigits; ++i) scaled.MulSmall(10);
  for (int i = 0; i < e2; ++i) scaled.DivSmall(2);
  for (int i = 0; i < e3; ++i) scaled.DivSmall(3);
  std::string digits = scaled.ToDecimal();
  if (static_cast<int>(digits.size()) <= kDigits) {
    digits = std::string(kDigits + 1 - digits.size(), '0') + digits;
  }
  std::string dec = digits.substr(0, digits.size() - kDigits) + "." +
                    digits.substr(digits.size() - kDigits);

  std::cout << "\n=== Exact optimal solo EV (start of game) ===" << std::endl;
  std::cout << "Reduced denominator: 2^" << e2 << " * 3^" << e3 << std::endl;
  std::cout << "Numerator   (" << num.ToDecimal().size() << " digits): "
            << num.ToDecimal() << std::endl;
  std::cout << "Denominator (" << den.ToDecimal().size() << " digits): "
            << den.ToDecimal() << std::endl;
  std::cout << "Decimal (truncated): " << dec << std::endl;
}

void RationalSolver::SaveToFile(const std::string& filename) const {
  std::cout << "\nSaving exact board table to " << filename << "..." << std::endl;
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Error: Could not open file for writing" << std::endl;
    return;
  }
  uint32_t magic = 0x54415259;  // "YRAT"
  uint32_t version = 1;
  uint32_t board_size = BOARD_STATE_SIZE;
  uint32_t limbs = MAX_LIMBS;
  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&board_size), sizeof(board_size));
  out.write(reinterpret_cast<const char*>(&limbs), sizeof(limbs));
  out.write(reinterpret_cast<const char*>(board_.data()), board_.size() * sizeof(u64));
  std::cout << "Saved " << board_.size() * sizeof(u64) / (1024 * 1024) << " MB" << std::endl;
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::cout << "=== Yahtzee Exact Rational EV Solver ===\n" << std::endl;
  std::string out_file = argc > 1 ? argv[1] : "yahtzee_rational_v1.bin";

  auto start = std::chrono::high_resolution_clock::now();
  RationalSolver solver;
  solver.Solve();
  solver.Report();
  solver.SaveToFile(out_file);
  auto secs = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  std::cout << "\nTotal time: " << secs << " seconds" << std::endl;
  return 0;
}
