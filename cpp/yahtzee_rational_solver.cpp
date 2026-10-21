/**
 * Yahtzee Exact (Rational) EV Solver
 *
 * Computes the optimal solo Yahtzee expected value with exact rational
 * arithmetic, using the same rules as every other solver (yahtzee_rules.h)
 * and no floating point anywhere in the DP.
 *
 * With k turns left every board EV is an integer over D_k = 6^(15k) (see
 * yahtzee_exact_turn.h), so the DP runs on fixed-width integer numerators:
 * no gcd, no fraction comparisons, no heap allocation in the hot loop. Each
 * layer is compiled with the fewest 64-bit limbs it needs (1-9).
 * Only reachable upper totals are solved.
 *
 * OUTPUT:
 * =======
 * yahtzee_rational_v1.bin: 16-byte header {magic "YRAT", version 1,
 * board_size, limbs = 9}, then for every board state (indexed as in
 * yahtzee_rules.h) a 9-limb little-endian numerator. The denominator is
 * implied: 6^(15k) where k = turns left. Unreachable upper totals are 0.
 */

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "yahtzee_exact_turn.h"

class RationalSolver {
 public:
  RationalSolver() : board_(static_cast<size_t>(BOARD_STATE_SIZE) * EV_LIMBS, 0) {
    std::cout << "Board table: "
              << board_.size() * sizeof(u64) / (1024 * 1024) << " MB" << std::endl;
  }
  void Solve();
  void SaveToFile(const std::string& filename) const;
  void Report() const;

 private:
  ScoringRules rules_;
  KeepLattice lattice_;
  std::vector<u64> board_;  // EV numerators, EV_LIMBS per state

  template <int L>
  void SolveLayer(const std::vector<std::pair<int, int>>& states, int turns_left);
};

template <int L>
void RationalSolver::SolveLayer(const std::vector<std::pair<int, int>>& states,
                                int turns_left) {
  ExactTurnSolver<L> turn(rules_, lattice_, turns_left, board_.data());

  #pragma omp parallel
  {
    std::vector<ExactTurnScratch<L>> scratch(1);  // on the heap: up to ~90 KB

    #pragma omp for schedule(dynamic, 1)
    for (size_t si = 0; si < states.size(); ++si) {
      const int mask = states[si].first;
      const int y_state = states[si].second;
      for (int upper = 0; upper < MAX_UPPER_SCORE; ++upper) {
        if (!rules_.Reachable(mask, upper)) continue;
        UInt<L> ev;
        turn.Solve(mask, upper, y_state, scratch[0], ev, nullptr);
        Store(&board_[EV_LIMBS * BoardStateIndex(mask, upper, y_state)], ev, EV_LIMBS);
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
    int limbs = EvLimbs(turns_left);

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
  BigUInt num(&board_[0], EV_LIMBS);
  int e2 = 195, e3 = 195;
  ReduceBy2And3(num, e2, e3);
  BigUInt den(1);
  for (int i = 0; i < e2; ++i) den.MulSmall(2);
  for (int i = 0; i < e3; ++i) den.MulSmall(3);

  std::cout << "\n=== Exact optimal solo EV (start of game) ===" << std::endl;
  std::cout << "Reduced denominator: 2^" << e2 << " * 3^" << e3 << std::endl;
  std::cout << "Numerator   (" << num.ToDecimal().size() << " digits): "
            << num.ToDecimal() << std::endl;
  std::cout << "Denominator (" << den.ToDecimal().size() << " digits): "
            << den.ToDecimal() << std::endl;
  std::cout << "Decimal (truncated): " << DecimalOf(num, e2, e3, 40) << std::endl;
}

void RationalSolver::SaveToFile(const std::string& filename) const {
  std::cout << "\nSaving exact board table to " << filename << "..." << std::endl;
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Error: Could not open file for writing" << std::endl;
    std::exit(1);
  }
  uint32_t header[4] = {0x54415259,  // "YRAT"
                        1, static_cast<uint32_t>(BOARD_STATE_SIZE),
                        static_cast<uint32_t>(EV_LIMBS)};
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(board_.data()), board_.size() * sizeof(u64));
  out.close();
  if (!out) {
    std::cerr << "Error: Failed to write " << filename << std::endl;
    std::exit(1);
  }
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
