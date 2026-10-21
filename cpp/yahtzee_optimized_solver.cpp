/**
 * Yahtzee Optimal Strategy Solver
 *
 * This program uses dynamic programming with reverse induction to compute the
 * optimal strategy for Yahtzee. It stores only the expected value at the
 * start of each turn; every mid-turn decision (which dice to keep, where to
 * score) follows from those values in microseconds, so readers recompute
 * them on demand with KeepLattice (yahtzee_keep_lattice.h).
 *
 * SPEED:
 * ======
 * - Rerolls use the keep lattice E[K] = (1/6) * sum_f E[K + {f}] (accumulated
 *   in double) instead of a 462x252 transition matrix
 * - Scoring choices come from precomputed tables (yahtzee_rules.h)
 *
 * OUTPUT (yahtzee_optimal_v4.bin, ~3 MB):
 * =======================================
 * 16-byte header {magic "YAHT", version 4, board_size, roll_size = 0}
 * followed by board_size floats, indexed by
 * board_index = (mask * 64 + upper) * 3 + y_state (see yahtzee_rules.h).
 * (Version 3 files additionally carried two 198M-entry mid-turn tables;
 * readers still accept them and ignore those.)
 */

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "yahtzee_keep_lattice.h"
#include "yahtzee_rules.h"

// -------------------------------------------------------------------------
// SOLVER CLASS
// -------------------------------------------------------------------------

/**
 * YahtzeeSolver: Computes optimal strategy using dynamic programming
 *
 * Stores only board_ev: the expected value at the start of each turn
 * (786K states, 3 MB as float).
 */
class YahtzeeSolver {
 public:
  YahtzeeSolver() : board_ev_(BOARD_STATE_SIZE, 0.0f) {}

  // Main solver - compute all EVs
  void Solve();

  // Save board EVs to binary file (format version 4)
  void SaveToFile(const std::string& filename) const;

 private:
  ScoringRules rules_;
  KeepLattice lattice_;
  std::vector<float> board_ev_;  // 786K entries x 4 bytes = 3.1 MB

  float GetBoardEV(int mask, int upper, int y_state) const {
    if (IsTerminal(mask, y_state)) return 0.0f;
    return board_ev_[BoardStateIndex(mask, upper, y_state)];
  }

  // Value of the best scoring move for a final roll
  float BestScoreValue(int dice_id, int mask, int upper, int y_state) const {
    ScoringOption opts[MAX_SCORING_OPTIONS];
    int n = rules_.Options(dice_id, mask, upper, y_state, opts);
    float best = n ? -1.0f : 0.0f;
    for (int i = 0; i < n; ++i) {
      const ScoringOption& o = opts[i];
      float val = o.reward + GetBoardEV(o.next_mask, o.next_upper, o.next_y_state);
      if (val > best) best = val;
    }
    return best;
  }
};

void YahtzeeSolver::Solve() {
  std::cout << "Starting optimization (reverse induction)...\n" << std::endl;

  // Process states in reverse order of TOTAL categories filled (including
  // Yahtzee): every turn fills one, so successors are always in the layer
  // just solved. Every upper total is solved, reachable or not.
  for (int total_filled = 13; total_filled >= 0; --total_filled) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Processing states with " << total_filled
              << " total categories filled..." << std::flush;

    std::vector<std::pair<int, int>> states;  // (mask, y_state)
    for (int m = 0; m < NUM_BOARD_MASKS; m++) {
      int mask_count = __builtin_popcount(m);
      if (mask_count == total_filled) states.push_back({m, 0});
      if (mask_count + 1 == total_filled) {
        states.push_back({m, 1});
        states.push_back({m, 2});
      }
    }

    #pragma omp parallel
    {
      float term_values[NUM_DICE_COMBOS];
      float roll2_values[NUM_DICE_COMBOS];
      float roll1_values[NUM_DICE_COMBOS];
      double subset_exp[KeepLattice::kNumSubsets];

      #pragma omp for schedule(dynamic, 1)
      for (size_t si = 0; si < states.size(); ++si) {
        int mask = states[si].first;
        int y_state = states[si].second;

        for (int upper = 0; upper < MAX_UPPER_SCORE; ++upper) {
          // Stage 1: value of each final roll (best scoring move)
          for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
            term_values[d] = BestScoreValue(d, mask, upper, y_state);
          }
          // Stage 2: one roll left
          lattice_.Reroll(term_values, roll2_values, subset_exp);
          // Stage 3: two rolls left
          lattice_.Reroll(roll2_values, roll1_values, subset_exp);
          // Stage 4: start of turn = expectation over the initial roll
          board_ev_[BoardStateIndex(mask, upper, y_state)] =
              static_cast<float>(lattice_.InitialRoll(roll1_values, subset_exp));
        }
      }
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << " (" << duration << " ms)" << std::endl;
  }

  std::cout << "\nOptimization complete!" << std::endl;
}

void YahtzeeSolver::SaveToFile(const std::string& filename) const {
  std::cout << "\nSaving strategy to " << filename << "..." << std::endl;

  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Error: Could not open file for writing" << std::endl;
    std::exit(1);
  }

  // Header (version 4: board EVs only, roll_size = 0)
  uint32_t header[4] = {
      0x59414854,  // "YAHT"
      4,           // version
      static_cast<uint32_t>(BOARD_STATE_SIZE),
      0};          // roll_size
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(board_ev_.data()),
            board_ev_.size() * sizeof(float));
  out.close();
  if (!out) {
    std::cerr << "Error: Failed to write " << filename << std::endl;
    std::exit(1);
  }

  std::cout << "Strategy saved successfully" << std::endl;
  std::cout << "  Board EVs: " << board_ev_.size() << " entries ("
            << (sizeof(header) + board_ev_.size() * sizeof(float)) / 1024
            << " KB)" << std::endl;
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------

int main() {
  std::cout << "=== Yahtzee Optimal Strategy Solver ===" << std::endl;
  std::cout << "Version: 4 (start-of-turn EVs only, float precision)\n" << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  YahtzeeSolver solver;
  solver.Solve();
  solver.SaveToFile("yahtzee_optimal_v4.bin");

  auto duration = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::high_resolution_clock::now() - start_time).count();
  std::cout << "\nTotal time: " << duration << " seconds" << std::endl;
  return 0;
}
