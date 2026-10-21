/**
 * Yahtzee Strategy Query Interface
 * 
 * This class loads the precomputed start-of-turn EV table and provides
 * queries for optimal gameplay decisions. Mid-turn values are recomputed
 * from the board EVs on demand (see yahtzee_keep_lattice.h).
 * 
 * USAGE:
 * ======
 * YahtzeeStrategy strategy("yahtzee_optimal_v4.bin");
 * 
 * // Get optimal category to score in
 * int category = strategy.GetOptimalCategory(dice, mask, upper, y_state);
 * 
 * // Get optimal dice to keep (returns bitmask)
 * int keep_mask = strategy.GetOptimalKeep(dice, mask, upper, y_state, rolls_left);
 * 
 * PERFORMANCE:
 * ===========
 * - Load time: a few milliseconds (3 MB file)
 * - Category query: up to 13 lookups
 * - Keep query: 252 scoring evaluations plus 1-2 lattice passes (well under 1 ms)
 * - Memory: ~3 MB
 *
 * Accepts version 4 files (board EVs only) and version 3 files (board EVs
 * followed by mid-turn tables, which are ignored).
 */

#include "yahtzee_strategy.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <bitset>
#include <memory>
#include <map>
#include <cstring>
#include <chrono>
 
// -------------------------------------------------------------------------
// IMPLEMENTATION
// -------------------------------------------------------------------------
 
 YahtzeeStrategy::YahtzeeStrategy(const std::string& filename) 
     : indexer_(5) {
   std::cout << "Loading strategy from " << filename << "..." << std::flush;
   auto start_time = std::chrono::high_resolution_clock::now();
   
   std::ifstream in(filename, std::ios::binary);
   if (!in) {
     std::cerr << "\nError: Could not open strategy file" << std::endl;
     throw std::runtime_error("File open failed");
   }
   
   // Read header
   uint32_t magic, version, board_size, roll_size;
   in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
   in.read(reinterpret_cast<char*>(&version), sizeof(version));
   in.read(reinterpret_cast<char*>(&board_size), sizeof(board_size));
   in.read(reinterpret_cast<char*>(&roll_size), sizeof(roll_size));
   
   if (magic != 0x59414854) {
     std::cerr << "\nError: Invalid file format (bad magic number)" << std::endl;
     throw std::runtime_error("Invalid file format");
   }
   
   if (version != 3 && version != 4) {
     std::cerr << "\nError: Unsupported version " << version 
               << " (expected version 3 or 4)" << std::endl;
     throw std::runtime_error("Unsupported version");
   }
   
   // Version 3 files carry mid-turn tables after the board table; they are
   // not needed (mid-turn values are recomputed), so only the board is read.
   uint32_t expected_roll = (version == 3) ? BOARD_STATE_SIZE * NUM_DICE_COMBOS : 0;
   if (board_size != BOARD_STATE_SIZE || roll_size != expected_roll) {
     std::cerr << "\nError: Size mismatch in file" << std::endl;
     throw std::runtime_error("Size mismatch");
   }
   
   // Read board table
   board_ev_.resize(board_size);
   in.read(reinterpret_cast<char*>(board_ev_.data()), 
           board_size * sizeof(float));
   
   if (!in) {
     std::cerr << "\nError: Failed to read strategy data" << std::endl;
     throw std::runtime_error("Read failed");
   }
   
   in.close();
   
   auto end_time = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
       end_time - start_time).count();
   
   std::cout << " done (" << duration << " ms)" << std::endl;
   std::cout << "  Loaded " << board_size << " board states (version "
             << version << ")\n" << std::endl;
 }
 
 void YahtzeeStrategy::KeepExpectations(int mask, int upper, int y_state,
                                        int rolls_left, double* exp) const {
   // Value of each final roll: best scoring move
   float term[NUM_DICE_COMBOS];
   for (int d = 0; d < NUM_DICE_COMBOS; d++) {
     term[d] = GetBestScoringMove(d, mask, upper, y_state).first;
   }
   if (rolls_left == 1) {
     lattice_.Expectations(term, exp);
     return;
   }
   // rolls_left == 2: value of each outcome with one roll left, then keep
   float roll2[NUM_DICE_COMBOS];
   lattice_.Reroll(term, roll2, exp);
   lattice_.Expectations(roll2, exp);
 }
 
float YahtzeeStrategy::GetBoardEV(int mask, int upper, int y_state) const {
  if (IsTerminal(mask & FULL_MASK, y_state)) return 0.0f;  // All filled
  return board_ev_[BoardIndex(mask, upper, y_state)];
}
 
 std::pair<float, int> YahtzeeStrategy::GetBestScoringMove(
     int dice_id, int mask, int upper, int y_state) const {
   ScoringOption opts[MAX_SCORING_OPTIONS];
   int n = rules_.Options(dice_id, mask & FULL_MASK, upper, y_state, opts);
   float max_val = -1.0f;
   int best_cat = -1;
   for (int i = 0; i < n; ++i) {
     const ScoringOption& o = opts[i];
     float val = o.reward + GetBoardEV(o.next_mask, o.next_upper, o.next_y_state);
     if (val > max_val) {
       max_val = val;
       best_cat = o.category;
     }
   }
   if (n == 0) return {0.0f, -1};  // game over
   return {max_val, best_cat};
 }
 
int YahtzeeStrategy::GetOptimalCategory(std::vector<int> dice, int mask, 
                                        int upper, int y_state) {
  int dice_id = indexer_.GetId(dice);
  return GetBestScoringMove(dice_id, mask, upper, y_state).second;
}

std::vector<std::pair<int, float>> YahtzeeStrategy::GetTopCategories(
    std::vector<int> dice, int mask, int upper, int y_state, int top_n) const {
  std::vector<std::pair<float, int>> options;  // (expected_value, category)
  
  // Evaluate all open categories
  for (int c = 0; c < NUM_CATS; ++c) {
    bool is_open = false;
    if (c == CAT_YAHTZEE) {
      is_open = (y_state == 0);
    } else {
      is_open = !((mask >> c) & 1);
    }
    
    if (is_open) {
      int score = CalculateScore(c, dice);
      int next_u = upper;
      float u_bonus = 0.0f;
      
      if (c < 6) {
        next_u = std::min(63, upper + score);
        if (upper < 63 && next_u >= 63) u_bonus = 35.0f;
      }
      
      int next_y = y_state;
      if (c == CAT_YAHTZEE) {
        next_y = (score == 50) ? 2 : 1;
      }
      
      float val = score + u_bonus + GetBoardEV(mask | (1 << c), next_u, next_y);
      options.push_back({val, c});
    }
  }
  
  // Sort by expected value (descending)
  std::sort(options.rbegin(), options.rend());
  
  // Return top N as (category, expected_value)
  std::vector<std::pair<int, float>> result;
  for (size_t i = 0; i < std::min((size_t)top_n, options.size()); ++i) {
    result.push_back({options[i].second, options[i].first});
  }
  
  return result;
}
 
 int YahtzeeStrategy::GetOptimalKeep(std::vector<int> current_dice, int mask,
                                     int upper, int y_state, int rolls_left) {
   if (rolls_left == 0) return 31;  // Keep all (0b11111)
   if (rolls_left > 2) {
     std::cerr << "Error: rolls_left must be 1 or 2" << std::endl;
     return 31;
   }
   
  double exp[KeepLattice::kNumSubsets];
  KeepExpectations(mask, upper, y_state, rolls_left, exp);

  double best_ev = -1.0;
  int best_keep_mask = 0;

  // Try all 32 possible keep masks; on exact ties prefer keeping more dice
  for (int k_mask = 0; k_mask < 32; ++k_mask) {
    double ev = exp[lattice_.KeptSubset(current_dice, k_mask)];
    if (ev > best_ev ||
        (ev == best_ev && __builtin_popcount(k_mask) > __builtin_popcount(best_keep_mask))) {
      best_ev = ev;
      best_keep_mask = k_mask;
    }
  }
   
  return best_keep_mask;
}

std::vector<std::pair<int, float>> YahtzeeStrategy::GetTopKeepChoices(
    std::vector<int> current_dice, int mask, int upper, int y_state, 
    int rolls_left, int top_n) const {
  if (rolls_left == 0) {
    // If no rolls left, only option is to keep all
    return {{31, 0.0f}};
  }
  if (rolls_left > 2) {
    return {};
  }
  
  double exp[KeepLattice::kNumSubsets];
  KeepExpectations(mask, upper, y_state, rolls_left, exp);

  std::vector<std::pair<float, int>> options;  // (expected_value, keep_mask)
  for (int k_mask = 0; k_mask < 32; ++k_mask) {
    options.push_back(
        {static_cast<float>(exp[lattice_.KeptSubset(current_dice, k_mask)]), k_mask});
  }
  
  // Sort by expected value (descending)
  std::sort(options.rbegin(), options.rend());
  
  // Return top N as (keep_mask, expected_value)
  std::vector<std::pair<int, float>> result;
  for (size_t i = 0; i < std::min((size_t)top_n, options.size()); ++i) {
    result.push_back({options[i].second, options[i].first});
  }
  
  return result;
}

// -------------------------------------------------------------------------
// DEMO/TEST
// -------------------------------------------------------------------------
 
 #ifdef BUILD_DEMO
 int main() {
   std::cout << "=== Yahtzee Strategy Query Demo ===\n" << std::endl;
   
   try {
     YahtzeeStrategy strategy("yahtzee_optimal_v4.bin");
     
     // Test 1: Initial roll
     std::cout << "Test 1: Initial roll [1,1,2,4,5], 2 rolls left" << std::endl;
     std::vector<int> dice1 = {1, 1, 2, 4, 5};
     int keep_mask = strategy.GetOptimalKeep(dice1, 0, 0, 0, 2);
     std::cout << "  Optimal keep mask: " << std::bitset<5>(keep_mask) << std::endl;
     
     std::vector<int> kept;
     for (int i = 0; i < 5; i++) {
       if ((keep_mask >> i) & 1) kept.push_back(dice1[i]);
     }
     std::cout << "  Keep dice: ";
     for (int d : kept) std::cout << d << " ";
     std::cout << std::endl;
     
     // Test 2: Final roll
     std::cout << "\nTest 2: Final roll [3,3,3,4,5], must score" << std::endl;
     std::vector<int> dice2 = {3, 3, 3, 4, 5};
     int category = strategy.GetOptimalCategory(dice2, 0, 0, 0);
     const char* cat_names[] = {"Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
                                "3ofK", "4ofK", "FH", "SS", "LS", "Chance", "Yahtzee"};
     std::cout << "  Optimal category: " << cat_names[category] << std::endl;
     
     // Test 3: Expected value query
     std::cout << "\nTest 3: Expected value at start of game" << std::endl;
     float ev = strategy.GetBoardEV(0, 0, 0);
     std::cout << "  Expected score (optimal play): " << ev << std::endl;
     
     std::cout << "\n=== Demo complete ===" << std::endl;
     
   } catch (const std::exception& e) {
     std::cerr << "Error: " << e.what() << std::endl;
     return 1;
   }
   
   return 0;
 }
 #endif  // BUILD_DEMO