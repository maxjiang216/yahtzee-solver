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
 
 // Forward declarations
 bool HasNOfAKind(const std::vector<int>& dice, int n);
 
 // -------------------------------------------------------------------------
 // DICE INDEXER IMPLEMENTATION
 // -------------------------------------------------------------------------
 
 DiceIndexer::DiceIndexer(int num_dice) {
   std::vector<int> current;
   Generate(current, 1, num_dice);
 }
 
 int DiceIndexer::GetId(std::vector<int> dice) const {
   std::sort(dice.begin(), dice.end());
   return dice_to_id.at(dice);
 }
 
 const std::vector<int>& DiceIndexer::GetDice(int id) const {
   return id_to_dice[id];
 }
 
 void DiceIndexer::Generate(std::vector<int>& current, int start_face, int max_depth) {
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
 
 // -------------------------------------------------------------------------
 // SCORING FUNCTIONS (standalone implementations)
 // -------------------------------------------------------------------------

int GetSum(const std::vector<int>& dice) {
  return std::accumulate(dice.begin(), dice.end(), 0);
}

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
    if (c == 5) return true;  // Joker rule
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
  // Check if all categories are filled (12 non-Yahtzee + Yahtzee filled)
  int non_yahtzee_mask = mask & ~(1 << 12);  // Strip Yahtzee bit
  if (non_yahtzee_mask == (1 << 12) - 1 && y_state > 0) return 0.0f;  // All filled
  return board_ev_[BoardIndex(mask, upper, y_state)];
}
 
 std::pair<float, int> YahtzeeStrategy::GetBestScoringMove(
     int dice_id, int mask, int upper, int y_state) const {
   const std::vector<int>& dice = indexer_.GetDice(dice_id);
   bool is_yahtzee = HasNOfAKind(dice, 5);
   int yahtzee_num = is_yahtzee ? dice[0] : 0;
   
   float max_val = -1.0f;
   int best_cat = -1;
 
  // Special case: Yahtzee roll
  if (is_yahtzee) {
    if (y_state == 0) {
      // Yahtzee box open - score 50 points
      // NOTE: Yahtzee (category 12) is NOT in the 12-bit mask
      // We only update y_state, not the mask (consistent with solver)
      float val = 50.0f + GetBoardEV(mask, upper, 2);  // y_state becomes 2
      if (val > max_val) {
        max_val = val;
        best_cat = CAT_YAHTZEE;
      }
    } else {
       // Joker rules
       float bonus = (y_state == 2) ? 100.0f : 0.0f;
       int upper_idx = yahtzee_num - 1;
       bool upper_open = !((mask >> upper_idx) & 1);
 
       if (upper_open) {
         int score = CalculateScore(upper_idx, dice);
         int next_u = std::min(63, upper + score);
         float u_bonus = (upper < 63 && next_u >= 63) ? 35.0f : 0.0f;
         float val = bonus + score + u_bonus + 
                    GetBoardEV(mask | (1 << upper_idx), next_u, y_state);
         if (val > max_val) {
           max_val = val;
           best_cat = upper_idx;
         }
       } else {
         bool can_play_lower = false;
         for (int c = 6; c <= 11; ++c) {
           if (!((mask >> c) & 1)) {
             can_play_lower = true;
             int score = CalculateScore(c, dice);
             if (c == 8) score = 25;
             else if (c == 9) score = 30;
             else if (c == 10) score = 40;
             
             float val = bonus + score + GetBoardEV(mask | (1 << c), upper, y_state);
             if (val > max_val) {
               max_val = val;
               best_cat = c;
             }
           }
         }
         
         if (!can_play_lower) {
           for (int c = 0; c < 6; ++c) {
             if (!((mask >> c) & 1)) {
               float val = bonus + GetBoardEV(mask | (1 << c), upper, y_state);
               if (val > max_val) {
                 max_val = val;
                 best_cat = c;
               }
             }
           }
         }
       }
       return {max_val, best_cat};
     }
   }
 
  // Standard case: try all open categories (0-11 in 12-bit mask, plus Yahtzee separately)
  for (int c = 0; c < 12; ++c) {  // 0-11, NOT including Yahtzee (12)
    if (!((mask >> c) & 1)) {
      int score = CalculateScore(c, dice);
      int next_u = upper;
      float u_bonus = 0.0f;
      
      if (c < 6) {
        next_u = std::min(63, upper + score);
        if (upper < 63 && next_u >= 63) u_bonus = 35.0f;
      }
      
      // Note: c cannot be CAT_YAHTZEE (12) in this loop
      float val = score + u_bonus + GetBoardEV(mask | (1 << c), next_u, y_state);
      if (val > max_val) {
        max_val = val;
        best_cat = c;
      }
    }
  }
  
  // Special case: Yahtzee category when we have a non-Yahtzee roll
  // For a non-Yahtzee roll, Yahtzee box can only be scored as zero if still open
  if (y_state == 0 && !HasNOfAKind(dice, 5)) {  // Yahtzee still open and not a Yahtzee roll
    int score = 0; // Must be zero since not a Yahtzee
    int next_y = 1; // Mark Yahtzee as closed (used with no Yahtzee scored)
    float val = GetBoardEV(mask, upper, next_y);
    if (val > max_val) {
      max_val = val;
      best_cat = CAT_YAHTZEE;
    }
  }
   
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