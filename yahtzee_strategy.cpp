/**
 * Yahtzee Strategy Query Interface
 * 
 * This class loads precomputed strategy tables and provides fast queries
 * for optimal gameplay decisions. Uses the optimized v3 format with
 * mid-turn storage for instant lookups.
 * 
 * USAGE:
 * ======
 * YahtzeeStrategy strategy("yahtzee_optimal_v3.bin");
 * 
 * // Get optimal category to score in
 * int category = strategy.GetOptimalCategory(dice, mask, upper, y_state);
 * 
 * // Get optimal dice to keep (returns bitmask)
 * int keep_mask = strategy.GetOptimalKeep(dice, mask, upper, y_state, rolls_left);
 * 
 * PERFORMANCE:
 * ===========
 * - Load time: ~2 seconds for 1.6 GB file
 * - Query time: O(1) lookups, ~5-50 nanoseconds
 * - Memory: 1.6 GB RAM
 * 
 * Can optionally use memory mapping for lower RAM usage with similar performance.
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
     : transitions_initialized_(false) {
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
   
   if (version != 3) {
     std::cerr << "\nError: Unsupported version " << version 
               << " (expected version 3)" << std::endl;
     throw std::runtime_error("Unsupported version");
   }
   
   if (board_size != BOARD_STATE_SIZE || roll_size != ROLL_STATE_SIZE) {
     std::cerr << "\nError: Size mismatch in file" << std::endl;
     throw std::runtime_error("Size mismatch");
   }
   
   // Read tables
   board_ev_.resize(board_size);
   roll1_ev_.resize(roll_size);
   roll2_ev_.resize(roll_size);
   
   in.read(reinterpret_cast<char*>(board_ev_.data()), 
           board_size * sizeof(float));
   in.read(reinterpret_cast<char*>(roll1_ev_.data()), 
           roll_size * sizeof(float));
   in.read(reinterpret_cast<char*>(roll2_ev_.data()), 
           roll_size * sizeof(float));
   
   if (!in) {
     std::cerr << "\nError: Failed to read strategy data" << std::endl;
     throw std::runtime_error("Read failed");
   }
   
   in.close();
   
   auto end_time = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
       end_time - start_time).count();
   
   std::cout << " done (" << duration << " ms)" << std::endl;
   std::cout << "  Loaded " << board_size << " board states" << std::endl;
   std::cout << "  Loaded " << roll_size << " roll states (x2)" << std::endl;
   std::cout << "  Memory usage: " 
             << (board_size + 2 * roll_size) * sizeof(float) / (1024 * 1024) 
             << " MB\n" << std::endl;
 }
 
 const DiceIndexer& YahtzeeStrategy::GetIndexer() const {
   if (!indexer_ptr_) {
     indexer_ptr_.reset(new DiceIndexer(5));
   }
   return *indexer_ptr_;
 }
 
 void YahtzeeStrategy::InitializeTransitions() const {
   if (transitions_initialized_) return;
   
   std::cout << "Initializing transition probabilities..." << std::flush;
   
   // Build all subsets (0-5 dice)
   for (int i = 0; i <= 5; i++) {
     DiceIndexer tmp(i);
     for (const auto& d : tmp.id_to_dice) {
       subset_to_id_[d] = all_subsets_dice_.size();
       all_subsets_dice_.push_back(d);
     }
   }
   
   // Build transition matrix
   trans_probs_.assign(all_subsets_dice_.size(), 
                       std::vector<float>(NUM_DICE_COMBOS, 0.0f));
   
  for (size_t i = 0; i < all_subsets_dice_.size(); i++) {
    std::vector<int> kept = all_subsets_dice_[i];
    int needed = 5 - kept.size();
    long long total = 1;
    for (int p = 0; p < needed; p++) total *= 6;
     
     for (long long j = 0; j < total; j++) {
       std::vector<int> current = kept;
       long long temp = j;
       for (int r = 0; r < needed; r++) {
         current.push_back((temp % 6) + 1);
         temp /= 6;
       }
       int final_id = GetIndexer().GetId(current);
       trans_probs_[i][final_id] += 1.0f;
     }
     
     for (int d = 0; d < NUM_DICE_COMBOS; d++) {
       trans_probs_[i][d] /= static_cast<float>(total);
     }
   }
   
   transitions_initialized_ = true;
   std::cout << " done" << std::endl;
 }
 
float YahtzeeStrategy::GetBoardEV(int mask, int upper, int y_state) const {
  // Check if all categories are filled (12 non-Yahtzee + Yahtzee filled)
  int non_yahtzee_mask = mask & ~(1 << 12);  // Strip Yahtzee bit
  if (non_yahtzee_mask == (1 << 12) - 1 && y_state > 0) return 0.0f;  // All filled
  return board_ev_[BoardIndex(mask, upper, y_state)];
}
 
 std::pair<float, int> YahtzeeStrategy::GetBestScoringMove(
     int dice_id, int mask, int upper, int y_state) const {
   std::vector<int> dice = GetIndexer().GetDice(dice_id);
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
  int dice_id = GetIndexer().GetId(dice);
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
   
  // Initialize transitions on first use
  InitializeTransitions();
  
  float best_ev = -1.0f;
   int best_keep_mask = 0;
 
   // Try all 32 possible keep masks
   for (int k_mask = 0; k_mask < 32; ++k_mask) {
     // Extract kept dice
     std::vector<int> kept;
     for (int i = 0; i < 5; i++) {
       if ((k_mask >> i) & 1) {
         kept.push_back(current_dice[i]);
       }
     }
     
     // Get subset ID for kept dice
     std::sort(kept.begin(), kept.end());
     int subset_id = subset_to_id_[kept];
     
     // Compute expected value using precomputed transition probabilities
     float ev = 0.0f;
     const auto& probs = trans_probs_[subset_id];
     
     for (int outcome_id = 0; outcome_id < NUM_DICE_COMBOS; outcome_id++) {
       if (probs[outcome_id] > 0) {
         float outcome_value;
         
         if (rolls_left == 2) {
           // After rerolling with 2 rolls left, outcome has 1 roll left -> use roll2_ev
           outcome_value = GetRoll2EV(mask, upper, y_state, outcome_id);
         } else {
           // After rerolling with 1 roll left, outcome has 0 rolls left -> terminal value
           outcome_value = GetBestScoringMove(outcome_id, mask, upper, y_state).first;
         }
         
         ev += probs[outcome_id] * outcome_value;
       }
     }
     
     if (ev > best_ev) {
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
  
  InitializeTransitions();
  
  std::vector<std::pair<float, int>> options;  // (expected_value, keep_mask)
  
  // Try all 32 possible keep masks
  for (int k_mask = 0; k_mask < 32; ++k_mask) {
    // Extract kept dice
    std::vector<int> kept;
    for (int i = 0; i < 5; i++) {
      if ((k_mask >> i) & 1) {
        kept.push_back(current_dice[i]);
      }
    }
    
    // Get subset ID for kept dice
    std::sort(kept.begin(), kept.end());
    int subset_id = subset_to_id_[kept];
    
    // Compute expected value using precomputed transition probabilities
    float ev = 0.0f;
    const auto& probs = trans_probs_[subset_id];
    
    for (int outcome_id = 0; outcome_id < NUM_DICE_COMBOS; outcome_id++) {
      if (probs[outcome_id] > 0) {
        float outcome_value;
        
        if (rolls_left == 2) {
          // After rerolling with 2 rolls left, outcome has 1 roll left -> use roll2_ev
          outcome_value = GetRoll2EV(mask, upper, y_state, outcome_id);
        } else {
          // After rerolling with 1 roll left, outcome has 0 rolls left -> terminal value
          outcome_value = GetBestScoringMove(outcome_id, mask, upper, y_state).first;
        }
        
        ev += probs[outcome_id] * outcome_value;
      }
    }
    
    options.push_back({ev, k_mask});
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
     YahtzeeStrategy strategy("yahtzee_optimal_v3.bin");
     
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