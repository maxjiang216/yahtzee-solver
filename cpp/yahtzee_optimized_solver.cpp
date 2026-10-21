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
 * - Scores are looked up from a precomputed 252x13 table
 *
 * OUTPUT (yahtzee_optimal_v4.bin, ~3 MB):
 * =======================================
 * 16-byte header {magic "YAHT", version 4, board_size, roll_size = 0}
 * followed by board_size floats. (Version 3 files additionally carried two
 * 198M-entry mid-turn tables; readers still accept them and ignore those.)
 * 
 * STATE REPRESENTATION:
 * ====================
 * Board state (excluding Yahtzee category):
 * - mask: 12-bit bitmask for non-Yahtzee categories (2^12 = 4,096)
 * - upper: Upper section score (0-63, capped at 63 for bonus)
 * - y_state: Yahtzee state (0=open, 1=filled with 0, 2=filled with 50)
 * 
 * INDEXING:
 * =========
 * board_index = (mask * 64 + upper) * 3 + y_state
 */

 #include <iostream>
 #include <vector>
 #include <numeric>
 #include <algorithm>
 #include <map>
 #include <cmath>
 #include <fstream>
 #include <cassert>
 #include <iomanip>
 #include <chrono>
 #include <bitset>
 #include <functional>
 #include <memory>
 #include <cstring>
 #include <array>
 #include <cstdint>
 #include <cstdlib>

 #include "yahtzee_keep_lattice.h"
 
 // -------------------------------------------------------------------------
 // CONSTANTS
 // -------------------------------------------------------------------------
 
 const int NUM_CATS = 13;
 const int NUM_UPPER = 6;  // Categories 0-5
 const int CAT_YAHTZEE = 12;
 
 // Non-Yahtzee categories: 2^12 = 4,096
 const int NUM_BOARD_MASKS = (1 << (NUM_CATS - 1));
 
 // Upper score capped at 63 because 63+ yields the same bonus
 const int MAX_UPPER_SCORE = 64;
 
 // Yahtzee State: 0 = Open, 1 = Filled(0), 2 = Filled(50)
 const int NUM_Y_STATES = 3;
 
 // Dice combinations (stars and bars: C(10,5) = 252)
 const int NUM_DICE_COMBOS = 252;
 
 // State space sizes
 const int BOARD_STATE_SIZE = NUM_BOARD_MASKS * MAX_UPPER_SCORE * NUM_Y_STATES;
 
 // -------------------------------------------------------------------------
 // DICE & INDEXING HELPERS
 // -------------------------------------------------------------------------
 
 /**
  * DiceIndexer: Maps sorted dice combinations to unique integer IDs
  * 
  * For 5 dice, there are 252 unique sorted combinations (multisets).
  * Uses stars and bars: C(6+5-1, 5) = C(10, 5) = 252
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
   const std::vector<int>& GetDice(int id) const {
     return id_to_dice[id];
   }
 
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
 // SCORING LOGIC
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
 
 /**
  * Calculate score for placing dice in a category
  * 
  * Categories:
  *   0-5:  Upper section (1s-6s)
  *   6:    Three of a kind
  *   7:    Four of a kind
  *   8:    Full house (25 points)
  *   9:    Small straight (30 points)
  *   10:   Large straight (40 points)
  *   11:   Chance
  *   12:   Yahtzee (50 points)
  */
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
 // SOLVER CLASS
 // -------------------------------------------------------------------------
 
 /**
  * YahtzeeSolver: Computes optimal strategy using dynamic programming
  * 
  * Stores only board_ev: the expected value at the start of each turn
  * (786K states, 3 MB as float). Mid-turn values (keep and scoring decisions)
  * follow from board_ev with a few lattice passes, so readers recompute them
  * on demand (see yahtzee_keep_lattice.h).
  */
 class YahtzeeSolver {
  public:
   YahtzeeSolver() : indexer_5_(5) {
     BuildScoreTables();
     board_ev_.assign(BOARD_STATE_SIZE, 0.0f);
   }
 
   // Main solver - compute all EVs
   void Solve();

   // Save board EVs to binary file (format version 4)
   void SaveToFile(const std::string& filename) const;
 
  private:
   // Dice indexing
   DiceIndexer indexer_5_;
   KeepLattice lattice_;

   // Precomputed per-outcome scores and Yahtzee face (0 if not a Yahtzee)
   std::vector<std::array<int, NUM_CATS>> score_;
   std::vector<int> yahtzee_face_;
 
   // Board EV table
   std::vector<float> board_ev_;   // 786K entries × 4 bytes = 3.1 MB
 
   // Indexing functions
   inline size_t BoardIndex(int mask, int upper, int y_state) const {
     return (mask * MAX_UPPER_SCORE + upper) * NUM_Y_STATES + y_state;
   }
 
   // EV accessors
   inline float GetBoardEV(int mask, int upper, int y_state) const {
     // Terminal check: all 12 non-Yahtzee categories filled AND Yahtzee scored
     if (mask == (1 << 12) - 1 && y_state > 0) return 0.0f;
     
     // Bounds check
     if (mask < 0 || mask >= NUM_BOARD_MASKS) {
       std::cerr << "GetBoardEV: mask out of bounds: " << mask << std::endl;
       return 0.0f;
     }
     
     return board_ev_[BoardIndex(mask, upper, y_state)];
   }
 
   inline void SetBoardEV(int mask, int upper, int y_state, float val) {
     board_ev_[BoardIndex(mask, upper, y_state)] = val;
   }
 
   // Helper functions
   void BuildScoreTables();
   
   // Get best scoring move for a dice outcome
   std::pair<float, int> GetBestScoringMove(int dice_id, int mask, int upper, 
                                            int y_state) const;
 };

 void YahtzeeSolver::BuildScoreTables() {
   for (int d = 0; d < NUM_DICE_COMBOS; d++) {
     assert(lattice_.SubsetId(indexer_5_.GetDice(d)) == KeepLattice::kFirst5 + d);
   }

   score_.resize(NUM_DICE_COMBOS);
   yahtzee_face_.resize(NUM_DICE_COMBOS);
   for (int d = 0; d < NUM_DICE_COMBOS; d++) {
     const std::vector<int>& dice = indexer_5_.GetDice(d);
     for (int c = 0; c < NUM_CATS; c++) score_[d][c] = CalculateScore(c, dice);
     yahtzee_face_[d] = HasNOfAKind(dice, 5) ? dice[0] : 0;
   }
 }
 
 std::pair<float, int> YahtzeeSolver::GetBestScoringMove(
     int dice_id, int mask, int upper, int y_state) const {
   // Terminal state check: all categories filled (game over)
   if (mask == (1 << 12) - 1 && y_state > 0) {
     // All 12 non-Yahtzee categories filled AND Yahtzee filled = game over
     return {0.0f, 0};  // EV is 0, category doesn't matter
   }
   
   // Bounds check
   if (dice_id < 0 || dice_id >= NUM_DICE_COMBOS) {
     std::cerr << "ERROR: Invalid dice_id " << dice_id << std::endl;
     return {0.0f, 0};
   }
   if (mask < 0 || mask >= NUM_BOARD_MASKS) {
     std::cerr << "ERROR: Invalid mask " << mask << std::endl;
     return {0.0f, 0};
   }
   if (upper < 0 || upper >= MAX_UPPER_SCORE) {
     std::cerr << "ERROR: Invalid upper " << upper << std::endl;
     return {0.0f, 0};
   }
   if (y_state < 0 || y_state >= NUM_Y_STATES) {
     std::cerr << "ERROR: Invalid y_state " << y_state << std::endl;
     return {0.0f, 0};
   }
   
   const auto& sc = score_[dice_id];
   int yahtzee_num = yahtzee_face_[dice_id];
   bool is_yahtzee = yahtzee_num != 0;
   
   float max_val = -1.0f;
   int best_cat = -1;
 
   // Special case: Yahtzee roll
   if (is_yahtzee) {
     if (y_state == 0) {
       // Yahtzee box open - always take it
       // NOTE: Yahtzee category (12) is NOT in the 12-bit mask
       // We only update y_state, not the mask
       float val = 50.0f + GetBoardEV(mask, upper, 2);  // y_state becomes 2
       if (val > max_val) {
         max_val = val;
         best_cat = CAT_YAHTZEE;
       }
     } else {
       // Joker rules apply
       float bonus = (y_state == 2) ? 100.0f : 0.0f;
       int upper_idx = yahtzee_num - 1;
       bool upper_open = !((mask >> upper_idx) & 1);
 
      if (upper_open) {
        // JOKER RULE: Must play in corresponding upper section if open
        // This is mandatory - no choice
        int score = sc[upper_idx];
        int next_u = std::min(63, upper + score);
        float u_bonus = (upper < 63 && next_u >= 63) ? 35.0f : 0.0f;
        float val = bonus + score + u_bonus + 
                   GetBoardEV(mask | (1 << upper_idx), next_u, y_state);
        max_val = val;
        best_cat = upper_idx;
      } else {
         // Try lower section categories (6-11 in 12-bit mask)
         bool can_play_lower = false;
         for (int c = 6; c < 12; ++c) {  // 6-11 (not including 12=Yahtzee)
           if (!((mask >> c) & 1)) {
             can_play_lower = true;
             int score = sc[c];
             // Joker guarantees full points
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
         
         // If no lower categories open, use upper for 0 points
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
       
       // Safety check: best_cat should always be set when joker rules apply
       // Exception: if all categories are filled (mask=4095), game is over
       if (best_cat < 0) {
         if (mask == (1 << 12) - 1) {
           // All categories filled - game over, return 0
           return {bonus, 0};  // Just return the bonus (if any) and category 0
         } else {
           std::cerr << "ERROR: Joker rules failed to find valid category!" << std::endl;
           std::cerr << "  mask=" << mask << " upper=" << upper << " y_state=" << y_state << std::endl;
           std::cerr << "  yahtzee_num=" << yahtzee_num << " upper_open=" << upper_open << std::endl;
           // Fallback: return bonus only (shouldn't happen in valid game states)
           return {bonus, 0};
         }
       }
       
       return {max_val, best_cat};
     }
   }
 
   // Standard case: try all open categories (0-11 in 12-bit mask)
   for (int c = 0; c < 12; ++c) {  // 0-11, NOT including Yahtzee (12)
     if (!((mask >> c) & 1)) {
       int score = sc[c];
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
   if (y_state == 0 && !is_yahtzee) {  // Yahtzee still open and not a Yahtzee roll
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
 
 void YahtzeeSolver::Solve() {
  std::cout << "Starting optimization (reverse induction)...\n" << std::endl;
  
  // Temporary arrays for computation
  std::vector<float> term_values(NUM_DICE_COMBOS);
  std::vector<float> roll2_values(NUM_DICE_COMBOS);
  std::vector<float> roll1_values(NUM_DICE_COMBOS);
  std::vector<double> subset_exp(KeepLattice::kNumSubsets);

  // Process states in reverse order of TOTAL categories filled (including Yahtzee)
  // Total filled = popcount(mask) + (y_state > 0 ? 1 : 0)
  // Range: 0 to 13
  for (int total_filled = 13; total_filled >= 0; --total_filled) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Processing states with " << total_filled 
              << " total categories filled..." << std::flush;
    
    // Collect all (mask, y_state) pairs with exactly 'total_filled' categories
    std::vector<std::pair<int, int>> states;  // (mask, y_state)
    
    for (int m = 0; m < (1 << 12); m++) {
      int mask_count = __builtin_popcount(m);  // Count bits in mask
      
      // y_state = 0: Yahtzee open, total = mask_count
      if (mask_count == total_filled) {
        states.push_back({m, 0});
      }
      
      // y_state = 1 or 2: Yahtzee filled, total = mask_count + 1
      if (mask_count + 1 == total_filled) {
        states.push_back({m, 1});
        states.push_back({m, 2});
      }
    }

    // Process all states
    #pragma omp parallel for firstprivate(term_values, roll2_values, roll1_values, subset_exp)
    for (size_t si = 0; si < states.size(); ++si) {
      int mask = states[si].first;
      int y_state = states[si].second;
      
      for (int upper = 0; upper < MAX_UPPER_SCORE; ++upper) {
        // Stage 1: Terminal values (best scoring move for each dice outcome)
        for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
          term_values[d] = GetBestScoringMove(d, mask, upper, y_state).first;
        }

        // Stage 2: Roll2 optimization (1 roll left)
        lattice_.Reroll(term_values.data(), roll2_values.data(), subset_exp.data());

        // Stage 3: Roll1 optimization (2 rolls left)
        lattice_.Reroll(roll2_values.data(), roll1_values.data(), subset_exp.data());

        // Stage 4: Start of turn = expectation over the initial roll (keep nothing)
        float start_ev = static_cast<float>(
            lattice_.InitialRoll(roll1_values.data(), subset_exp.data()));
        
        SetBoardEV(mask, upper, y_state, start_ev);
      }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
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
 
 #ifndef BUILD_GUIDE
 int main() {
   std::cout << "=== Yahtzee Optimal Strategy Solver (Optimized) ===" << std::endl;
   std::cout << "Version: 4 (start-of-turn EVs only, float precision)\n" << std::endl;
   
   auto start_time = std::chrono::high_resolution_clock::now();
   
   YahtzeeSolver solver;
   solver.Solve();
   solver.SaveToFile("yahtzee_optimal_v4.bin");
   
   auto end_time = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::seconds>(
       end_time - start_time).count();
   
   std::cout << "\nTotal time: " << duration << " seconds" << std::endl;
   
   return 0;
 }
 #endif  // BUILD_GUIDE