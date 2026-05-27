/**
 * Yahtzee PMF Solver
 * 
 * Computes the probability mass function (PMF) of future scores for each
 * board state, assuming optimal play. This allows computing win probabilities
 * against an opponent.
 * 
 * For each state (mask, upper, y_state), we compute:
 *   P(future_score = s) for all possible s
 * 
 * This matches the EV solver's board_ev semantics - it's the distribution
 * of points scored FROM THIS POINT FORWARD, not cumulative.
 * 
 * DATA STRUCTURE:
 * ==============
 * To handle the long tail from Yahtzee bonuses efficiently:
 * - Dense array for scores 0-400 (covers 99.9%+ of probability mass)
 * - Sparse map for scores 401+ (rare multi-Yahtzee scenarios)
 * 
 * MEMORY:
 * =======
 * - Dense: 786,432 states × 401 scores × 4 bytes = 1.23 GB
 * - Sparse: negligible (few states have significant mass above 400)
 */

 #include <iostream>
 #include <vector>
 #include <map>
 #include <algorithm>
 #include <numeric>
 #include <fstream>
 #include <cmath>
 #include <chrono>
 #include <cassert>
 #include <iomanip>
 #include <omp.h>
 
 // -------------------------------------------------------------------------
 // CONSTANTS
 // -------------------------------------------------------------------------
 
 const int NUM_CATS = 13;
 const int CAT_YAHTZEE = 12;
 const int NUM_BOARD_MASKS = (1 << 12);  // 4096
 const int MAX_UPPER_SCORE = 64;
 const int NUM_Y_STATES = 3;
 const int NUM_DICE_COMBOS = 252;
 const int BOARD_STATE_SIZE = NUM_BOARD_MASKS * MAX_UPPER_SCORE * NUM_Y_STATES;
 
 // PMF constants
 const int DENSE_MAX = 1575; // Dense storage for scores 0-1575 (exact Yahtzee max)
 const int DENSE_SIZE = DENSE_MAX + 1;  // 1576 entries
 const float PROB_THRESHOLD = 1e-10f;  // Ignore probabilities below this
 
 // -------------------------------------------------------------------------
 // PMF DATA STRUCTURE
 // -------------------------------------------------------------------------
 
 /**
  * Probability Mass Function with dense + sparse storage
  */
 struct PMF {
     std::vector<float> dense;  // P(score = 0..DENSE_MAX)
     std::map<int, float> sparse;  // P(score > DENSE_MAX)
     
     PMF() : dense(DENSE_SIZE, 0.0f) {}
     
     void Clear() {
         std::fill(dense.begin(), dense.end(), 0.0f);
         sparse.clear();
     }
     
     void SetProb(int score, float prob) {
         if (prob < PROB_THRESHOLD) return;
         if (score <= DENSE_MAX) {
             dense[score] = prob;
         } else {
             sparse[score] = prob;
         }
     }
     
     void AddProb(int score, float prob) {
         if (prob < PROB_THRESHOLD) return;
         if (score <= DENSE_MAX) {
             dense[score] += prob;
         } else {
             sparse[score] += prob;
         }
     }
     
     float GetProb(int score) const {
         if (score <= DENSE_MAX) {
             return dense[score];
         }
         auto it = sparse.find(score);
         return (it != sparse.end()) ? it->second : 0.0f;
     }
     
     // Compute expected value
     float ExpectedValue() const {
         float ev = 0.0f;
         for (int s = 0; s <= DENSE_MAX; s++) {
             ev += s * dense[s];
         }
         for (const auto& [score, prob] : sparse) {
             ev += score * prob;
         }
         return ev;
     }
     
     // Total probability (should sum to 1.0)
     float TotalProb() const {
         float total = 0.0f;
         for (float p : dense) total += p;
         for (const auto& [score, prob] : sparse) total += prob;
         return total;
     }
     
     // Shift PMF by adding points (for convolution)
     // Returns a new PMF where P'(s + points) = P(s)
     PMF Shift(int points) const {
         PMF result;
         for (int s = 0; s <= DENSE_MAX; s++) {
             if (dense[s] > PROB_THRESHOLD) {
                 result.AddProb(s + points, dense[s]);
             }
         }
         for (const auto& [score, prob] : sparse) {
             result.AddProb(score + points, prob);
         }
         return result;
     }
     
     // Add another PMF scaled by probability (for mixing distributions)
     void AddScaled(const PMF& other, float scale) {
         if (scale < PROB_THRESHOLD) return;
         for (int s = 0; s <= DENSE_MAX; s++) {
             if (other.dense[s] > PROB_THRESHOLD) {
                 AddProb(s, other.dense[s] * scale);
             }
         }
         for (const auto& [score, prob] : other.sparse) {
             AddProb(score, prob * scale);
         }
     }
     
     // Debug print
     void Print(int max_show = 20) const {
         std::cout << "PMF (EV=" << std::fixed << std::setprecision(2) 
                   << ExpectedValue() << ", total=" << TotalProb() << "):" << std::endl;
         int shown = 0;
         for (int s = 0; s <= DENSE_MAX && shown < max_show; s++) {
             if (dense[s] > 0.001f) {
                 std::cout << "  P(" << s << ") = " << dense[s] << std::endl;
                 shown++;
             }
         }
         for (const auto& [score, prob] : sparse) {
             if (shown >= max_show) break;
             if (prob > 0.001f) {
                 std::cout << "  P(" << score << ") = " << prob << std::endl;
                 shown++;
             }
         }
     }
 };
 
 // -------------------------------------------------------------------------
 // DICE INDEXER
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
     
     const std::vector<int>& GetDice(int id) const {
         return id_to_dice[id];
     }
     
 private:
     void Generate(std::vector<int>& current, int start, int depth) {
         if ((int)current.size() == depth) {
             dice_to_id[current] = id_to_dice.size();
             id_to_dice.push_back(current);
             return;
         }
         for (int i = start; i <= 6; i++) {
             current.push_back(i);
             Generate(current, i, depth);
             current.pop_back();
         }
     }
 };
 
 // -------------------------------------------------------------------------
 // SCORING FUNCTIONS
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
     for (int i = 1; i <= 6; i++) {
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
 // PMF SOLVER CLASS
 // -------------------------------------------------------------------------
 
 class YahtzeePMFSolver {
 public:
     YahtzeePMFSolver(const std::string& ev_file) : indexer_5_(5) {
         LoadEVStrategy(ev_file);
         InitDiceSubsets();
         BuildTransitionMatrix();
         AllocatePMFTables();
     }
     
     void Solve();
     void SaveToFile(const std::string& filename);
     
     // Query interface
     const PMF& GetPMF(int mask, int upper, int y_state) const {
         return board_pmf_[BoardIndex(mask, upper, y_state)];
     }
 
 private:
     DiceIndexer indexer_5_;
     
     // Dice subsets and transitions (same as EV solver)
     std::vector<std::vector<int>> all_subsets_dice_;
     std::map<std::vector<int>, int> subset_to_id_;
     std::vector<std::vector<float>> trans_probs_;
     std::vector<std::vector<int>> valid_subsets_for_outcome_;
     
     // Loaded EV strategy (for determining optimal actions)
     std::vector<float> board_ev_;
     std::vector<float> roll1_ev_;
     std::vector<float> roll2_ev_;
     
     // PMF tables
     std::vector<PMF> board_pmf_;  // PMF at start of turn
     
     // Indexing
     inline size_t BoardIndex(int mask, int upper, int y_state) const {
         return (mask * MAX_UPPER_SCORE + upper) * NUM_Y_STATES + y_state;
     }
     
     inline size_t RollIndex(int mask, int upper, int y_state, int dice_id) const {
         return ((mask * MAX_UPPER_SCORE + upper) * NUM_Y_STATES + y_state) * 
                NUM_DICE_COMBOS + dice_id;
     }
     
     // EV accessors
     float GetBoardEV(int mask, int upper, int y_state) const {
         if (mask == (1 << 12) - 1 && y_state > 0) return 0.0f;
         return board_ev_[BoardIndex(mask, upper, y_state)];
     }
     
     float GetRoll1EV(int mask, int upper, int y_state, int dice_id) const {
         return roll1_ev_[RollIndex(mask, upper, y_state, dice_id)];
     }
     
     float GetRoll2EV(int mask, int upper, int y_state, int dice_id) const {
         return roll2_ev_[RollIndex(mask, upper, y_state, dice_id)];
     }
     
     void LoadEVStrategy(const std::string& filename);
     void InitDiceSubsets();
     void BuildTransitionMatrix();
     void AllocatePMFTables();
     
     // Get optimal action and resulting state transition
     struct ScoringResult {
         int category;
         int points;      // Immediate points scored
         int next_mask;
         int next_upper;
         int next_y_state;
     };
     
     ScoringResult GetOptimalScoringMove(int dice_id, int mask, int upper, int y_state) const;
     
     // Get optimal keep choice (returns subset_id)
     int GetOptimalKeep(int dice_id, int mask, int upper, int y_state, int rolls_left) const;
 };
 
 void YahtzeePMFSolver::LoadEVStrategy(const std::string& filename) {
     std::cout << "Loading EV strategy from " << filename << "..." << std::flush;
     
     std::ifstream in(filename, std::ios::binary);
     if (!in) {
         throw std::runtime_error("Could not open EV strategy file");
     }
     
     uint32_t magic, version, board_size, roll_size;
     in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
     in.read(reinterpret_cast<char*>(&version), sizeof(version));
     in.read(reinterpret_cast<char*>(&board_size), sizeof(board_size));
     in.read(reinterpret_cast<char*>(&roll_size), sizeof(roll_size));
     
     if (magic != 0x59414854 || version != 3) {
         throw std::runtime_error("Invalid EV strategy file format");
     }
     
     board_ev_.resize(board_size);
     roll1_ev_.resize(roll_size);
     roll2_ev_.resize(roll_size);
     
     in.read(reinterpret_cast<char*>(board_ev_.data()), board_size * sizeof(float));
     in.read(reinterpret_cast<char*>(roll1_ev_.data()), roll_size * sizeof(float));
     in.read(reinterpret_cast<char*>(roll2_ev_.data()), roll_size * sizeof(float));
     
     std::cout << " done" << std::endl;
 }
 
 void YahtzeePMFSolver::InitDiceSubsets() {
     for (int i = 0; i <= 5; i++) {
         DiceIndexer tmp(i);
         for (const auto& d : tmp.id_to_dice) {
             subset_to_id_[d] = all_subsets_dice_.size();
             all_subsets_dice_.push_back(d);
         }
     }
 }
 
 void YahtzeePMFSolver::BuildTransitionMatrix() {
     std::cout << "Building transition matrix..." << std::flush;
     
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
             int final_id = indexer_5_.GetId(current);
             trans_probs_[i][final_id] += 1.0f;
         }
         
         for (int d = 0; d < NUM_DICE_COMBOS; d++) {
             trans_probs_[i][d] /= static_cast<float>(total);
         }
     }
     
     // Build valid subsets for each outcome
     valid_subsets_for_outcome_.resize(NUM_DICE_COMBOS);
     for (int d = 0; d < NUM_DICE_COMBOS; d++) {
         std::vector<int> dice = indexer_5_.GetDice(d);
         for (int m = 0; m < 32; m++) {
             std::vector<int> sub;
             for (int b = 0; b < 5; b++) {
                 if ((m >> b) & 1) sub.push_back(dice[b]);
             }
             std::sort(sub.begin(), sub.end());
             valid_subsets_for_outcome_[d].push_back(subset_to_id_[sub]);
         }
         std::sort(valid_subsets_for_outcome_[d].begin(), 
                   valid_subsets_for_outcome_[d].end());
         auto last = std::unique(valid_subsets_for_outcome_[d].begin(), 
                                valid_subsets_for_outcome_[d].end());
         valid_subsets_for_outcome_[d].erase(last, valid_subsets_for_outcome_[d].end());
     }
     
     std::cout << " done" << std::endl;
 }
 
 void YahtzeePMFSolver::AllocatePMFTables() {
     std::cout << "Allocating PMF tables..." << std::flush;
     board_pmf_.resize(BOARD_STATE_SIZE);
     size_t mem_mb = BOARD_STATE_SIZE * DENSE_SIZE * sizeof(float) / (1024 * 1024);
     std::cout << " done (" << mem_mb << " MB for dense part)" << std::endl;
 }
 
 YahtzeePMFSolver::ScoringResult YahtzeePMFSolver::GetOptimalScoringMove(
     int dice_id, int mask, int upper, int y_state) const {
     
     std::vector<int> dice = indexer_5_.GetDice(dice_id);
     bool is_yahtzee = HasNOfAKind(dice, 5);
     int yahtzee_num = is_yahtzee ? dice[0] : 0;
     
     ScoringResult best;
     best.category = -1;
     float max_val = -1e9f;
     
     // Special case: Yahtzee roll
     if (is_yahtzee) {
         if (y_state == 0) {
             // Take Yahtzee
             float val = 50.0f + GetBoardEV(mask, upper, 2);
             if (val > max_val) {
                 max_val = val;
                 best.category = CAT_YAHTZEE;
                 best.points = 50;
                 best.next_mask = mask;
                 best.next_upper = upper;
                 best.next_y_state = 2;
             }
         } else {
             // Joker rules
             float bonus = (y_state == 2) ? 100.0f : 0.0f;
             int upper_idx = yahtzee_num - 1;
             bool upper_open = !((mask >> upper_idx) & 1);
             
             if (upper_open) {
                 // Must play corresponding upper
                 int score = CalculateScore(upper_idx, dice);
                 int next_u = std::min(63, upper + score);
                 float u_bonus = (upper < 63 && next_u >= 63) ? 35.0f : 0.0f;
                 
                 best.category = upper_idx;
                 best.points = (int)(bonus + score + u_bonus);
                 best.next_mask = mask | (1 << upper_idx);
                 best.next_upper = next_u;
                 best.next_y_state = y_state;
                 return best;
             } else {
                 // Try lower section
                 for (int c = 6; c < 12; c++) {
                     if (!((mask >> c) & 1)) {
                         int score = CalculateScore(c, dice);
                         if (c == 8) score = 25;
                         else if (c == 9) score = 30;
                         else if (c == 10) score = 40;
                         
                         float val = bonus + score + GetBoardEV(mask | (1 << c), upper, y_state);
                         if (val > max_val) {
                             max_val = val;
                             best.category = c;
                             best.points = (int)(bonus + score);
                             best.next_mask = mask | (1 << c);
                             best.next_upper = upper;
                             best.next_y_state = y_state;
                         }
                     }
                 }
                 
                 // If no lower, use upper for 0
                 if (best.category < 0) {
                     for (int c = 0; c < 6; c++) {
                         if (!((mask >> c) & 1)) {
                             float val = bonus + GetBoardEV(mask | (1 << c), upper, y_state);
                             if (val > max_val) {
                                 max_val = val;
                                 best.category = c;
                                 best.points = (int)bonus;
                                 best.next_mask = mask | (1 << c);
                                 best.next_upper = upper;
                                 best.next_y_state = y_state;
                             }
                         }
                     }
                 }
             }
             return best;
         }
     }
     
     // Standard case: try all open categories
     for (int c = 0; c < 12; c++) {
         if (!((mask >> c) & 1)) {
             int score = CalculateScore(c, dice);
             int next_u = upper;
             float u_bonus = 0.0f;
             
             if (c < 6) {
                 next_u = std::min(63, upper + score);
                 if (upper < 63 && next_u >= 63) u_bonus = 35.0f;
             }
             
             float val = score + u_bonus + GetBoardEV(mask | (1 << c), next_u, y_state);
             if (val > max_val) {
                 max_val = val;
                 best.category = c;
                 best.points = score + (int)u_bonus;
                 best.next_mask = mask | (1 << c);
                 best.next_upper = next_u;
                 best.next_y_state = y_state;
             }
         }
     }
     
     // Yahtzee category with non-Yahtzee roll
     if (y_state == 0) {
         float val = GetBoardEV(mask, upper, 1);
         if (val > max_val) {
             max_val = val;
             best.category = CAT_YAHTZEE;
             best.points = 0;
             best.next_mask = mask;
             best.next_upper = upper;
             best.next_y_state = 1;
         }
     }
     
     return best;
 }
 
 int YahtzeePMFSolver::GetOptimalKeep(int dice_id, int mask, int upper, 
                                       int y_state, int rolls_left) const {
     std::vector<int> dice = indexer_5_.GetDice(dice_id);
     
     float best_ev = -1e9f;
     int best_subset = 0;
     
     for (int sid : valid_subsets_for_outcome_[dice_id]) {
         float ev = 0.0f;
         const auto& probs = trans_probs_[sid];
         
         for (int outcome = 0; outcome < NUM_DICE_COMBOS; outcome++) {
             if (probs[outcome] > 0) {
                 float outcome_ev;
                 if (rolls_left == 2) {
                     outcome_ev = GetRoll2EV(mask, upper, y_state, outcome);
                 } else {
                     // Terminal - compute best scoring move
                     auto result = GetOptimalScoringMove(outcome, mask, upper, y_state);
                     outcome_ev = result.points + GetBoardEV(result.next_mask, 
                                                             result.next_upper, 
                                                             result.next_y_state);
                 }
                 ev += probs[outcome] * outcome_ev;
             }
         }
         
         if (ev > best_ev) {
             best_ev = ev;
             best_subset = sid;
         }
     }
     
     return best_subset;
 }
 
 void YahtzeePMFSolver::Solve() {
     std::cout << "\nStarting PMF computation (reverse induction)...\n" << std::endl;

     for (int total_filled = 13; total_filled >= 0; --total_filled) {
         auto start_time = std::chrono::high_resolution_clock::now();
         std::cout << "Processing states with " << total_filled
                   << " categories filled..." << std::flush;

         // Flatten (mask, y_state, upper) for this fill level into one list
         // so OpenMP can parallelize all of them in one loop.
         std::vector<std::tuple<int,int,int>> jobs;
         jobs.reserve(4096 * 64 * 3);
         for (int m = 0; m < (1 << 12); m++) {
             int mc = __builtin_popcount(m);
             for (int u = 0; u < MAX_UPPER_SCORE; u++) {
                 if (mc == total_filled)
                     jobs.emplace_back(m, u, 0);
                 if (mc + 1 == total_filled) {
                     jobs.emplace_back(m, u, 1);
                     jobs.emplace_back(m, u, 2);
                 }
             }
         }

         struct FinalResult { int points; size_t next_idx; };

         // Three-pass PMF accumulation with thread-private buffers (booty-dice pattern).
         // Each pass is a 252×nonzero×401 matrix product — ~125× faster than 252³ loop.
         #pragma omp parallel
         {
             // Per-thread heap buffers: [outcome][score_entry] = 252×401 floats = 404 KB each.
             std::vector<float> shifted_buf(NUM_DICE_COMBOS * DENSE_SIZE, 0.0f);
             std::vector<float> pmf2_buf(NUM_DICE_COMBOS * DENSE_SIZE, 0.0f);
             std::vector<float> pmf1_buf(NUM_DICE_COMBOS * DENSE_SIZE, 0.0f);

             #pragma omp for schedule(dynamic, 4)
             for (int ji = 0; ji < (int)jobs.size(); ji++) {
                 auto [mask, upper, y_state] = jobs[ji];
                 PMF& pmf = board_pmf_[BoardIndex(mask, upper, y_state)];
                 pmf.Clear();

                 if (mask == (1 << 12) - 1 && y_state > 0) {
                     pmf.SetProb(0, 1.0f);
                     continue;
                 }

                 // --- Precompute keep decisions via dot products (keep2 fix) ---
                 int keep1_for[NUM_DICE_COMBOS], keep2_for[NUM_DICE_COMBOS];
                 FinalResult final_res[NUM_DICE_COMBOS];

                 for (int d = 0; d < NUM_DICE_COMBOS; d++) {
                     auto r = GetOptimalScoringMove(d, mask, upper, y_state);
                     final_res[d] = {r.points,
                                     BoardIndex(r.next_mask, r.next_upper, r.next_y_state)};
                 }

                 float term_ev[NUM_DICE_COMBOS];
                 for (int d = 0; d < NUM_DICE_COMBOS; d++)
                     term_ev[d] = final_res[d].points + board_ev_[final_res[d].next_idx];

                 const int NUM_SUBSETS = 462;
                 float subset_ev[NUM_SUBSETS];
                 float subset_roll2_ev[NUM_SUBSETS];
                 size_t board_base = BoardIndex(mask, upper, y_state) * NUM_DICE_COMBOS;
                 for (int sid = 0; sid < NUM_SUBSETS; sid++) {
                     const auto& probs = trans_probs_[sid];
                     float ev2 = 0.0f, ev1 = 0.0f;
                     for (int d = 0; d < NUM_DICE_COMBOS; d++) {
                         float p = probs[d];
                         ev2 += p * term_ev[d];
                         ev1 += p * roll2_ev_[board_base + d];
                     }
                     subset_ev[sid] = ev2;
                     subset_roll2_ev[sid] = ev1;
                 }

                 for (int d = 0; d < NUM_DICE_COMBOS; d++) {
                     float best1 = -1e9f, best2 = -1e9f;
                     int sid1 = 0, sid2 = 0;
                     for (int sid : valid_subsets_for_outcome_[d]) {
                         if (subset_roll2_ev[sid] > best1) { best1 = subset_roll2_ev[sid]; sid1 = sid; }
                         if (subset_ev[sid] > best2) { best2 = subset_ev[sid]; sid2 = sid; }
                     }
                     keep1_for[d] = sid1;
                     keep2_for[d] = sid2;
                 }

                 // --- Pass 0: shifted[r3][s] = board_pmf_[next[r3]].dense[s - pts[r3]] ---
                 std::fill(shifted_buf.begin(), shifted_buf.end(), 0.0f);
                 for (int r3 = 0; r3 < NUM_DICE_COMBOS; r3++) {
                     const PMF& fut = board_pmf_[final_res[r3].next_idx];
                     int pts = final_res[r3].points;
                     float* row = shifted_buf.data() + r3 * DENSE_SIZE;
                     for (int s = 0; s <= DENSE_MAX; s++) {
                         int ns = s + pts;
                         if (ns <= DENSE_MAX) row[ns] = fut.dense[s];
                     }
                 }

                 // --- Pass 1: pmf2[r2][s] = Σ_r3 trans[keep2_for[r2]][r3] × shifted[r3][s] ---
                 std::fill(pmf2_buf.begin(), pmf2_buf.end(), 0.0f);
                 for (int r2 = 0; r2 < NUM_DICE_COMBOS; r2++) {
                     const float* trans2 = trans_probs_[keep2_for[r2]].data();
                     float* out = pmf2_buf.data() + r2 * DENSE_SIZE;
                     for (int r3 = 0; r3 < NUM_DICE_COMBOS; r3++) {
                         float p = trans2[r3];
                         if (p < PROB_THRESHOLD) continue;
                         const float* in = shifted_buf.data() + r3 * DENSE_SIZE;
                         for (int s = 0; s <= DENSE_MAX; s++) out[s] += p * in[s];
                     }
                 }

                 // --- Pass 2: pmf1[r1][s] = Σ_r2 trans[keep1_for[r1]][r2] × pmf2[r2][s] ---
                 std::fill(pmf1_buf.begin(), pmf1_buf.end(), 0.0f);
                 for (int r1 = 0; r1 < NUM_DICE_COMBOS; r1++) {
                     const float* trans1 = trans_probs_[keep1_for[r1]].data();
                     float* out = pmf1_buf.data() + r1 * DENSE_SIZE;
                     for (int r2 = 0; r2 < NUM_DICE_COMBOS; r2++) {
                         float p = trans1[r2];
                         if (p < PROB_THRESHOLD) continue;
                         const float* in = pmf2_buf.data() + r2 * DENSE_SIZE;
                         for (int s = 0; s <= DENSE_MAX; s++) out[s] += p * in[s];
                     }
                 }

                 // --- Pass 3: result[s] = Σ_r1 init[r1] × pmf1[r1][s] ---
                 const float* init_p = trans_probs_[0].data();
                 for (int r1 = 0; r1 < NUM_DICE_COMBOS; r1++) {
                     float p = init_p[r1];
                     if (p < PROB_THRESHOLD) continue;
                     const float* in = pmf1_buf.data() + r1 * DENSE_SIZE;
                     for (int s = 0; s <= DENSE_MAX; s++) pmf.dense[s] += p * in[s];
                 }

                 // Sparse correction: scores > DENSE_MAX from Yahtzee bonus chains.
                 // Uses triple loop but only executes for the rare states that have them.
                 bool has_sparse = false;
                 for (int r3 = 0; r3 < NUM_DICE_COMBOS && !has_sparse; r3++)
                     has_sparse = !board_pmf_[final_res[r3].next_idx].sparse.empty();
                 if (has_sparse) {
                     for (int r1 = 0; r1 < NUM_DICE_COMBOS; r1++) {
                         float p1 = init_p[r1];
                         if (p1 < PROB_THRESHOLD) continue;
                         const float* tr1 = trans_probs_[keep1_for[r1]].data();
                         for (int r2 = 0; r2 < NUM_DICE_COMBOS; r2++) {
                             float p12 = p1 * tr1[r2];
                             if (p12 < PROB_THRESHOLD) continue;
                             const float* tr2 = trans_probs_[keep2_for[r2]].data();
                             for (int r3 = 0; r3 < NUM_DICE_COMBOS; r3++) {
                                 float tp = p12 * tr2[r3];
                                 if (tp < PROB_THRESHOLD) continue;
                                 const PMF& fut = board_pmf_[final_res[r3].next_idx];
                                 if (fut.sparse.empty()) continue;
                                 int pts = final_res[r3].points;
                                 for (const auto& [sc, prob] : fut.sparse)
                                     pmf.sparse[sc + pts] += prob * tp;
                             }
                         }
                     }
                 }
             }
         }

         auto end_time = std::chrono::high_resolution_clock::now();
         auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
             end_time - start_time).count();
         std::cout << " (" << ms << " ms, " << jobs.size() << " states)" << std::endl;
     }

     std::cout << "\nPMF computation complete!" << std::endl;

     const PMF& start_pmf = board_pmf_[BoardIndex(0, 0, 0)];
     std::cout << "\nVerification:" << std::endl;
     std::cout << "  PMF expected value at start: " << start_pmf.ExpectedValue() << std::endl;
     std::cout << "  EV strategy value at start:  " << GetBoardEV(0, 0, 0) << std::endl;
     std::cout << "  PMF total probability: " << start_pmf.TotalProb() << std::endl;
 }
 
 void YahtzeePMFSolver::SaveToFile(const std::string& filename) {
     std::cout << "\nSaving PMF to " << filename << "..." << std::endl;
     
     std::ofstream out(filename, std::ios::binary);
     if (!out) {
         std::cerr << "Error: Could not open file for writing" << std::endl;
         return;
     }
     
     // Header
     uint32_t magic = 0x504D4659;  // "PMFY"
     uint32_t version = 1;
     uint32_t num_states = BOARD_STATE_SIZE;
     uint32_t dense_size = DENSE_SIZE;
     
     out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
     out.write(reinterpret_cast<const char*>(&version), sizeof(version));
     out.write(reinterpret_cast<const char*>(&num_states), sizeof(num_states));
     out.write(reinterpret_cast<const char*>(&dense_size), sizeof(dense_size));
     
     // Write dense parts
     for (const auto& pmf : board_pmf_) {
         out.write(reinterpret_cast<const char*>(pmf.dense.data()), 
                   DENSE_SIZE * sizeof(float));
     }
     
     // Write sparse parts (count + pairs)
     for (const auto& pmf : board_pmf_) {
         uint32_t sparse_count = pmf.sparse.size();
         out.write(reinterpret_cast<const char*>(&sparse_count), sizeof(sparse_count));
         for (const auto& [score, prob] : pmf.sparse) {
             int32_t s = score;
             out.write(reinterpret_cast<const char*>(&s), sizeof(s));
             out.write(reinterpret_cast<const char*>(&prob), sizeof(prob));
         }
     }
     
     out.close();
     std::cout << "PMF saved successfully" << std::endl;
 }
 
 // -------------------------------------------------------------------------
 // MAIN
 // -------------------------------------------------------------------------
 
 /*
  * PERFORMANCE NOTE:
  * =================
  * The current implementation has O(252^3) = 16M iterations per state for the
  * triple-nested roll loop. With 786K states, this is ~10^13 operations total.
  * 
  * Optimizations to consider:
  * 1. Precompute "turn outcome distribution" for each (mask, upper, y_state)
  *    This is a distribution over (points_scored, next_state) pairs.
  *    Can be computed once and reused.
  * 
  * 2. Use sparse representation - most dice outcomes have very low probability
  *    after the keep decisions. Could threshold at 1e-6 or so.
  * 
  * 3. Parallelize with OpenMP (add #pragma omp parallel for)
  * 
  * 4. Memoize the turn PMF computation - many states share similar structure.
  * 
  * Current estimate: ~10-30 minutes without optimization, ~1-5 min with OpenMP.
  */
 
 int main(int argc, char* argv[]) {
     std::cout << "=== Yahtzee PMF Solver ===" << std::endl;
     std::cout << "Computes probability distribution of future scores\n" << std::endl;
     
     std::string ev_file = "yahtzee_optimal_v3.bin";
     if (argc > 1) {
         ev_file = argv[1];
     }
     
     auto start_time = std::chrono::high_resolution_clock::now();
     
     try {
         YahtzeePMFSolver solver(ev_file);
         solver.Solve();
         solver.SaveToFile("yahtzee_pmf_v1.bin");
         
         // Print some example PMFs
         std::cout << "\n=== Example PMFs ===" << std::endl;
         
         std::cout << "\nStart of game (mask=0, upper=0, y_state=0):" << std::endl;
         solver.GetPMF(0, 0, 0).Print(15);
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
     
     auto end_time = std::chrono::high_resolution_clock::now();
     auto duration = std::chrono::duration_cast<std::chrono::seconds>(
         end_time - start_time).count();
     
     std::cout << "\nTotal time: " << duration << " seconds" << std::endl;
     
     return 0;
 }