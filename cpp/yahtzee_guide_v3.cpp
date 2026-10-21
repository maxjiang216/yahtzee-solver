/**
 * Yahtzee Interactive Guide v3
 * 
 * An interactive program that uses the optimal Yahtzee strategy to guide gameplay.
 * Like "Stockfish for Yahtzee" - provides optimal recommendations at each step.
 * 
 * COMPILATION:
 *   g++ -O3 -std=c++17 -o yahtzee_guide yahtzee_guide_v3.cpp yahtzee_strategy.cpp
 * 
 * USAGE:
 *   ./yahtzee_guide [strategy_file.bin]
 *   Default: yahtzee_optimal_v4.bin
 * 
 * INPUT FORMAT:
 *   - Dice: Enter as digits like "12345" or space-separated "1 2 3 4 5"
 *   - Default action: Just press Enter to accept the recommended move
 *   - Alternative: Type the action (e.g., "keep 12" or "score 3")
 *   - Undo: Type "undo" to go back one step
 *   - Quit: Type "quit" or "q"
 */

 #include <iostream>
 #include <vector>
 #include <string>
 #include <sstream>
 #include <algorithm>
 #include <bitset>
 #include <cstdlib>
 #include <numeric>
 #include <iomanip>
 
 #include "yahtzee_strategy.h"
 
 // Category names for display
 const char* CATEGORY_NAMES[] = {
     "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
     "Three of a Kind", "Four of a Kind", "Full House",
     "Small Straight", "Large Straight", "Chance", "Yahtzee"
 };
 
 // Simple game state tracker
 struct GameState {
     int mask = 0;           // Bitmask of filled categories (12 bits, excluding Yahtzee)
     int upper = 0;          // Upper section score
     int y_state = 0;        // Yahtzee state (0=open, 1=filled 0, 2=filled 50)
     int turn = 1;           // Current turn (1-13)
     int rolls_left = 2;     // Rolls remaining this turn
     std::vector<int> current_dice;  // Current dice values
     std::vector<int> frozen_dice;   // Dice kept from previous roll
     int total_score = 0;    // Total score so far
     
     // History for undo
     struct HistoryEntry {
         int mask, upper, y_state, turn, rolls_left, total_score;
         std::vector<int> dice;
         std::vector<int> frozen;
     };
     std::vector<HistoryEntry> history;
     
     void SaveState() {
         history.push_back({mask, upper, y_state, turn, rolls_left, 
                           total_score, current_dice, frozen_dice});
     }
     
     bool Undo() {
         if (history.empty()) return false;
         HistoryEntry last = history.back();
         history.pop_back();
         mask = last.mask;
         upper = last.upper;
         y_state = last.y_state;
         turn = last.turn;
         rolls_left = last.rolls_left;
         total_score = last.total_score;
         current_dice = last.dice;
         frozen_dice = last.frozen;
         return true;
     }
     
     bool IsCategoryFilled(int cat) const {
         if (cat == 12) return y_state > 0;  // Yahtzee
         return (mask >> cat) & 1;
     }
     
     int GetTotalScore() const {
         int score = total_score;
         if (upper >= 63) score += 35;
         return score;
     }
 };
 
 // Parse dice input - accepts "12345" or "1 2 3 4 5" or "1,2,3,4,5"
 std::vector<int> ParseDice(const std::string& input) {
     std::vector<int> dice;
     
     // Try parsing as single string of digits (e.g., "12345")
     if (input.find_first_not_of("123456 ") == std::string::npos && input.length() <= 10) {
         for (char c : input) {
             if (c >= '1' && c <= '6') {
                 dice.push_back(c - '0');
             }
         }
         if (dice.size() > 0) return dice;
     }
     
     // Parse as space or comma separated
     std::stringstream ss(input);
     std::string token;
     while (ss >> token) {
         token.erase(std::remove(token.begin(), token.end(), ','), token.end());
         try {
             int val = std::stoi(token);
             if (val >= 1 && val <= 6) {
                 dice.push_back(val);
             }
         } catch (...) {
             // Invalid input, ignore
         }
     }
     return dice;
 }
 
 // Display current game state
 void DisplayState(const GameState& state) {
     std::cout << "\n" << std::string(60, '=') << std::endl;
     std::cout << "Turn " << state.turn << " / 13";
     std::cout << "  |  Upper: " << state.upper;
     if (state.upper >= 63) std::cout << " (+35 bonus)";
     std::cout << "  |  Total: " << state.GetTotalScore() << std::endl;
     std::cout << "Rolls Left: " << state.rolls_left << std::endl;
     
     if (!state.current_dice.empty()) {
         std::cout << "Current Dice: ";
         for (size_t i = 0; i < state.current_dice.size(); i++) {
             std::cout << "[" << i << "]=" << state.current_dice[i] << " ";
         }
         std::cout << std::endl;
     }
     
     if (!state.frozen_dice.empty()) {
         std::cout << "Frozen Dice: ";
         for (int d : state.frozen_dice) std::cout << d << " ";
         std::cout << std::endl;
     }
     
     // Show filled categories
     std::cout << "\nFilled: ";
     bool any = false;
     for (int i = 0; i < 13; i++) {
         if (state.IsCategoryFilled(i)) {
             if (any) std::cout << ", ";
             std::cout << CATEGORY_NAMES[i];
             any = true;
         }
     }
     if (!any) std::cout << "None";
     std::cout << std::endl;
     std::cout << std::string(60, '=') << std::endl;
 }
 
// Convert keep mask to human-readable format
std::string KeepMaskToString(int mask, const std::vector<int>& dice) {
    std::string result;
    for (int i = 0; i < 5; i++) {
        if ((mask >> i) & 1) {
            if (!result.empty()) result += " ";
            result += std::to_string(dice[i]);
        }
    }
    return result.empty() ? "none" : result;
}

// Convert guide's 12-bit mask + y_state to format expected by strategy
// The strategy actually uses a 12-bit mask (excluding Yahtzee) internally
// and tracks Yahtzee separately via y_state, so we just pass the mask as-is
// (The strategy functions accept a 13-bit mask but strip bit 12 internally)
int BuildFullMask(int mask, int y_state) {
    // The strategy's internal storage uses 12-bit masks (categories 0-11)
    // Yahtzee (category 12) is tracked separately via y_state
    // So we can pass the mask directly - bit 12 will be ignored in indexing
    // But we set it for the strategy's logic checks
    if (y_state > 0) {
        return mask | (1 << 12);  // Set Yahtzee bit for strategy logic
    }
    return mask;
}
 
 // Find dice indices to keep based on values
 std::vector<int> FindKeepIndices(const std::vector<int>& dice, 
                                  const std::vector<int>& keep_values) {
     std::vector<int> indices;
     std::vector<bool> used(dice.size(), false);
     
     for (int val : keep_values) {
         for (size_t i = 0; i < dice.size(); i++) {
             if (dice[i] == val && !used[i]) {
                 indices.push_back(i);
                 used[i] = true;
                 break;
             }
         }
     }
     return indices;
 }
 
 void ShowHelp() {
     std::cout << "\nCommands:\n";
     std::cout << "  Enter dice: '12345' or '1 2 3 4 5'\n";
     std::cout << "  Accept recommendation: Press Enter\n";
     std::cout << "  Alternative keep: 'keep 12' (keep dice with values 1 and 2)\n";
     std::cout << "  Alternative score: 'score 3' (score in category 3)\n";
     std::cout << "  Undo: 'undo'\n";
     std::cout << "  Quit: 'quit' or 'q'\n";
 }
 
 int main(int argc, char* argv[]) {
     std::string strategy_file = "yahtzee_optimal_v4.bin";
     if (argc > 1) {
         strategy_file = argv[1];
     }
     
     std::cout << "=== Yahtzee Interactive Guide v3 ===" << std::endl;
     std::cout << "Loading strategy from: " << strategy_file << std::endl;
     
     YahtzeeStrategy* strategy = nullptr;
     try {
         strategy = new YahtzeeStrategy(strategy_file);
     } catch (...) {
         std::cerr << "Error: Could not load strategy file: " << strategy_file << std::endl;
         std::cerr << "Make sure you've run 'make solve' to generate the strategy file." << std::endl;
         return 1;
     }
     
     std::cout << "\nWelcome! Enter dice rolls to get optimal recommendations." << std::endl;
     std::cout << "Type 'help' for commands, 'quit' to exit.\n" << std::endl;
     
     GameState state;
     std::string input;
     
     while (state.turn <= 13) {
         DisplayState(state);
         
         if (state.rolls_left == 2) {
             // Start of turn - need initial roll
             if (state.current_dice.empty()) {
                 std::cout << "\n>>> Enter your dice roll (e.g., '12345'): ";
                 std::getline(std::cin, input);
                 
                 if (input == "quit" || input == "q") break;
                 if (input == "help") {
                     ShowHelp();
                     continue;
                 }
                 
                 std::vector<int> dice = ParseDice(input);
                 if (dice.size() != 5) {
                     std::cout << "Error: Need exactly 5 dice. You entered " 
                               << dice.size() << " values." << std::endl;
                     continue;
                 }
                 
                 std::sort(dice.begin(), dice.end());
                 state.current_dice = dice;
                 state.frozen_dice.clear();
             }
             
            // Recommend what to keep with 2 rolls left
            int full_mask = BuildFullMask(state.mask, state.y_state);
            auto top_keep = strategy->GetTopKeepChoices(state.current_dice, full_mask, 
                                                       state.upper, state.y_state, 2, 3);
            
            if (top_keep.empty()) {
                std::cout << "Error: No valid keep choices." << std::endl;
                break;
            }
            
            int keep_mask = top_keep[0].first;
            std::vector<int> to_keep;
            for (int i = 0; i < 5; i++) {
                if ((keep_mask >> i) & 1) {
                    to_keep.push_back(state.current_dice[i]);
                }
            }
            
            std::cout << "\n>>> Top 3 Keep Choices:" << std::endl;
            for (size_t i = 0; i < top_keep.size(); ++i) {
                int km = top_keep[i].first;
                float ev = top_keep[i].second;
                std::cout << "  " << (i + 1) << ". Keep " << KeepMaskToString(km, state.current_dice)
                          << ": EV = " << std::fixed << std::setprecision(1) << ev << std::endl;
            }
            std::cout << "\n>>> Recommended: Keep " 
                      << KeepMaskToString(keep_mask, state.current_dice);
            std::cout << "\n    Press Enter to accept, or type alternative (e.g., 'keep 12'): ";
             std::getline(std::cin, input);
             
             if (input == "quit" || input == "q") break;
             if (input == "undo") {
                 if (state.Undo()) {
                     std::cout << "Undone!" << std::endl;
                     continue;
                 } else {
                     std::cout << "Nothing to undo." << std::endl;
                     continue;
                 }
             }
             
             std::vector<int> kept;
             if (input.empty()) {
                 kept = to_keep;
             } else if (input.substr(0, 4) == "keep") {
                 std::string values = input.substr(4);
                 std::vector<int> keep_values = ParseDice(values);
                 std::vector<int> indices = FindKeepIndices(state.current_dice, keep_values);
                 for (int idx : indices) {
                     kept.push_back(state.current_dice[idx]);
                 }
             } else {
                 std::cout << "Invalid input. Use 'keep <values>' or press Enter." << std::endl;
                 continue;
             }
             
             state.SaveState();
             state.frozen_dice = kept;
             state.rolls_left = 1;
             
             // Need second roll
             int needed = 5 - kept.size();
             if (needed > 0) {
                 std::cout << "\n>>> Enter remaining dice roll (need " << needed << " dice): ";
                 std::getline(std::cin, input);
                 
                 if (input == "quit" || input == "q") break;
                 if (input == "undo") {
                     state.Undo();
                     continue;
                 }
                 
                 std::vector<int> new_dice = ParseDice(input);
                 if ((int)new_dice.size() != needed) {
                     std::cout << "Error: Need " << needed << " dice. You entered " 
                               << new_dice.size() << "." << std::endl;
                     state.Undo();
                     continue;
                 }
                 
                 state.current_dice = kept;
                 state.current_dice.insert(state.current_dice.end(), 
                                          new_dice.begin(), new_dice.end());
                 std::sort(state.current_dice.begin(), state.current_dice.end());
             } else {
                 // Kept all 5 dice
                 state.current_dice = kept;
             }
             
         } else if (state.rolls_left == 1) {
            // After second roll - recommend what to keep with 1 roll left
            int full_mask = BuildFullMask(state.mask, state.y_state);
            auto top_keep = strategy->GetTopKeepChoices(state.current_dice, full_mask, 
                                                       state.upper, state.y_state, 1, 3);
            
            if (top_keep.empty()) {
                std::cout << "Error: No valid keep choices." << std::endl;
                break;
            }
            
            int keep_mask = top_keep[0].first;
            std::vector<int> to_keep;
            for (int i = 0; i < 5; i++) {
                if ((keep_mask >> i) & 1) {
                    to_keep.push_back(state.current_dice[i]);
                }
            }
            
            std::cout << "\n>>> Top 3 Keep Choices:" << std::endl;
            for (size_t i = 0; i < top_keep.size(); ++i) {
                int km = top_keep[i].first;
                float ev = top_keep[i].second;
                std::cout << "  " << (i + 1) << ". Keep " << KeepMaskToString(km, state.current_dice)
                          << ": EV = " << std::fixed << std::setprecision(1) << ev << std::endl;
            }
            std::cout << "\n>>> Recommended: Keep " 
                      << KeepMaskToString(keep_mask, state.current_dice);
            std::cout << "\n    Press Enter to accept, or type alternative (e.g., 'keep 12'): ";
             std::getline(std::cin, input);
             
             if (input == "quit" || input == "q") break;
             if (input == "undo") {
                 if (state.Undo()) {
                     std::cout << "Undone!" << std::endl;
                     continue;
                 } else {
                     std::cout << "Nothing to undo." << std::endl;
                     continue;
                 }
             }
             
             std::vector<int> kept;
             if (input.empty()) {
                 kept = to_keep;
             } else if (input.substr(0, 4) == "keep") {
                 std::string values = input.substr(4);
                 std::vector<int> keep_values = ParseDice(values);
                 std::vector<int> indices = FindKeepIndices(state.current_dice, keep_values);
                 for (int idx : indices) {
                     kept.push_back(state.current_dice[idx]);
                 }
             } else {
                 std::cout << "Invalid input. Use 'keep <values>' or press Enter." << std::endl;
                 continue;
             }
             
             state.SaveState();
             state.frozen_dice = kept;
             state.rolls_left = 0;
             
             // Need third roll
             int needed = 5 - kept.size();
             if (needed > 0) {
                 std::cout << "\n>>> Enter remaining dice roll (need " << needed << " dice): ";
                 std::getline(std::cin, input);
                 
                 if (input == "quit" || input == "q") break;
                 if (input == "undo") {
                     state.Undo();
                     continue;
                 }
                 
                 std::vector<int> new_dice = ParseDice(input);
                 if ((int)new_dice.size() != needed) {
                     std::cout << "Error: Need " << needed << " dice. You entered " 
                               << new_dice.size() << "." << std::endl;
                     state.Undo();
                     continue;
                 }
                 
                 state.current_dice = kept;
                 state.current_dice.insert(state.current_dice.end(), 
                                          new_dice.begin(), new_dice.end());
                 std::sort(state.current_dice.begin(), state.current_dice.end());
             } else {
                 // Kept all 5 dice
                 state.current_dice = kept;
             }
             
        } else {
            // Final roll - recommend scoring category
            if (state.current_dice.size() != 5) {
                std::cout << "Error: Invalid dice state. Need exactly 5 dice." << std::endl;
                break;
            }
            
            int full_mask = BuildFullMask(state.mask, state.y_state);
            auto top_choices = strategy->GetTopCategories(state.current_dice, full_mask, 
                                                          state.upper, state.y_state, 3);
            
            if (top_choices.empty()) {
                std::cout << "Error: No valid categories available." << std::endl;
                break;
            }
            
            int category = top_choices[0].first;
            
            if (category < 0 || category >= 13) {
                std::cout << "Error: Invalid category recommendation." << std::endl;
                break;
            }
            
            if (state.IsCategoryFilled(category)) {
                std::cout << "Error: Recommended category already filled. Finding alternative..." 
                          << std::endl;
                category = -1;
                for (int i = 0; i < 13; i++) {
                    if (!state.IsCategoryFilled(i)) {
                        category = i;
                        break;
                    }
                }
                if (category == -1) {
                    std::cout << "All categories filled!" << std::endl;
                    break;
                }
            }
            
            std::cout << "\n>>> Top 3 Recommendations:" << std::endl;
            for (size_t i = 0; i < top_choices.size(); ++i) {
                int cat = top_choices[i].first;
                float ev = top_choices[i].second;
                int immediate_score = CalculateScore(cat, state.current_dice);
                std::cout << "  " << (i + 1) << ". " << CATEGORY_NAMES[cat] 
                          << " (cat " << cat << "): " << immediate_score 
                          << " points, EV = " << std::fixed << std::setprecision(1) << ev << std::endl;
            }
            std::cout << "\n>>> Recommended: Score in '" << CATEGORY_NAMES[category] 
                      << "' (category " << category << ")";
            std::cout << "\n    Press Enter to accept, or type alternative (e.g., 'score 3'): ";
             std::getline(std::cin, input);
             
             if (input == "quit" || input == "q") break;
             if (input == "undo") {
                 if (state.Undo()) {
                     std::cout << "Undone!" << std::endl;
                     continue;
                 } else {
                     std::cout << "Nothing to undo." << std::endl;
                     continue;
                 }
             }
             
             int score_category = category;
             if (!input.empty()) {
                 if (input.substr(0, 5) == "score") {
                     try {
                         int cat = std::stoi(input.substr(5));
                         if (cat >= 0 && cat < 13 && !state.IsCategoryFilled(cat)) {
                             score_category = cat;
                         } else {
                             std::cout << "Invalid category or already filled." << std::endl;
                             continue;
                         }
                     } catch (...) {
                         std::cout << "Invalid category number." << std::endl;
                         continue;
                     }
                 } else {
                     std::cout << "Invalid input. Use 'score <number>' or press Enter." << std::endl;
                     continue;
                 }
             }
             
             // Calculate score and update state
             int score = CalculateScore(score_category, state.current_dice);
             
             state.SaveState();
             
             // Update state based on category
             if (score_category == 12) {
                 // Yahtzee - update y_state, not mask
                 state.y_state = (score == 50) ? 2 : 1;
             } else {
                 // Other categories - update mask
                 state.mask |= (1 << score_category);
             }
             
             state.total_score += score;
             
             if (score_category < 6) {
                 state.upper = std::min(63, state.upper + score);
             }
             
             std::cout << "\n✓ Scored " << score << " points in " 
                       << CATEGORY_NAMES[score_category] << "!" << std::endl;
             
             // Next turn
             state.turn++;
             state.rolls_left = 2;
             state.current_dice.clear();
             state.frozen_dice.clear();
         }
     }
     
     std::cout << "\n" << std::string(60, '=') << std::endl;
     std::cout << "Game Complete!" << std::endl;
     std::cout << "Final Score: " << state.GetTotalScore() << std::endl;
     std::cout << std::string(60, '=') << std::endl;
     
     delete strategy;
     return 0;
 }