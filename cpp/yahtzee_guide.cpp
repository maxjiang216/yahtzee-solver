/**
 * Yahtzee Interactive Guide
 * 
 * An interactive program that uses the optimal Yahtzee strategy to guide gameplay.
 * Like "Stockfish for Yahtzee" - provides optimal recommendations at each step.
 * 
 * COMPILATION:
 *   g++ -O3 -o yahtzee_guide yahtzee_guide.cpp yahtzee_single_ev_max.cpp -fopenmp
 * 
 * USAGE:
 *   ./yahtzee_guide [strategy_file.bin]
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

// Include the solver code - compile with yahtzee_single_ev_max.cpp
// Compilation: g++ -O3 -o yahtzee_guide yahtzee_guide.cpp yahtzee_single_ev_max.cpp -fopenmp

// Forward declare YahtzeeStrategy - will use SQLite version
class YahtzeeStrategy {
public:
    YahtzeeStrategy(const std::string& filename);
    int get_optimal_category(std::vector<int> dice, int mask, int upper, int y_state);
    int get_optimal_keep(std::vector<int> current_dice, int mask, int upper, int y_state, int rolls_left);
};

// Declare scoring function - defined in yahtzee_single_ev_max.cpp
int calculate_score(int category, const std::vector<int>& dice);

// Category names for display
const char* CATEGORY_NAMES[] = {
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "Three of a Kind", "Four of a Kind", "Full House",
    "Small Straight", "Large Straight", "Chance", "Yahtzee"
};

// Simple game state tracker
struct GameState {
    int mask = 0;           // Bitmask of filled categories
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
    
    void save_state() {
        history.push_back({mask, upper, y_state, turn, rolls_left, total_score, current_dice, frozen_dice});
    }
    
    bool undo() {
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
    
    bool is_category_filled(int cat) const {
        return (mask >> cat) & 1;
    }
    
    int get_total_score() const {
        int score = total_score;
        if (upper >= 63) score += 35;
        return score;
    }
};

// Parse dice input - accepts "12345" or "1 2 3 4 5" or "1,2,3,4,5"
std::vector<int> parse_dice(const std::string& input) {
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
        // Remove commas
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
void display_state(const GameState& state) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Turn " << state.turn << " / 13";
    std::cout << "  |  Upper: " << state.upper;
    if (state.upper >= 63) std::cout << " (+35 bonus)";
    std::cout << "  |  Total: " << state.get_total_score() << std::endl;
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
        if (state.is_category_filled(i)) {
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
std::string keep_mask_to_string(int mask, const std::vector<int>& dice) {
    std::string result;
    for (int i = 0; i < 5; i++) {
        if ((mask >> i) & 1) {
            if (!result.empty()) result += " ";
            result += std::to_string(dice[i]);
        }
    }
    return result;
}

// Find dice indices to keep based on values
std::vector<int> find_keep_indices(const std::vector<int>& dice, const std::vector<int>& keep_values) {
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

int main(int argc, char* argv[]) {
    std::string strategy_file = "yahtzee_optimal.db";
    if (argc > 1) {
        strategy_file = argv[1];
    }
    
    std::cout << "=== Yahtzee Interactive Guide ===" << std::endl;
    std::cout << "Loading strategy from: " << strategy_file << std::endl;
    
    // Note: Compile with: g++ -O3 -o yahtzee_guide yahtzee_guide.cpp yahtzee_single_ev_max.cpp -fopenmp
    // The YahtzeeStrategy class is defined in yahtzee_single_ev_max.cpp
    
    YahtzeeStrategy* strategy = nullptr;
    try {
        strategy = new YahtzeeStrategy(strategy_file);
        std::cout << "Strategy loaded successfully!" << std::endl;
    } catch (...) {
        std::cerr << "Error: Could not load strategy file: " << strategy_file << std::endl;
        std::cerr << "Make sure you've run the solver first to generate the strategy file." << std::endl;
        return 1;
    }
    
    std::cout << "\nWelcome! Enter dice rolls to get optimal recommendations." << std::endl;
    std::cout << "Type 'help' for commands, 'quit' to exit.\n" << std::endl;
    
    GameState state;
    std::string input;
    
    while (state.turn <= 13) {
        display_state(state);
        
        if (state.rolls_left == 2) {
            // Start of turn - need initial roll OR after first roll with 2 rolls left
            if (state.current_dice.empty()) {
                // Need initial roll
                std::cout << "\n>>> Enter your dice roll (e.g., '12345'): ";
                std::getline(std::cin, input);
                
                if (input == "quit" || input == "q") break;
                if (input == "help") {
                    std::cout << "\nCommands:\n";
                    std::cout << "  Enter dice: '12345' or '1 2 3 4 5'\n";
                    std::cout << "  Accept recommendation: Press Enter\n";
                    std::cout << "  Alternative keep: 'keep 12' (keep dice with values 1 and 2)\n";
                    std::cout << "  Alternative score: 'score 3' (score in category 3)\n";
                    std::cout << "  Undo: 'undo'\n";
                    std::cout << "  Quit: 'quit' or 'q'\n";
                    continue;
                }
                
                std::vector<int> dice = parse_dice(input);
                if (dice.size() != 5) {
                    std::cout << "Error: Need exactly 5 dice. You entered " << dice.size() << " values." << std::endl;
                    continue;
                }
                
                std::sort(dice.begin(), dice.end());
                state.current_dice = dice;
                state.frozen_dice.clear();
            }
            
            // After first roll (or initial roll) - recommend what to keep with 2 rolls left
            // Get optimal keep recommendation
            int keep_mask = strategy->get_optimal_keep(state.current_dice, state.mask, state.upper, state.y_state, 2);
            
            std::vector<int> to_keep;
            for (int i = 0; i < 5; i++) {
                if ((keep_mask >> i) & 1) {
                    to_keep.push_back(state.current_dice[i]);
                }
            }
            
            std::cout << "\n>>> Recommended: Keep " << keep_mask_to_string(keep_mask, state.current_dice);
            std::cout << "\n    Press Enter to accept, or type alternative (e.g., 'keep 12'): ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "q") break;
            if (input == "undo") {
                if (state.undo()) {
                    std::cout << "Undone!" << std::endl;
                    continue;
                } else {
                    std::cout << "Nothing to undo." << std::endl;
                    continue;
                }
            }
            
            std::vector<int> kept;
            if (input.empty()) {
                // Accept recommendation
                kept = to_keep;
            } else if (input.substr(0, 4) == "keep") {
                // Parse alternative keep
                std::string values = input.substr(4);
                std::vector<int> keep_values = parse_dice(values);
                std::vector<int> indices = find_keep_indices(state.current_dice, keep_values);
                for (int idx : indices) {
                    kept.push_back(state.current_dice[idx]);
                }
            } else {
                std::cout << "Invalid input. Use 'keep <values>' or press Enter." << std::endl;
                continue;
            }
            
            if (kept.empty()) {
                std::cout << "Error: Must keep at least one die." << std::endl;
                continue;
            }
            
            state.save_state();
            state.frozen_dice = kept;
            state.rolls_left = 1;  // After keeping, we have 1 roll left
            
            // Need second roll
            int needed = 5 - kept.size();
            std::cout << "\n>>> Enter remaining dice roll (need " << needed << " dice): ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "q") break;
            if (input == "undo") {
                state.undo();
                continue;
            }
            
            std::vector<int> new_dice = parse_dice(input);
            if ((int)new_dice.size() != needed) {
                std::cout << "Error: Need " << needed << " dice. You entered " << new_dice.size() << "." << std::endl;
                state.undo();
                continue;
            }
            
            // Combine frozen and new dice
            state.current_dice = kept;
            state.current_dice.insert(state.current_dice.end(), new_dice.begin(), new_dice.end());
            std::sort(state.current_dice.begin(), state.current_dice.end());
            
        } else if (state.rolls_left == 1) {
            // After second roll - recommend what to keep with 1 roll left
            // Get optimal keep recommendation
            int keep_mask = strategy->get_optimal_keep(state.current_dice, state.mask, state.upper, state.y_state, 1);
            
            std::vector<int> to_keep;
            for (int i = 0; i < 5; i++) {
                if ((keep_mask >> i) & 1) {
                    to_keep.push_back(state.current_dice[i]);
                }
            }
            
            std::cout << "\n>>> Recommended: Keep " << keep_mask_to_string(keep_mask, state.current_dice);
            std::cout << "\n    Press Enter to accept, or type alternative (e.g., 'keep 12'): ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "q") break;
            if (input == "undo") {
                if (state.undo()) {
                    std::cout << "Undone!" << std::endl;
                    continue;
                } else {
                    std::cout << "Nothing to undo." << std::endl;
                    continue;
                }
            }
            
            std::vector<int> kept;
            if (input.empty()) {
                // Accept recommendation
                kept = to_keep;
            } else if (input.substr(0, 4) == "keep") {
                // Parse alternative keep
                std::string values = input.substr(4);
                std::vector<int> keep_values = parse_dice(values);
                std::vector<int> indices = find_keep_indices(state.current_dice, keep_values);
                for (int idx : indices) {
                    kept.push_back(state.current_dice[idx]);
                }
            } else {
                std::cout << "Invalid input. Use 'keep <values>' or press Enter." << std::endl;
                continue;
            }
            
            if (kept.empty()) {
                std::cout << "Error: Must keep at least one die." << std::endl;
                continue;
            }
            
            state.save_state();
            state.frozen_dice = kept;
            state.rolls_left = 0;  // After keeping, we have 0 rolls left (must score)
            
            // Need third roll
            int needed = 5 - kept.size();
            std::cout << "\n>>> Enter remaining dice roll (need " << needed << " dice): ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "q") break;
            if (input == "undo") {
                state.undo();
                continue;
            }
            
            std::vector<int> new_dice = parse_dice(input);
            if ((int)new_dice.size() != needed) {
                std::cout << "Error: Need " << needed << " dice. You entered " << new_dice.size() << "." << std::endl;
                state.undo();
                continue;
            }
            
            // Combine frozen and new dice
            state.current_dice = kept;
            state.current_dice.insert(state.current_dice.end(), new_dice.begin(), new_dice.end());
            std::sort(state.current_dice.begin(), state.current_dice.end());
            
        } else {
            // Final roll - recommend scoring category
            // Get optimal category recommendation
            int category = strategy->get_optimal_category(state.current_dice, state.mask, state.upper, state.y_state);
            
            if (category < 0 || category >= 13) {
                std::cout << "Error: Invalid category recommendation." << std::endl;
                break;
            }
            
            if (state.is_category_filled(category)) {
                std::cout << "Error: Recommended category already filled. Finding alternative..." << std::endl;
                // Find first open category as fallback
                category = -1;
                for (int i = 0; i < 13; i++) {
                    if (!state.is_category_filled(i)) {
                        category = i;
                        break;
                    }
                }
                if (category == -1) {
                    std::cout << "All categories filled!" << std::endl;
                    break;
                }
            }
            
            std::cout << "\n>>> Recommended: Score in '" << CATEGORY_NAMES[category] << "' (category " << category << ")";
            std::cout << "\n    Press Enter to accept, or type alternative (e.g., 'score 3'): ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "q") break;
            if (input == "undo") {
                if (state.undo()) {
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
                        if (cat >= 0 && cat < 13 && !state.is_category_filled(cat)) {
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
            int score = calculate_score(score_category, state.current_dice);
            
            state.save_state();
            state.mask |= (1 << score_category);
            state.total_score += score;
            
            if (score_category < 6) {
                state.upper = std::min(63, state.upper + score);
            }
            
            if (score_category == 12) { // Yahtzee
                state.y_state = (score == 50) ? 2 : 1;
            }
            
            std::cout << "\n✓ Scored " << score << " points in " << CATEGORY_NAMES[score_category] << "!" << std::endl;
            
            // Next turn
            state.turn++;
            state.rolls_left = 2;
            state.current_dice.clear();
            state.frozen_dice.clear();
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Game Complete!" << std::endl;
    std::cout << "Final Score: " << state.get_total_score() << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
