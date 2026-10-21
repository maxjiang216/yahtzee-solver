#ifndef YAHTZEE_STRATEGY_H
#define YAHTZEE_STRATEGY_H

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "yahtzee_keep_lattice.h"

// -------------------------------------------------------------------------
// SCORING FUNCTIONS (forward declarations)
// -------------------------------------------------------------------------

/**
 * Calculate score for a given category and dice
 * 
 * @param category Category index (0-12)
 * @param dice Dice values (5 dice)
 * @return Score for this category
 */
int CalculateScore(int category, const std::vector<int>& dice);

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
// DICE INDEXER
// -------------------------------------------------------------------------

class DiceIndexer {
 public:
  std::map<std::vector<int>, int> dice_to_id;
  std::vector<std::vector<int>> id_to_dice;

  explicit DiceIndexer(int num_dice);

  int GetId(std::vector<int> dice) const;
  const std::vector<int>& GetDice(int id) const;

 private:
  void Generate(std::vector<int>& current, int start_face, int max_depth);
};

// -------------------------------------------------------------------------
// YAHTZEE STRATEGY
// -------------------------------------------------------------------------

/**
 * YahtzeeStrategy: Fast query interface for optimal decisions
 * 
 * Loads the start-of-turn EV table (3 MB) and provides two main queries:
 * 1. GetOptimalCategory() - Which category to score in
 * 2. GetOptimalKeep() - Which dice to keep for reroll
 * 
 * Mid-turn values are recomputed from the board EVs with KeepLattice:
 * one keep query is 252 scoring evaluations plus one or two lattice passes.
 */
class YahtzeeStrategy {
 public:
  explicit YahtzeeStrategy(const std::string& filename);
  ~YahtzeeStrategy() = default;

  /**
   * Get optimal category to score in
   * 
   * @param dice Current dice values (5 dice)
   * @param mask Bitmask of filled categories (13 bits)
   * @param upper Upper section score (0-63)
   * @param y_state Yahtzee state (0=open, 1=filled(0), 2=filled(50))
   * @return Category index (0-12) to score in
   */
  int GetOptimalCategory(std::vector<int> dice, int mask, int upper, int y_state);

  /**
   * Get optimal dice to keep for reroll
   * 
   * @param current_dice Current dice values (5 dice)
   * @param mask Bitmask of filled categories
   * @param upper Upper section score
   * @param y_state Yahtzee state
   * @param rolls_left Number of rolls remaining (1 or 2)
   * @return Bitmask indicating which dice positions to keep (0-4)
   *         Example: 0b10110 = keep dice at positions 1, 2, 4
   */
  int GetOptimalKeep(std::vector<int> current_dice, int mask, int upper, 
                     int y_state, int rolls_left);

  /**
   * Get top N keep choices with their expected values
   * 
   * @param current_dice Current dice values (5 dice)
   * @param mask Bitmask of filled categories
   * @param upper Upper section score
   * @param y_state Yahtzee state
   * @param rolls_left Number of rolls remaining (1 or 2)
   * @param top_n Number of top choices to return
   * @return Vector of pairs: (keep_mask, expected_value)
   */
  std::vector<std::pair<int, float>> GetTopKeepChoices(
      std::vector<int> current_dice, int mask, int upper, int y_state, 
      int rolls_left, int top_n = 3) const;

  /**
   * Get expected value for a board state (start of turn)
   * 
   * @return Expected score from this position playing optimally
   */
  float GetBoardEV(int mask, int upper, int y_state) const;

  /**
   * Get top N scoring categories with their expected values
   * 
   * @param dice Current dice values (5 dice)
   * @param mask Bitmask of filled categories (13 bits)
   * @param upper Upper section score (0-63)
   * @param y_state Yahtzee state (0=open, 1=filled(0), 2=filled(50))
   * @param top_n Number of top choices to return
   * @return Vector of pairs: (category_index, expected_value)
   */
  std::vector<std::pair<int, float>> GetTopCategories(
      std::vector<int> dice, int mask, int upper, int y_state, int top_n = 3) const;

 private:
  // Loaded start-of-turn EV table
  std::vector<float> board_ev_;

  DiceIndexer indexer_;
  KeepLattice lattice_;

  // Indexing function
  inline size_t BoardIndex(int mask, int upper, int y_state) const {
    // Strip bit 12 from mask, as Yahtzee is handled by y_state
    mask &= ~(1 << CAT_YAHTZEE);
    return (mask * MAX_UPPER_SCORE + upper) * NUM_Y_STATES + y_state;
  }

  // exp[sid] = EV of keeping subset sid with rolls_left (1 or 2) rolls left
  void KeepExpectations(int mask, int upper, int y_state, int rolls_left,
                        double* exp) const;

  // Best scoring move helper
  std::pair<float, int> GetBestScoringMove(int dice_id, int mask, int upper, 
                                           int y_state) const;
};

#endif  // YAHTZEE_STRATEGY_H

