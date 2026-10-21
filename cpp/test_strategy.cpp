/**
 * Unit tests for YahtzeeStrategy
 * 
 * Tests:
 * 1. Yahtzee scored with 50 should have higher EV than Yahtzee scored with 0
 *    (because of multi-Yahtzee possibility)
 * 2. Higher upper score should have at least as great EV
 *    (greater likelihood of upper score bonus)
 */

#include "yahtzee_strategy.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

void TestYahtzeeScoring() {
  std::cout << "\n=== Test 1: Yahtzee Scoring (50 vs 0) ===" << std::endl;
  
  YahtzeeStrategy strategy("yahtzee_optimal_v4.bin");
  
  int failures = 0;
  int tests = 0;
  
  // Test various positions where Yahtzee is filled
  // We'll test with different masks and upper scores
  std::vector<int> test_masks = {
    0,                    // No categories filled
    1,                    // Only 1s filled
    0b111111,             // All upper filled
    0b111111000000,       // All lower filled except Yahtzee
    0b101010101010,      // Mixed pattern
  };
  
  std::vector<int> test_upper = {0, 30, 50, 63};
  
  for (int mask : test_masks) {
    // Skip if Yahtzee bit is already set in mask
    if (mask & (1 << CAT_YAHTZEE)) continue;
    
    for (int upper : test_upper) {
      tests++;
      
      // Get EV with Yahtzee scored as 0 (y_state = 1)
      float ev_yahtzee_0 = strategy.GetBoardEV(mask, upper, 1);
      
      // Get EV with Yahtzee scored as 50 (y_state = 2)
      float ev_yahtzee_50 = strategy.GetBoardEV(mask, upper, 2);
      
      bool passed = ev_yahtzee_50 >= ev_yahtzee_0;
      
      if (!passed) {
        failures++;
        std::cout << "  FAIL: mask=0x" << std::hex << mask << std::dec 
                  << ", upper=" << upper << std::endl;
        std::cout << "    EV(Yahtzee=0):  " << std::fixed << std::setprecision(2) 
                  << ev_yahtzee_0 << std::endl;
        std::cout << "    EV(Yahtzee=50): " << ev_yahtzee_50 << std::endl;
        std::cout << "    Difference: " << (ev_yahtzee_50 - ev_yahtzee_0) << std::endl;
      } else if (ev_yahtzee_50 > ev_yahtzee_0) {
        std::cout << "  PASS: mask=0x" << std::hex << mask << std::dec 
                  << ", upper=" << upper 
                  << " (EV difference: +" << std::fixed << std::setprecision(2)
                  << (ev_yahtzee_50 - ev_yahtzee_0) << ")" << std::endl;
      }
    }
  }
  
  std::cout << "\nTest 1 Results: " << (tests - failures) << "/" << tests 
            << " passed" << std::endl;
  if (failures > 0) {
    std::cout << "  " << failures << " failures detected!" << std::endl;
  }
}

void TestUpperScoreMonotonicity() {
  std::cout << "\n=== Test 2: Upper Score Monotonicity ===" << std::endl;
  
  YahtzeeStrategy strategy("yahtzee_optimal_v4.bin");
  
  int failures = 0;
  int tests = 0;
  
  // Test various positions with different upper scores
  std::vector<int> test_masks = {
    0,                    // No categories filled
    0b111111,             // All upper filled
    0b111000,             // Some upper filled
    0b000111,             // Some lower filled
    0b101010101010,       // Mixed pattern
  };
  
  std::vector<int> test_y_states = {0, 1, 2};  // Yahtzee open, filled(0), filled(50)
  
  for (int mask : test_masks) {
    for (int y_state : test_y_states) {
      // Test pairs of upper scores where upper1 < upper2
      for (int upper1 = 0; upper1 < 63; upper1 += 10) {
        for (int upper2 = upper1 + 1; upper2 <= 63; upper2 += 10) {
          tests++;
          
          float ev_lower = strategy.GetBoardEV(mask, upper1, y_state);
          float ev_higher = strategy.GetBoardEV(mask, upper2, y_state);
          
          // Use epsilon for floating point comparison (allow small differences due to precision)
          const float epsilon = 0.01f;
          bool passed = ev_higher >= (ev_lower - epsilon);
          
          if (!passed) {
            failures++;
            std::cout << "  FAIL: mask=0x" << std::hex << mask << std::dec 
                      << ", y_state=" << y_state 
                      << ", upper1=" << upper1 << ", upper2=" << upper2 << std::endl;
            std::cout << "    EV(upper=" << upper1 << "): " << std::fixed 
                      << std::setprecision(2) << ev_lower << std::endl;
            std::cout << "    EV(upper=" << upper2 << "): " << ev_higher << std::endl;
            std::cout << "    Difference: " << (ev_higher - ev_lower) << std::endl;
          }
        }
      }
    }
  }
  
  std::cout << "\nTest 2 Results: " << (tests - failures) << "/" << tests 
            << " passed" << std::endl;
  if (failures > 0) {
    std::cout << "  " << failures << " failures detected!" << std::endl;
  }
}

void TestYahtzeeOpenVsScoredZero() {
  std::cout << "\n=== Test 3: Yahtzee Open vs Scored Zero ===" << std::endl;
  
  YahtzeeStrategy strategy("yahtzee_optimal_v4.bin");
  
  int failures = 0;
  int tests = 0;
  
  // Test various positions where Yahtzee is not yet filled
  std::vector<int> test_masks = {
    0,                    // No categories filled
    1,                    // Only 1s filled
    0b111111,             // All upper filled
    0b111111000000,       // All lower filled except Yahtzee
    0b101010101010,       // Mixed pattern
  };
  
  std::vector<int> test_upper = {0, 20, 40, 63};
  
  for (int mask : test_masks) {
    // Skip if Yahtzee bit is already set in mask (shouldn't happen, but be safe)
    if (mask & (1 << CAT_YAHTZEE)) continue;
    
    for (int upper : test_upper) {
      tests++;
      
      // Get EV with Yahtzee open (y_state = 0)
      float ev_yahtzee_open = strategy.GetBoardEV(mask, upper, 0);
      
      // Get EV with Yahtzee scored as 0 (y_state = 1)
      float ev_yahtzee_zero = strategy.GetBoardEV(mask, upper, 1);
      
      // Yahtzee open should have higher EV (can still score 50 points)
      bool passed = ev_yahtzee_open > ev_yahtzee_zero;
      
      if (!passed) {
        failures++;
        std::cout << "  FAIL: mask=0x" << std::hex << mask << std::dec 
                  << ", upper=" << upper << std::endl;
        std::cout << "    EV(Yahtzee open):  " << std::fixed << std::setprecision(2) 
                  << ev_yahtzee_open << std::endl;
        std::cout << "    EV(Yahtzee=0):     " << ev_yahtzee_zero << std::endl;
        std::cout << "    Difference: " << (ev_yahtzee_open - ev_yahtzee_zero) << std::endl;
      } else if (ev_yahtzee_open > ev_yahtzee_zero + 0.01f) {
        // Only print if there's a meaningful difference (>0.01)
        std::cout << "  PASS: mask=0x" << std::hex << mask << std::dec 
                  << ", upper=" << upper 
                  << " (EV difference: +" << std::fixed << std::setprecision(2)
                  << (ev_yahtzee_open - ev_yahtzee_zero) << ")" << std::endl;
      }
    }
  }
  
  std::cout << "\nTest 3 Results: " << (tests - failures) << "/" << tests 
            << " passed" << std::endl;
  if (failures > 0) {
    std::cout << "  " << failures << " failures detected!" << std::endl;
  }
}

void TestSpecificCases() {
  std::cout << "\n=== Test 4: Specific Test Cases ===" << std::endl;
  
  YahtzeeStrategy strategy("yahtzee_optimal_v4.bin");
  
  // Test case 1: Empty board, Yahtzee scored differently
  std::cout << "\nCase 1: Empty board (mask=0, upper=0)" << std::endl;
  float ev_empty_yahtzee_0 = strategy.GetBoardEV(0, 0, 1);
  float ev_empty_yahtzee_50 = strategy.GetBoardEV(0, 0, 2);
  std::cout << "  EV(Yahtzee=0):  " << std::fixed << std::setprecision(2) 
            << ev_empty_yahtzee_0 << std::endl;
  std::cout << "  EV(Yahtzee=50): " << ev_empty_yahtzee_50 << std::endl;
  std::cout << "  Difference: " << (ev_empty_yahtzee_50 - ev_empty_yahtzee_0) << std::endl;
  std::cout << "  Status: " << (ev_empty_yahtzee_50 >= ev_empty_yahtzee_0 ? "PASS" : "FAIL") << std::endl;
  
  // Test case 2: Upper score progression
  std::cout << "\nCase 2: Empty board, upper score progression" << std::endl;
  for (int upper = 0; upper <= 63; upper += 15) {
    float ev = strategy.GetBoardEV(0, upper, 0);
    std::cout << "  EV(upper=" << std::setw(2) << upper << "): " 
              << std::fixed << std::setprecision(2) << ev << std::endl;
  }
  
  // Test case 3: Same mask, different upper scores
  std::cout << "\nCase 3: Same mask (0b111111), different upper scores" << std::endl;
  int mask_all_upper = 0b111111;
  for (int upper = 0; upper <= 63; upper += 20) {
    float ev = strategy.GetBoardEV(mask_all_upper, upper, 0);
    std::cout << "  EV(upper=" << std::setw(2) << upper << "): " 
              << std::fixed << std::setprecision(2) << ev << std::endl;
  }
}

int main() {
  std::cout << "Yahtzee Strategy Unit Tests" << std::endl;
  std::cout << "==========================" << std::endl;
  
  try {
    TestYahtzeeScoring();
    TestUpperScoreMonotonicity();
    TestYahtzeeOpenVsScoredZero();
    TestSpecificCases();
    
    std::cout << "\n=== All Tests Complete ===" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

