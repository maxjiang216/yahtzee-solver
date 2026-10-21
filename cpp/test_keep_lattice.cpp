/**
 * Unit tests for KeepLattice
 *
 * Tests:
 * 1. Lattice expectations equal brute-force enumeration of every reroll
 *    (all 6^(5-|K|) ordered outcomes) for every kept subset K
 * 2. Keeps(d) is exactly the set of distinct sub-multisets of outcome d
 * 3. Subset ids are ordered by size and five-dice ids match DiceIndexer
 */

#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "yahtzee_keep_lattice.h"
#include "yahtzee_rules.h"

static int failures = 0;

static void Check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::cout << "  FAIL: " << what << std::endl;
  }
}

void TestExpectations(const KeepLattice& lat, const DiceIndexer& idx) {
  std::cout << "\n=== Test 1: Lattice expectations vs brute force ===" << std::endl;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);
  float values[NUM_DICE_COMBOS];
  for (float& v : values) v = dist(rng);

  double exp[KeepLattice::kNumSubsets];
  lat.Expectations(values, exp);

  double worst = 0;
  for (int sid = 0; sid < KeepLattice::kNumSubsets; ++sid) {
    const std::vector<int>& kept = lat.Dice(sid);
    int needed = 5 - static_cast<int>(kept.size());
    int total = 1;
    for (int i = 0; i < needed; ++i) total *= 6;
    double sum = 0;
    for (int j = 0; j < total; ++j) {
      std::vector<int> dice = kept;
      for (int r = 0, t = j; r < needed; ++r, t /= 6) dice.push_back(t % 6 + 1);
      sum += values[idx.GetId(dice)];
    }
    worst = std::max(worst, std::fabs(sum / total - exp[sid]));
  }
  Check(worst < 1e-9, "max |lattice - enumeration| = " + std::to_string(worst));
  std::cout << "  462 subsets, max difference " << worst << std::endl;
}

void TestKeeps(const KeepLattice& lat, const DiceIndexer& idx) {
  std::cout << "\n=== Test 2: Keep options per outcome ===" << std::endl;
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
    const std::vector<int>& dice = idx.GetDice(d);
    std::set<std::vector<int>> expected;
    for (int m = 0; m < 32; ++m) {
      std::vector<int> sub;
      for (int b = 0; b < 5; ++b) if ((m >> b) & 1) sub.push_back(dice[b]);
      expected.insert(sub);  // dice are sorted, so sub is sorted
    }
    std::set<std::vector<int>> got;
    for (int sid : lat.Keeps(d)) got.insert(lat.Dice(sid));
    Check(got == expected && got.size() == lat.Keeps(d).size(),
          "keeps of outcome " + std::to_string(d));
  }
  std::cout << "  252 outcomes checked" << std::endl;
}

void TestOrdering(const KeepLattice& lat, const DiceIndexer& idx) {
  std::cout << "\n=== Test 3: Subset id ordering ===" << std::endl;
  for (int sid = 1; sid < KeepLattice::kNumSubsets; ++sid) {
    Check(lat.Size(sid - 1) <= lat.Size(sid), "ids ascend by size at " + std::to_string(sid));
  }
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
    Check(lat.SubsetId(idx.GetDice(d)) == KeepLattice::kFirst5 + d,
          "outcome " + std::to_string(d) + " id");
  }
  for (int sid = 0; sid < KeepLattice::kFirst5; ++sid) {
    for (int f = 1; f <= 6; ++f) {
      std::vector<int> c = lat.Dice(sid);
      c.push_back(f);
      Check(lat.Children(sid)[f - 1] == lat.SubsetId(c), "child of " + std::to_string(sid));
    }
  }
  std::cout << "  462 ids checked" << std::endl;
}

int main() {
  KeepLattice lat;
  DiceIndexer idx(5);
  TestExpectations(lat, idx);
  TestKeeps(lat, idx);
  TestOrdering(lat, idx);
  std::cout << "\n" << (failures ? "FAILED" : "All keep lattice tests passed")
            << " (" << failures << " failures)" << std::endl;
  return failures ? 1 : 0;
}
