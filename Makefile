# Makefile for the Yahtzee solvers
#
# Build targets:
#   make all            - Build every program
#   make solver         - Float EV solver            (yahtzee_solver)
#   make rational       - Exact rational EV solver   (yahtzee_rational_solver)
#   make pmf            - Exact PMF solver           (yahtzee_pmf_solver)
#   make guide          - Interactive guide          (yahtzee_guide)
#   make demo           - Strategy query demo        (strategy_demo)
#
# Run targets:
#   make solve          - Generate yahtzee_optimal_v4.bin   (~4 s, 3 MB)
#   make rational-solve - Generate yahtzee_rational_v1.bin  (~5 s, 54 MB)
#   make pmf-solve      - Generate yahtzee_pmf_v2.bin + yahtzee_pmf_exact_start.txt (~40 s, ~4 GB RAM)
#   make pmf-solve-low  - Same with ~2 GB RAM (spills exact layers to temporary files)
#   make test           - Build and run all unit tests (generates the EV table if needed)
#
# Cleanup:
#   make clean          - Remove executables
#   make distclean      - Also remove generated data files

CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
OMPFLAGS = -fopenmp
# The exact solvers are integer-only, so tuning for the build machine is safe
EXACTFLAGS = -march=native

HEADERS = $(wildcard cpp/*.h)
STRATEGY_SRC = cpp/yahtzee_strategy.cpp

SOLVER_BIN = yahtzee_solver
RATIONAL_BIN = yahtzee_rational_solver
PMF_BIN = yahtzee_pmf_solver
GUIDE_BIN = yahtzee_guide
DEMO_BIN = strategy_demo
TEST_BINS = test_strategy test_keep_lattice

STRATEGY_FILE = yahtzee_optimal_v4.bin
RATIONAL_FILE = yahtzee_rational_v1.bin
PMF_FILE = yahtzee_pmf_v2.bin
PMF_EXACT_FILE = yahtzee_pmf_exact_start.txt

.PHONY: all solver rational pmf guide demo solve rational-solve pmf-solve pmf-solve-low \
        test clean distclean help

all: $(SOLVER_BIN) $(RATIONAL_BIN) $(PMF_BIN) $(GUIDE_BIN) $(DEMO_BIN) $(TEST_BINS)

# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

solver: $(SOLVER_BIN)
$(SOLVER_BIN): cpp/yahtzee_optimized_solver.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

rational: $(RATIONAL_BIN)
$(RATIONAL_BIN): cpp/yahtzee_rational_solver.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(EXACTFLAGS) $(OMPFLAGS) -o $@ $<

pmf: $(PMF_BIN)
$(PMF_BIN): cpp/yahtzee_pmf_solver.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(EXACTFLAGS) $(OMPFLAGS) -o $@ $<

guide: $(GUIDE_BIN)
$(GUIDE_BIN): cpp/yahtzee_guide.cpp $(STRATEGY_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ cpp/yahtzee_guide.cpp $(STRATEGY_SRC)

demo: $(DEMO_BIN)
$(DEMO_BIN): $(STRATEGY_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DBUILD_DEMO -o $@ $(STRATEGY_SRC)

test_strategy: cpp/test_strategy.cpp $(STRATEGY_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ cpp/test_strategy.cpp $(STRATEGY_SRC)

test_keep_lattice: cpp/test_keep_lattice.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

solve: $(STRATEGY_FILE)
$(STRATEGY_FILE): $(SOLVER_BIN)
	./$(SOLVER_BIN)

rational-solve: $(RATIONAL_BIN)
	./$(RATIONAL_BIN) $(RATIONAL_FILE)

pmf-solve: $(PMF_BIN)
	./$(PMF_BIN) $(PMF_FILE) $(PMF_EXACT_FILE)

pmf-solve-low: $(PMF_BIN)
	./$(PMF_BIN) --low-memory $(PMF_FILE) $(PMF_EXACT_FILE)

# ---------------------------------------------------------------------------
# Tests and cleanup
# ---------------------------------------------------------------------------

test: $(TEST_BINS) $(STRATEGY_FILE)
	./test_keep_lattice
	./test_strategy

clean:
	rm -f $(SOLVER_BIN) $(RATIONAL_BIN) $(PMF_BIN) $(GUIDE_BIN) $(DEMO_BIN) $(TEST_BINS)

distclean: clean
	rm -f $(STRATEGY_FILE) $(RATIONAL_FILE) $(PMF_FILE) $(PMF_EXACT_FILE)

help:
	@sed -n '1,21p' Makefile
