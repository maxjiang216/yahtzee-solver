# Makefile for Yahtzee Optimal Strategy Solver
# 
# Targets:
#   make solver     - Build the solver (generates strategy file)
#   make demo       - Build the strategy query demo
#   make guide      - Build the interactive guide (requires yahtzee_guide.cpp)
#   make all        - Build everything
#   make clean      - Remove all build artifacts
#   make solve      - Run the solver to generate strategy file
#   make test       - Run the demo to test the strategy

CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
LDFLAGS = -fopenmp

# Source files
SOLVER_SRC = cpp/yahtzee_optimized_solver.cpp
STRATEGY_SRC = cpp/yahtzee_strategy.cpp
GUIDE_SRC = cpp/yahtzee_guide_v3.cpp

# Object files
SOLVER_OBJ = yahtzee_optimized_solver.o
STRATEGY_OBJ = yahtzee_strategy.o

# Executables
SOLVER_BIN = yahtzee_solver
DEMO_BIN = strategy_demo
GUIDE_BIN = yahtzee_guide
TEST_BIN = test_strategy

# Strategy file
STRATEGY_FILE = yahtzee_optimal_v3.bin
PMF_FILE = yahtzee_pmf_v1.bin
PMF_BIN = yahtzee_pmf_solver
RATIONAL_BIN = yahtzee_rational_solver
RATIONAL_FILE = yahtzee_rational_v1.bin

# Default target
.PHONY: all
all: solver demo pmf

# Build PMF solver
.PHONY: pmf
pmf: $(PMF_BIN)

$(PMF_BIN): cpp/yahtzee_pmf_solver.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo "PMF solver built. Run 'make pmf-solve' to generate $(PMF_FILE)"

# Run PMF solver (requires yahtzee_optimal_v3.bin)
.PHONY: pmf-solve
pmf-solve: $(PMF_BIN)
	@if [ ! -f $(STRATEGY_FILE) ]; then \
		echo "Error: $(STRATEGY_FILE) not found. Run 'make solve' first."; \
		exit 1; \
	fi
	@echo "Running PMF solver (this may take 5-30 minutes)..."
	./$(PMF_BIN) $(STRATEGY_FILE)
	@echo "PMF file generated: $(PMF_FILE)"

# Build exact rational EV solver
.PHONY: rational
rational: $(RATIONAL_BIN)

$(RATIONAL_BIN): cpp/yahtzee_rational_solver.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Rational solver built. Run 'make rational-solve' to generate $(RATIONAL_FILE)"

# Run exact rational EV solver (~5 s on 4 cores)
.PHONY: rational-solve
rational-solve: $(RATIONAL_BIN)
	./$(RATIONAL_BIN) $(RATIONAL_FILE)

# Build solver
.PHONY: solver
solver: $(SOLVER_BIN)

$(SOLVER_BIN): $(SOLVER_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo ""
	@echo "Solver built successfully!"
	@echo "Run './$(SOLVER_BIN)' to generate the strategy file"
	@echo ""

# Build demo
.PHONY: demo
demo: $(DEMO_BIN)

$(DEMO_BIN): $(STRATEGY_SRC)
	$(CXX) $(CXXFLAGS) -DBUILD_DEMO -o $@ $<
	@echo ""
	@echo "Demo built successfully!"
	@echo "Make sure to run the solver first to generate $(STRATEGY_FILE)"
	@echo "Then run './$(DEMO_BIN)' to test queries"
	@echo ""

# Build guide (if source file exists)
.PHONY: guide
guide: $(GUIDE_BIN)

$(GUIDE_BIN): $(GUIDE_SRC) $(STRATEGY_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo ""
	@echo "Guide built successfully!"
	@echo "Run './$(GUIDE_BIN) $(STRATEGY_FILE)' to use the interactive guide"
	@echo ""

# Run solver to generate strategy file
.PHONY: solve
solve: $(SOLVER_BIN)
	@echo "Running solver (this may take 5-30 minutes)..."
	@echo ""
	./$(SOLVER_BIN)
	@echo ""
	@echo "Strategy file generated: $(STRATEGY_FILE)"
	@echo "File size: $$(du -h $(STRATEGY_FILE) | cut -f1)"
	@echo ""

# Build test
.PHONY: test-build
test-build: $(TEST_BIN)

$(TEST_BIN): cpp/test_strategy.cpp $(STRATEGY_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo ""
	@echo "Test built successfully!"
	@echo "Run './$(TEST_BIN)' to run unit tests"
	@echo ""

# Test the strategy with demo
.PHONY: test
test: $(DEMO_BIN)
	@if [ ! -f $(STRATEGY_FILE) ]; then \
		echo "Error: $(STRATEGY_FILE) not found"; \
		echo "Run 'make solve' first to generate the strategy file"; \
		exit 1; \
	fi
	./$(DEMO_BIN)

# Run unit tests
.PHONY: test-unit
test-unit: $(TEST_BIN)
	@if [ ! -f $(STRATEGY_FILE) ]; then \
		echo "Error: $(STRATEGY_FILE) not found"; \
		echo "Run 'make solve' first to generate the strategy file"; \
		exit 1; \
	fi
	./$(TEST_BIN)

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(SOLVER_BIN) $(DEMO_BIN) $(GUIDE_BIN) $(TEST_BIN) $(RATIONAL_BIN)
	rm -f $(SOLVER_OBJ) $(STRATEGY_OBJ)
	@echo "Cleaned build artifacts"

# Clean everything including strategy file
.PHONY: distclean
distclean: clean
	rm -f $(STRATEGY_FILE)
	@echo "Cleaned all files including strategy data"

# Help target
.PHONY: help
help:
	@echo "Yahtzee Optimal Strategy Solver - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make solver    - Build the solver executable"
	@echo "  make demo      - Build the strategy query demo"
	@echo "  make guide     - Build the interactive guide"
	@echo "  make rational  - Build the exact rational EV solver"
	@echo "  make rational-solve - Compute the exact EV table (~5 s)"
	@echo "  make all       - Build solver and demo"
	@echo "  make solve     - Run solver to generate strategy file (~5-30 min)"
	@echo "  make test      - Run demo to test the strategy"
	@echo "  make clean     - Remove executables and object files"
	@echo "  make distclean - Remove everything including strategy file"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Typical workflow:"
	@echo "  1. make solver       # Build the solver"
	@echo "  2. make solve        # Generate strategy (takes time!)"
	@echo "  3. make demo         # Build the demo"
	@echo "  4. make test         # Test the strategy"
	@echo "  5. make guide        # Build interactive guide (optional)"
	@echo ""
	@echo "Or simply:"
	@echo "  make all && make solve && make test"
	@echo ""

# Show current status
.PHONY: status
status:
	@echo "Build Status:"
	@echo "-------------"
	@if [ -f $(SOLVER_BIN) ]; then \
		echo "✓ Solver built: $(SOLVER_BIN)"; \
	else \
		echo "✗ Solver not built (run 'make solver')"; \
	fi
	@if [ -f $(DEMO_BIN) ]; then \
		echo "✓ Demo built: $(DEMO_BIN)"; \
	else \
		echo "✗ Demo not built (run 'make demo')"; \
	fi
	@if [ -f $(GUIDE_BIN) ]; then \
		echo "✓ Guide built: $(GUIDE_BIN)"; \
	else \
		echo "✗ Guide not built (run 'make guide')"; \
	fi
	@echo ""
	@echo "Data Files:"
	@echo "-----------"
	@if [ -f $(STRATEGY_FILE) ]; then \
		echo "✓ Strategy file exists: $(STRATEGY_FILE)"; \
		echo "  Size: $$(du -h $(STRATEGY_FILE) | cut -f1)"; \
	else \
		echo "✗ Strategy file missing (run 'make solve')"; \
	fi
	@echo ""