/**
 * Yahtzee Exact PMF Solver
 *
 * Computes, for every reachable board state, the exact probability
 * distribution of the score still to come under the EV-optimal policy,
 * using rational arithmetic throughout (no floating point in the DP).
 *
 * POLICY: decided by exact EVs (yahtzee_exact_turn.h), so near-ties are never
 * misjudged by rounding. Exact ties, which leave the EV unchanged but can
 * change the distribution, are broken by fixed rules: scoring takes the first
 * best option in rule order; keeps prefer more dice kept, then lowest id.
 *
 * WHY IT IS EXACT AND CHEAP:
 * ==========================
 * With k turns left, every probability is an integer over D_k = 6^(15k).
 * Once the turn's policy is fixed, the probability of each final roll is an
 * integer over 6^15 (< 2^39), found by pushing integer weights through the
 * keep lattice (1,260 additions per reroll). Final rolls that lead to the same
 * (points, next board) are merged, and then
 *
 *   PMF_k[s] = sum over (points r, next n, weight w) of w * PMF_{k-1}[n][s - r]
 *
 * which is one big-integer multiply-add by a 64-bit weight per (pair, score).
 *
 * MEMORY: successors are always in the previous layer, so only that layer's
 * exact distributions are needed. Each is stored over its score range only,
 * and each probability with only the limbs it needs. By default the two live
 * layers stay in RAM (peak ~4 GB). With --low-memory, each finished layer is
 * written to a temporary file (<output>.layer0/1) and read back through a
 * read-only memory map, so resident memory stays around one layer (~2 GB)
 * and the OS can drop those clean pages under pressure. Float rows go to a
 * temporary file (<output>.rows) and are copied into the output in board
 * order at the end, so the output is byte-for-byte reproducible.
 *
 * CHECKS: for every state, the probabilities must sum to exactly 1 and the
 * distribution's mean must equal the exact EV. Any mismatch aborts.
 *
 * OUTPUT:
 * =======
 * yahtzee_pmf_v2.bin (read by python/yahtzee_2player_env.py):
 *   16-byte header {magic "PMFY", version 2, num_states, dense_size = 1576}
 *   num_states index entries {uint32 offset, int16 lo, int16 width}: the
 *     state's probabilities for scores lo .. lo+width-1 are floats
 *     offset .. offset+width-1 of the data section (width 0: not reachable)
 *   data section: floats, each the exact probability rounded to float
 * yahtzee_pmf_exact_start.txt: the exact distribution at the start of the
 *   game, as numerators over 6^195.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "yahtzee_exact_turn.h"

const int DENSE_SIZE = MAX_FUTURE_SCORE + 1;  // 1576 scores: 0..1575

// Output index entry
struct PMFIndexEntry {
  uint32_t offset;  // first float in the data section
  int16_t lo;       // lowest score with nonzero probability
  int16_t width;    // number of scores stored (0 = not computed)
};
static_assert(sizeof(PMFIndexEntry) == 8, "index entry must be 8 bytes");

class ExactPMFSolver {
 public:
  explicit ExactPMFSolver(bool low_memory);
  ~ExactPMFSolver();
  void Solve(const std::string& filename);
  void WriteExactStart(const std::string& filename) const;

 private:
  ScoringRules rules_;
  KeepLattice lattice_;
  u64 init_count_[NUM_DICE_COMBOS];  // ordered ways to roll each outcome

  std::vector<u64> ev_;           // EV numerators, EV_LIMBS per board
  std::vector<int16_t> lo_, hi_;  // score range per board

  // Packed distribution layout: ceil(width / 8) words holding one limb count
  // per score (one byte each), then each score's limbs, high zero limbs
  // dropped (a zero probability takes no limbs).
  static int LensWords(int width) { return (width + 7) / 8; }

  // Exact layers: in RAM (mem_), or in temporary files (low-memory mode)
  const bool low_memory_;
  std::vector<std::vector<u64>> mem_;  // packed distribution per board
  std::vector<u64> off_;               // low-memory: word offset in layer file
  std::string layer_path_[2];
  int cur_fd_ = -1;                    // layer file being written
  std::atomic<size_t> cur_words_{0};
  const u64* prev_ = nullptr;          // memory-mapped previous layer
  size_t prev_bytes_ = 0;
  int prev_slot_ = 0;

  // Float rows are appended to a temporary file as states finish (in thread
  // order); WriteOutput then copies them in board-index order, so the output
  // file is byte-for-byte reproducible without holding rows in memory.
  int rows_fd_ = -1;
  std::string rows_path_;
  std::vector<PMFIndexEntry> index_;  // offsets into the rows file until output
  std::atomic<size_t> row_floats_{0};
  void WriteOutput(const std::string& filename);

  // Packed distribution of a board in the previous layer
  const u64* Packed(size_t idx) const {
    return low_memory_ ? prev_ + off_[idx] : mem_[idx].data();
  }

  // Stores a finished distribution (exact, packed) and its float row
  void Emit(size_t idx, int lo, const std::vector<u64>& packed,
            const float* row, int width);

  // Layer bookkeeping (low-memory mode: file per layer)
  void StartLayer(int slot);
  void FinishLayer(int slot, const std::vector<size_t>& older_states);

  // Probabilities of the five-dice outcomes after one reroll decision:
  // out[d] = sum over a[d1] * P(d | keep[d1]), scaled to a total of 6^5 * sum(a)
  void PushWeights(const u64* a, const int16_t* keep, u64* out) const;

  template <int L>
  void SolveLayer(const std::vector<size_t>& states, int turns_left);

  static void WriteAt(int fd, size_t offset, const void* data, size_t bytes);

  // Reachable boards with exactly turns_left turns to go
  std::vector<size_t> Layer(int turns_left) const;
};

ExactPMFSolver::ExactPMFSolver(bool low_memory)
    : ev_(static_cast<size_t>(BOARD_STATE_SIZE) * EV_LIMBS, 0),
      lo_(BOARD_STATE_SIZE, 0),
      hi_(BOARD_STATE_SIZE, -1),
      low_memory_(low_memory),
      index_(BOARD_STATE_SIZE, PMFIndexEntry{0, 0, 0}) {
  if (low_memory_) {
    off_.assign(BOARD_STATE_SIZE, 0);
  } else {
    mem_.resize(BOARD_STATE_SIZE);
  }
  // Rolling all five dice: a single unit kept as the empty subset, pushed
  // through the lattice, gives the ordered counts of each outcome
  u64 one[NUM_DICE_COMBOS] = {0};
  int16_t keep_empty[NUM_DICE_COMBOS] = {0};
  one[0] = 1;
  PushWeights(one, keep_empty, init_count_);
}

ExactPMFSolver::~ExactPMFSolver() {
  if (prev_) {
    munmap(const_cast<u64*>(prev_), prev_bytes_);
    unlink(layer_path_[prev_slot_].c_str());
  }
}

void ExactPMFSolver::PushWeights(const u64* a, const int16_t* keep, u64* out) const {
  static const u64 kPow6[6] = {1, 6, 36, 216, 1296, 7776};
  u64 M[KeepLattice::kNumSubsets] = {0};
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
    if (a[d]) M[keep[d]] += a[d] * kPow6[lattice_.Size(keep[d])];
  }
  // Ids ascend by size, so each subset is complete before it is pushed
  for (int i = 0; i < KeepLattice::kFirst5; ++i) {
    if (!M[i]) continue;
    for (int child : lattice_.Children(i)) M[child] += M[i];
  }
  for (int d = 0; d < NUM_DICE_COMBOS; ++d) out[d] = M[KeepLattice::kFirst5 + d];
}

std::vector<size_t> ExactPMFSolver::Layer(int turns_left) const {
  std::vector<size_t> states;
  for (int m = 0; m < NUM_BOARD_MASKS; ++m) {
    for (int y = 0; y < NUM_Y_STATES; ++y) {
      if (TurnsLeft(m, y) != turns_left) continue;
      for (int u = 0; u < MAX_UPPER_SCORE; ++u) {
        if (rules_.Reachable(m, u)) states.push_back(BoardStateIndex(m, u, y));
      }
    }
  }
  return states;
}

void ExactPMFSolver::WriteAt(int fd, size_t offset, const void* data, size_t bytes) {
  const char* p = static_cast<const char*>(data);
  while (bytes > 0) {
    ssize_t n = pwrite(fd, p, bytes, static_cast<off_t>(offset));
    if (n < 0) {
      if (errno == EINTR) continue;
      std::perror("FATAL: pwrite failed");
      std::exit(1);
    }
    p += n;
    offset += n;
    bytes -= n;
  }
}

void ExactPMFSolver::Emit(size_t idx, int lo, const std::vector<u64>& packed,
                          const float* row, int width) {
  // Exact distribution, for the next layer
  if (low_memory_) {
    size_t pos = cur_words_.fetch_add(packed.size());
    WriteAt(cur_fd_, pos * sizeof(u64), packed.data(), packed.size() * sizeof(u64));
    off_[idx] = pos;
  } else {
    mem_[idx] = packed;
  }
  // Float probabilities, to the rows file
  size_t pos = row_floats_.fetch_add(width);
  WriteAt(rows_fd_, pos * sizeof(float), row, width * sizeof(float));
  index_[idx] = {static_cast<uint32_t>(pos), static_cast<int16_t>(lo),
                 static_cast<int16_t>(width)};
  lo_[idx] = static_cast<int16_t>(lo);
  hi_[idx] = static_cast<int16_t>(lo + width - 1);
}

void ExactPMFSolver::WriteOutput(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::perror("FATAL: could not create output file");
    std::exit(1);
  }
  const size_t data_offset = 16 + sizeof(PMFIndexEntry) * index_.size();
  const size_t rows_bytes = row_floats_ * sizeof(float);
  void* map = mmap(nullptr, rows_bytes, PROT_READ, MAP_SHARED, rows_fd_, 0);
  if (map == MAP_FAILED) {
    std::perror("FATAL: mmap failed");
    std::exit(1);
  }
  const float* rows = static_cast<const float*>(map);
  const size_t kChunk = 8 << 20;  // floats per write
  std::vector<float> buf;
  buf.reserve(kChunk + DENSE_SIZE);
  size_t out_pos = 0, buf_start = 0;
  for (size_t idx = 0; idx < index_.size(); ++idx) {
    PMFIndexEntry& e = index_[idx];
    if (e.width == 0) continue;
    buf.insert(buf.end(), rows + e.offset, rows + e.offset + e.width);
    e.offset = static_cast<uint32_t>(out_pos);
    out_pos += e.width;
    if (buf.size() >= kChunk) {
      WriteAt(fd, data_offset + buf_start * sizeof(float), buf.data(), buf.size() * sizeof(float));
      buf_start = out_pos;
      buf.clear();
    }
  }
  WriteAt(fd, data_offset + buf_start * sizeof(float), buf.data(), buf.size() * sizeof(float));

  uint32_t header[4] = {0x504D4659,  // "PMFY"
                        2, static_cast<uint32_t>(BOARD_STATE_SIZE),
                        static_cast<uint32_t>(DENSE_SIZE)};
  WriteAt(fd, 0, header, sizeof(header));
  WriteAt(fd, sizeof(header), index_.data(), index_.size() * sizeof(PMFIndexEntry));
  if (close(fd) != 0) {
    std::perror("FATAL: close failed");
    std::exit(1);
  }
  munmap(map, rows_bytes);
  close(rows_fd_);
  rows_fd_ = -1;
  unlink(rows_path_.c_str());
  std::cout << "Wrote " << filename << " ("
            << (data_offset + out_pos * sizeof(float)) / (1024 * 1024) << " MB)" << std::endl;
}

void ExactPMFSolver::StartLayer(int slot) {
  if (!low_memory_) return;
  cur_fd_ = open(layer_path_[slot].c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (cur_fd_ < 0) {
    std::perror("FATAL: could not create layer file");
    std::exit(1);
  }
  cur_words_ = 0;
}

void ExactPMFSolver::FinishLayer(int slot, const std::vector<size_t>& older_states) {
  if (!low_memory_) {
    // The layer before the one just finished is no longer needed
    for (size_t idx : older_states) std::vector<u64>().swap(mem_[idx]);
    return;
  }
  // Drop the previous layer's file, then map the layer just written
  if (prev_) {
    munmap(const_cast<u64*>(prev_), prev_bytes_);
    unlink(layer_path_[1 - slot].c_str());
  }
  prev_bytes_ = cur_words_ * sizeof(u64);
  void* p = mmap(nullptr, prev_bytes_, PROT_READ, MAP_SHARED, cur_fd_, 0);
  if (p == MAP_FAILED) {
    std::perror("FATAL: mmap failed");
    std::exit(1);
  }
  close(cur_fd_);
  cur_fd_ = -1;
  prev_ = static_cast<const u64*>(p);
  prev_slot_ = slot;
}

template <int L>
void ExactPMFSolver::SolveLayer(const std::vector<size_t>& states, int turns_left) {
  ExactTurnSolver<L> turn(rules_, lattice_, turns_left, ev_.data());

  // D_k = 6^(15k), exactly and as a long double
  UInt<L> denom;
  SetSmall(denom, 1);
  long double inv_denom = 1.0L;
  for (int t = 0; t < 15 * turns_left; ++t) {
    MulSmall(denom, denom, 6);
    inv_denom /= 6.0L;
  }
  const u64 kTurnTotal = 470184984576ULL;  // 6^15

  #pragma omp parallel
  {
    std::vector<ExactTurnScratch<L>> scratch(1);
    std::vector<UInt<L>> acc(DENSE_SIZE);
    std::vector<float> row(DENSE_SIZE);
    std::vector<int> lens_of(DENSE_SIZE);
    std::vector<u64> packed;
    TurnPolicy pol;
    u64 w2[NUM_DICE_COMBOS], w3[NUM_DICE_COMBOS];
    struct Pair { u64 key; u64 weight; };
    std::vector<Pair> pairs;
    pairs.reserve(NUM_DICE_COMBOS);

    #pragma omp for schedule(dynamic, 4)
    for (size_t si = 0; si < states.size(); ++si) {
      const size_t idx = states[si];
      const int y_state = static_cast<int>(idx % NUM_Y_STATES);
      const int upper = static_cast<int>((idx / NUM_Y_STATES) % MAX_UPPER_SCORE);
      const int mask = static_cast<int>(idx / (NUM_Y_STATES * MAX_UPPER_SCORE));

      // Exact EV and optimal policy
      UInt<L> ev;
      turn.Solve(mask, upper, y_state, scratch[0], ev, &pol);
      Store(&ev_[EV_LIMBS * idx], ev, EV_LIMBS);

      // Probability of each final roll (integers over 6^15)
      PushWeights(init_count_, pol.keep1, w2);
      PushWeights(w2, pol.keep2, w3);

      // Merge final rolls by (points, next board)
      pairs.clear();
      u64 total = 0;
      for (int d = 0; d < NUM_DICE_COMBOS; ++d) {
        if (!w3[d]) continue;
        total += w3[d];
        pairs.push_back({(static_cast<u64>(pol.reward[d]) << 32) | pol.next_idx[d], w3[d]});
      }
      if (total != kTurnTotal) {
        std::cerr << "FATAL: turn probabilities sum to " << total << std::endl;
        std::exit(1);
      }
      std::sort(pairs.begin(), pairs.end(),
                [](const Pair& a, const Pair& b) { return a.key < b.key; });
      size_t np = 0;
      for (size_t i = 0; i < pairs.size(); ++i) {
        if (np && pairs[np - 1].key == pairs[i].key) {
          pairs[np - 1].weight += pairs[i].weight;
        } else {
          pairs[np++] = pairs[i];
        }
      }
      pairs.resize(np);

      // Score range
      int lo = DENSE_SIZE, hi = -1;
      for (const Pair& p : pairs) {
        int r = static_cast<int>(p.key >> 32);
        size_t n = p.key & 0xffffffffULL;
        lo = std::min(lo, r + lo_[n]);
        hi = std::max(hi, r + hi_[n]);
      }
      const int width = hi - lo + 1;

      // PMF_k[s] = sum of w * PMF_{k-1}[n][s - r]
      for (int j = 0; j < width; ++j) SetZero(acc[j]);
      for (const Pair& p : pairs) {
        int r = static_cast<int>(p.key >> 32);
        size_t n = p.key & 0xffffffffULL;
        const int base = r + lo_[n] - lo;
        const int cnt = hi_[n] - lo_[n] + 1;
        const uint8_t* lens = reinterpret_cast<const uint8_t*>(Packed(n));
        const u64* src = Packed(n) + LensWords(cnt);
        for (int j = 0; j < cnt; ++j) {
          const int len = lens[j];
          if (len) MulAddSmall(acc[base + j], src, len, p.weight);
          src += len;
        }
      }

      // Exact checks: total probability 1, mean equals the exact EV
      UInt<L> sum, mean, t;
      SetZero(sum);
      SetZero(mean);
      for (int j = 0; j < width; ++j) {
        AddTo(sum, acc[j]);
        MulSmall(t, acc[j], static_cast<u64>(lo + j));
        AddTo(mean, t);
      }
      if (!Equal(sum, denom) || !Equal(mean, ev)) {
        std::cerr << "FATAL: exact check failed at board " << idx
                  << (Equal(sum, denom) ? " (mean != EV)" : " (sum != 1)") << std::endl;
        std::exit(1);
      }

      // Pack the exact distribution and round each probability to float
      size_t words = LensWords(width);
      for (int j = 0; j < width; ++j) {
        int len = L;
        while (len > 0 && acc[j].w[len - 1] == 0) --len;
        lens_of[j] = len;
        words += len;
      }
      packed.assign(words, 0);
      uint8_t* lens = reinterpret_cast<uint8_t*>(packed.data());
      u64* dst = packed.data() + LensWords(width);
      for (int j = 0; j < width; ++j) {
        const int len = lens_of[j];
        lens[j] = static_cast<uint8_t>(len);
        for (int i = 0; i < len; ++i) *dst++ = acc[j].w[i];
        row[j] = static_cast<float>(ToLongDouble(acc[j].w, len) * inv_denom);
      }
      Emit(idx, lo, packed, row.data(), width);
    }
  }
}

void ExactPMFSolver::Solve(const std::string& filename) {
  rows_path_ = filename + ".rows";
  rows_fd_ = open(rows_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (rows_fd_ < 0) {
    std::perror("FATAL: could not create rows file");
    std::exit(1);
  }
  layer_path_[0] = filename + ".layer0";
  layer_path_[1] = filename + ".layer1";

  std::cout << "Starting exact PMF computation (reverse induction"
            << (low_memory_ ? ", low-memory mode" : "") << ")...\n" << std::endl;

  // Terminal boards: nothing more to score
  std::vector<size_t> prev = Layer(0), older;
  StartLayer(0);
  {
    const std::vector<u64> certain = {1, 1};  // one score (0), one limb, numerator 1
    const float one = 1.0f;
    for (size_t idx : prev) Emit(idx, 0, certain, &one, 1);
  }
  FinishLayer(0, older);

  for (int turns_left = 1; turns_left <= 13; ++turns_left) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<size_t> states = Layer(turns_left);
    int limbs = EvLimbs(turns_left);
    std::cout << "Layer " << std::setw(2) << 13 - turns_left << " filled ("
              << std::setw(6) << states.size() << " states, " << limbs
              << " limbs)..." << std::flush;

    const int slot = turns_left % 2;
    StartLayer(slot);
    switch (limbs) {
      case 1: SolveLayer<1>(states, turns_left); break;
      case 2: SolveLayer<2>(states, turns_left); break;
      case 3: SolveLayer<3>(states, turns_left); break;
      case 4: SolveLayer<4>(states, turns_left); break;
      case 5: SolveLayer<5>(states, turns_left); break;
      case 6: SolveLayer<6>(states, turns_left); break;
      case 7: SolveLayer<7>(states, turns_left); break;
      case 8: SolveLayer<8>(states, turns_left); break;
      case 9: SolveLayer<9>(states, turns_left); break;
      default:
        std::cerr << "FATAL: unsupported limb count " << limbs << std::endl;
        std::exit(1);
    }
    size_t layer_bytes = cur_words_ * sizeof(u64);
    if (!low_memory_) {
      for (size_t idx : states) layer_bytes += mem_[idx].size() * sizeof(u64);
    }
    older.swap(prev);
    prev.swap(states);
    FinishLayer(slot, older);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    std::cout << " (" << ms << " ms, " << layer_bytes / (1024 * 1024)
              << " MB exact)" << std::endl;
  }

  std::cout << "\nPMF computation complete! Every state passed the exact checks"
            << " (sum = 1, mean = EV)." << std::endl;
  WriteOutput(filename);
}

void ExactPMFSolver::WriteExactStart(const std::string& filename) const {
  const size_t start = BoardStateIndex(0, 0, 0);
  const int lo = lo_[start], hi = hi_[start];
  const uint8_t* lens = reinterpret_cast<const uint8_t*>(Packed(start));
  const u64* num = Packed(start) + LensWords(hi - lo + 1);

  BigUInt den(1);
  for (int i = 0; i < 195; ++i) den.MulSmall(6);

  std::ofstream out(filename);
  out << "# Exact distribution of the final Yahtzee score under the EV-optimal policy\n"
      << "# (solo game, same rules as yahtzee_rules.h)\n"
      << "# P(score = s) = numerator / 6^195\n"
      << "# 6^195 = " << den.ToDecimal() << "\n"
      << "# score numerator\n";

  // Summary: exact mean, and a few probabilities in decimal
  long double inv = 1.0L;
  for (int i = 0; i < 195; ++i) inv /= 6.0L;
  long double tail[5] = {0, 0, 0, 0, 0};
  const int thresholds[5] = {200, 250, 300, 400, 500};
  int mode = lo;
  long double mode_p = -1;
  for (int s = lo; s <= hi; ++s) {
    const int len = lens[s - lo];
    long double p = ToLongDouble(num, len) * inv;
    for (int t = 0; t < 5; ++t) if (s >= thresholds[t]) tail[t] += p;
    if (p > mode_p) { mode_p = p; mode = s; }
    if (len) out << s << " " << BigUInt(num, len).ToDecimal() << "\n";
    num += len;
  }

  BigUInt mean(&ev_[EV_LIMBS * start], EV_LIMBS);
  int e2 = 195, e3 = 195;
  ReduceBy2And3(mean, e2, e3);
  std::cout << "\n=== Exact final-score distribution (start of game) ===" << std::endl;
  std::cout << "Mean (exact EV): " << DecimalOf(mean, e2, e3, 30) << std::endl;
  std::cout << "Score range: " << lo << " to " << hi << std::endl;
  std::cout << "Most likely score: " << mode << " (P = " << std::setprecision(6)
            << static_cast<double>(mode_p) << ")" << std::endl;
  for (int t = 0; t < 5; ++t) {
    std::cout << "P(score >= " << thresholds[t] << ") = " << std::setprecision(10)
              << static_cast<double>(tail[t]) << std::endl;
  }
  std::cout << "Exact distribution written to " << filename << std::endl;
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::cout << "=== Yahtzee Exact PMF Solver ===" << std::endl;
  std::cout << "Exact distribution of future score under the EV-optimal policy\n"
            << std::endl;

  // Usage: yahtzee_pmf_solver [--low-memory] [output.bin] [exact_start.txt]
  bool low_memory = false;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--low-memory") low_memory = true;
    else files.push_back(a);
  }
  std::string out_file = files.size() > 0 ? files[0] : "yahtzee_pmf_v2.bin";
  std::string exact_file = files.size() > 1 ? files[1] : "yahtzee_pmf_exact_start.txt";

  auto start = std::chrono::high_resolution_clock::now();
  ExactPMFSolver solver(low_memory);
  solver.Solve(out_file);
  solver.WriteExactStart(exact_file);
  auto secs = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  std::cout << "\nTotal time: " << secs << " seconds" << std::endl;
  return 0;
}
