#ifndef YAHTZEE_BIGINT_H
#define YAHTZEE_BIGINT_H

/**
 * Unsigned big integers for the exact (rational) solvers.
 *
 * UInt<L>: fixed width, L x 64-bit limbs, little-endian, no allocation. The
 * exact solvers keep every value as an integer numerator over a denominator
 * shared by a whole layer, so they only need add, multiply by a small
 * constant, and compare.
 *
 * BigUInt: arbitrary width, for reporting results (decimal output, reducing
 * a fraction by small primes).
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using u64 = uint64_t;
using u128 = unsigned __int128;

// -------------------------------------------------------------------------
// FIXED-WIDTH UNSIGNED INTEGER
// -------------------------------------------------------------------------

template <int L>
struct UInt {
  u64 w[L];
};

template <int L>
inline void SetZero(UInt<L>& a) {
  for (int i = 0; i < L; ++i) a.w[i] = 0;
}

template <int L>
inline void SetSmall(UInt<L>& a, u64 v) {
  SetZero(a);
  a.w[0] = v;
}

// a += b
template <int L>
inline void AddTo(UInt<L>& a, const UInt<L>& b) {
  u64 c = 0;
  for (int i = 0; i < L; ++i) {
    u128 s = static_cast<u128>(a.w[i]) + b.w[i] + c;
    a.w[i] = static_cast<u64>(s);
    c = static_cast<u64>(s >> 64);
  }
}

// r = a + b
template <int L>
inline void Add(UInt<L>& r, const UInt<L>& a, const UInt<L>& b) {
  u64 c = 0;
  for (int i = 0; i < L; ++i) {
    u128 s = static_cast<u128>(a.w[i]) + b.w[i] + c;
    r.w[i] = static_cast<u64>(s);
    c = static_cast<u64>(s >> 64);
  }
}

// r = a * m  (m small)
template <int L>
inline void MulSmall(UInt<L>& r, const UInt<L>& a, u64 m) {
  u64 carry = 0;
  for (int i = 0; i < L; ++i) {
    u128 p = static_cast<u128>(a.w[i]) * m + carry;
    r.w[i] = static_cast<u64>(p);
    carry = static_cast<u64>(p >> 64);
  }
}

// acc += src * m, where src has n <= L limbs
template <int L>
inline void MulAddSmall(UInt<L>& acc, const u64* src, int n, u64 m) {
  u64 carry = 0;
  int i = 0;
  for (; i < n; ++i) {
    u128 p = static_cast<u128>(src[i]) * m + acc.w[i] + carry;
    acc.w[i] = static_cast<u64>(p);
    carry = static_cast<u64>(p >> 64);
  }
  for (; i < L && carry; ++i) {
    u128 s = static_cast<u128>(acc.w[i]) + carry;
    acc.w[i] = static_cast<u64>(s);
    carry = static_cast<u64>(s >> 64);
  }
}

template <int L>
inline bool Less(const UInt<L>& a, const UInt<L>& b) {
  for (int i = L - 1; i >= 0; --i) {
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
  }
  return false;
}

template <int L>
inline bool Equal(const UInt<L>& a, const UInt<L>& b) {
  for (int i = 0; i < L; ++i) {
    if (a.w[i] != b.w[i]) return false;
  }
  return true;
}

// Load n <= L limbs, zero-extending
template <int L>
inline void Load(UInt<L>& a, const u64* src, int n = L) {
  std::memcpy(a.w, src, sizeof(u64) * n);
  for (int i = n; i < L; ++i) a.w[i] = 0;
}

// Store into n >= L limbs, zero-filling; or n < L limbs when the high limbs
// are known to be zero
template <int L>
inline void Store(u64* dst, const UInt<L>& a, int n) {
  int m = n < L ? n : L;
  std::memcpy(dst, a.w, sizeof(u64) * m);
  for (int i = m; i < n; ++i) dst[i] = 0;
}

// Limbs needed for numerators below 2^extra_bits * 6^(15k), with a spare bit
inline int LimbsFor(int turns_left, int extra_bits) {
  // 15 * log2(6) = 38.77443751...; round up generously
  int bits = extra_bits + (3877444 * turns_left + 99999) / 100000 + 1;
  int limbs = (bits + 63) / 64;
  return limbs < 1 ? 1 : limbs;
}

// Converts limbs to long double (about 64 significant bits)
inline long double ToLongDouble(const u64* w, int n) {
  long double r = 0;
  for (int i = n - 1; i >= 0; --i) {
    r = r * 18446744073709551616.0L + static_cast<long double>(w[i]);
  }
  return r;
}

// -------------------------------------------------------------------------
// ARBITRARY-WIDTH UNSIGNED INTEGER (reporting only)
// -------------------------------------------------------------------------

struct BigUInt {
  std::vector<u64> w;  // little-endian, no trailing zeros

  explicit BigUInt(u64 v = 0) {
    if (v) w.push_back(v);
  }
  BigUInt(const u64* limbs, int n) : w(limbs, limbs + n) { Trim(); }

  void Trim() {
    while (!w.empty() && w.back() == 0) w.pop_back();
  }
  bool IsZero() const { return w.empty(); }

  void MulSmall(u64 m) {
    u64 carry = 0;
    for (auto& x : w) {
      u128 p = static_cast<u128>(x) * m + carry;
      x = static_cast<u64>(p);
      carry = static_cast<u64>(p >> 64);
    }
    if (carry) w.push_back(carry);
    Trim();
  }

  // Divides in place, returns remainder
  u64 DivSmall(u64 d) {
    u128 rem = 0;
    for (int i = static_cast<int>(w.size()) - 1; i >= 0; --i) {
      u128 cur = (rem << 64) | w[i];
      w[i] = static_cast<u64>(cur / d);
      rem = cur % d;
    }
    Trim();
    return static_cast<u64>(rem);
  }

  u64 ModSmall(u64 d) const {
    u128 rem = 0;
    for (int i = static_cast<int>(w.size()) - 1; i >= 0; --i) {
      rem = ((rem << 64) | w[i]) % d;
    }
    return static_cast<u64>(rem);
  }

  std::string ToDecimal() const {
    if (IsZero()) return "0";
    BigUInt t = *this;
    std::vector<u64> chunks;
    const u64 kBase = 1000000000000000000ULL;  // 10^18
    while (!t.IsZero()) chunks.push_back(t.DivSmall(kBase));
    std::string s = std::to_string(chunks.back());
    for (int i = static_cast<int>(chunks.size()) - 2; i >= 0; --i) {
      std::string part = std::to_string(chunks[i]);
      s += std::string(18 - part.size(), '0') + part;
    }
    return s;
  }
};

// Reduces num / (2^e2 * 3^e3) in place by common factors of 2 and 3
inline void ReduceBy2And3(BigUInt& num, int& e2, int& e3) {
  while (e2 > 0 && !num.IsZero() && num.ModSmall(2) == 0) { num.DivSmall(2); --e2; }
  while (e3 > 0 && !num.IsZero() && num.ModSmall(3) == 0) { num.DivSmall(3); --e3; }
}

// floor(num / (2^e2 * 3^e3)) as a decimal string with `digits` decimals
inline std::string DecimalOf(const BigUInt& num, int e2, int e3, int digits) {
  BigUInt scaled = num;
  for (int i = 0; i < digits; ++i) scaled.MulSmall(10);
  for (int i = 0; i < e2; ++i) scaled.DivSmall(2);
  for (int i = 0; i < e3; ++i) scaled.DivSmall(3);
  std::string s = scaled.ToDecimal();
  if (static_cast<int>(s.size()) <= digits) s = std::string(digits + 1 - s.size(), '0') + s;
  return s.substr(0, s.size() - digits) + "." + s.substr(s.size() - digits);
}

#endif  // YAHTZEE_BIGINT_H
