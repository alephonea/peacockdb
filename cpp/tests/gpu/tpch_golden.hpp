// tpch_golden.hpp — shared scaffolding for the bare-cuDF TPC-H tests.
//
// Extracted from test_tpch.cpp so a NEW query file (test_tpchv.cpp) inherits the same
// golden-comparison semantics rather than growing a second, subtly different copy. The
// per-column ColSpec is the point: every column names how it is compared and the default
// is EXACT, so a new test cannot accidentally become tolerant everywhere.
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace peacock_test {


// ---------------------------------------------------------------------------
// paths — overridable so the same binary runs on the GPU host and a dev box
// ---------------------------------------------------------------------------
inline std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : fallback;
}

inline std::string data_dir() {
  return env_or("PEACOCK_TPCH_SF40_DIR", "/home/info/peacock-datasets/testdata/tpch.sf40");
}
inline std::string golden_dir() {
  return env_or("PEACOCK_TPCH_GOLDEN_DIR", "testdata/goldens/tpch.sf40");
}

inline bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

// ---------------------------------------------------------------------------
// exact decimal comparison
//
// The golden is DuckDB's DECIMAL printed with full scale ("5252371252.6144"), never
// through float formatting, so no precision is lost at the file boundary. We parse it
// into an unscaled __int128 + a scale and compare VALUES, not representations: DuckDB
// and cuDF each pick their own result scale by their own promotion rules (both exact),
// so 1234.5600 and 1234.56 are the same answer and must compare equal. Rule: rescale the
// coarser of the two up to the finer scale (exact, only ever multiplies by 10^k), then
// compare the integers. No double is involved anywhere on this path.
// ---------------------------------------------------------------------------
struct Decimal {
  __int128 unscaled = 0;
  int scale = 0;  // digits after the point; value = unscaled / 10^scale
  bool ok = true; // false => the golden field was not a plain decimal (see parse_decimal)
};

inline Decimal parse_decimal(const std::string& raw) {
  std::string s;
  for (char c : raw) {
    if (!std::isspace(static_cast<unsigned char>(c))) s += c;
  }
  bool neg = !s.empty() && s[0] == '-';
  if (neg || (!s.empty() && s[0] == '+')) s.erase(0, 1);
  Decimal d;
  bool seen_point = false;
  bool any_digit = false;
  // STRICT: reject anything that is not a plain decimal instead of skipping it.
  // This used to `continue` past unexpected characters, which meant a golden field in
  // scientific notation (say "1.23e9", if a future DuckDB changed its formatting) would
  // silently parse as 1239 and be compared as a WRONG value rather than failing. That is
  // the same silent-wrong-value shape that bit us three times from the cuDF side (INT32
  // count, TIMESTAMP_DAYS, INT16 year) — this one would have been in our own parser.
  for (char c : s) {
    if (c == '.') {
      if (seen_point) { d.ok = false; return d; }   // two decimal points
      seen_point = true;
      continue;
    }
    if (c < '0' || c > '9') { d.ok = false; return d; }
    any_digit = true;
    d.unscaled = d.unscaled * 10 + (c - '0');
    if (seen_point) ++d.scale;
  }
  if (!any_digit) { d.ok = false; return d; }
  if (neg) d.unscaled = -d.unscaled;
  return d;
}

inline __int128 pow10_i128(int n) {
  __int128 r = 1;
  for (int i = 0; i < n; ++i) r *= 10;
  return r;
}

// true iff the two decimals denote the SAME exact value, regardless of scale
inline bool decimal_values_equal(Decimal a, Decimal b) {
  if (a.scale < b.scale) {
    a.unscaled *= pow10_i128(b.scale - a.scale);
  } else if (b.scale < a.scale) {
    b.unscaled *= pow10_i128(a.scale - b.scale);
  }
  return a.unscaled == b.unscaled;
}

inline std::string to_string_i128(__int128 v) {
  if (v == 0) return "0";
  bool neg = v < 0;
  unsigned __int128 u = neg ? static_cast<unsigned __int128>(-v) : static_cast<unsigned __int128>(v);
  std::string s;
  while (u) {
    s += static_cast<char>('0' + static_cast<int>(u % 10));
    u /= 10;
  }
  if (neg) s += '-';
  return std::string(s.rbegin(), s.rend());
}

inline std::string decimal_to_string(Decimal d) {
  std::string digits = to_string_i128(d.unscaled < 0 ? -d.unscaled : d.unscaled);
  std::string sign = d.unscaled < 0 ? "-" : "";
  if (d.scale == 0) return sign + digits;
  while (static_cast<int>(digits.size()) <= d.scale) digits.insert(digits.begin(), '0');
  digits.insert(digits.end() - d.scale, '.');
  return sign + digits;
}

// read a single-value golden csv (one field, no header)
inline std::string read_single_value_golden(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  std::getline(f, line);
  return line;
}

// days since the unix epoch for a calendar date — cudf's timestamp_D is exactly this
inline int32_t days_since_epoch(int y, unsigned m, unsigned d) {
  using namespace std::chrono;
  return static_cast<int32_t>(
      sys_days{year{y} / month{m} / day{d}}.time_since_epoch().count());
}

// ---------------------------------------------------------------------------
// fixture: skip loudly when the dataset is absent, and report GPU memory
// ---------------------------------------------------------------------------
class TpchSf40 : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto dir = data_dir();
    if (!file_exists(dir + "/lineitem.parquet")) {
      GTEST_SKIP() << "\n"
                   << "*** SKIPPING TPC-H sf40 GPU tests — DATASET NOT PRESENT ***\n"
                   << "    expected: " << dir << "/lineitem.parquet\n"
                   << "    This test reads an EXISTING sf40 dataset in place; it never\n"
                   << "    downloads or regenerates one. Point PEACOCK_TPCH_SF40_DIR at a\n"
                   << "    generated tpch.sf40 to run it.\n"
                   << "    NOTHING WAS VERIFIED by this test binary.\n";
    }
    cudaMemGetInfo(&free_before_, &total_);
  }

  void TearDown() override {
    size_t free_after = 0, total = 0;
    cudaMemGetInfo(&free_after, &total);
    // peak is sampled, not tracked: report the high-water mark we observed during the
    // query via note_peak(), which is what the CI cost estimate needs.
    std::fprintf(stderr,
                 "[gpu-mem] device total %.1f GiB; peak used by this test %.2f GiB\n",
                 total_ / 1073741824.0, peak_used_ / 1073741824.0);
  }

  // call after the memory-heaviest step
  void note_peak() {
    size_t free_now = 0, total = 0;
    cudaMemGetInfo(&free_now, &total);
    size_t used = free_before_ > free_now ? free_before_ - free_now : 0;
    if (used > peak_used_) peak_used_ = used;
  }

  size_t free_before_ = 0, total_ = 0, peak_used_ = 0;
};



// one golden row = the raw CSV fields, split on commas (no quoted fields in these goldens)
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

std::vector<std::vector<std::string>> read_csv_golden(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    rows.push_back(split_csv(line));
  }
  return rows;
}

// pull one element back to the host. cudf::get_element returns a scalar, which keeps the
// exact fixed-point representation — no round-trip through double for the decimal columns.
inline Decimal decimal_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  auto* fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(s.get());
  EXPECT_NE(fp, nullptr) << "expected a decimal128 column";
  if (!fp) return Decimal{};
  Decimal d;
  d.unscaled = static_cast<__int128>(fp->value());
  d.scale = -fp->type().scale();
  return d;
}

inline double double_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  auto* n = dynamic_cast<cudf::numeric_scalar<double>*>(s.get());
  EXPECT_NE(n, nullptr) << "expected a float64 column";
  return n ? n->value() : 0.0;
}

// WIDTH MISMATCH BETWEEN THE ENGINES — read this before touching it.
// cuDF's groupby COUNT returns INT32; DuckDB's count(*) is BIGINT. This is a silent
// wrong-answer generator: when this helper assumed int64, the dynamic_cast returned
// null and EVERY count read as 0 — while the four decimal sums were already matching
// perfectly. It failed LOUDLY only because the counts are compared at all; a test that
// checked sums alone would have been green with the counts entirely broken.
// OVERFLOW HEADROOM IS THIN: at sf40 the largest group count is 116,640,476, within ~18x
// of INT32_MAX (2,147,483,647). Around sf700 a TPC-H Q1 group would overflow int32
// outright. Anyone running a materially larger scale factor must check what cuDF returns
// here rather than assuming this still holds.
// So: accept either width and compare the VALUE, never the representation.
inline int64_t int64_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  if (auto* n64 = dynamic_cast<cudf::numeric_scalar<int64_t>*>(s.get())) return n64->value();
  if (auto* n32 = dynamic_cast<cudf::numeric_scalar<int32_t>*>(s.get())) return n32->value();
  // INT16 too: cudf::datetime::extract_year returns INT16, so Q8's o_year lands here.
  if (auto* n16 = dynamic_cast<cudf::numeric_scalar<int16_t>*>(s.get())) return n16->value();
  ADD_FAILURE() << "expected an integer column, got type id "
                << static_cast<int>(col.type().id());
  return 0;
}

// o_orderdate stays a TIMESTAMP_DAYS column through the whole chain (type id 12), so it
// needs its own reader — int64_at cannot see it, and casting the column to an integer just
// to compare would throw away the type information the plan is supposed to preserve.
inline int64_t days_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  auto* ts = dynamic_cast<cudf::timestamp_scalar<cudf::timestamp_D>*>(s.get());
  EXPECT_NE(ts, nullptr) << "expected a timestamp_D column";
  return ts ? static_cast<int64_t>(ts->value().time_since_epoch().count()) : 0;
}

inline std::string string_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  auto* str = dynamic_cast<cudf::string_scalar*>(s.get());
  EXPECT_NE(str, nullptr) << "expected a string column";
  return str ? str->to_string() : std::string{};
}

// ---------------------------------------------------------------------------
// TABLE-VS-GOLDEN COMPARISON
//
// Q1 and Q3 both did: read golden rows -> assert row count -> assert field count ->
// compare column by column with MIXED exact/tolerant semantics -> on mismatch print both
// values, the row and the column. That is ONE helper driven by a per-column spec, so a new
// query declares its column types and inherits every assertion.
//
// THE SEMANTICS ARE THE POINT, not the deduplication. Each column names how it is compared
// and the default is EXACT. A helper that quietly made everything tolerant would destroy
// the property this suite exists to hold: 5 of Q1's 8 aggregates, and every column of Q3
// and Q6, are compared EXACTLY. Only a value that genuinely cannot be exact (an AVG, a
// division) gets a tolerance, and only one justified by measurement.
// ---------------------------------------------------------------------------
enum class Cmp {
  ExactString,    // string column vs the raw golden field
  ExactDecimal,   // fixed-point, scale-normalized before comparing
  ExactInt,       // integer of either width (see int64_at)
  ExactDate,      // TIMESTAMP_DAYS vs a YYYY-MM-DD golden field
  TolerantDouble  // float64, RELATIVE tolerance — only where a float is unavoidable
};

struct ColSpec {
  const char* name;      // used in failure messages
  Cmp cmp;
  double rel_tol = 0.0;  // read only for TolerantDouble
};

// Compare `table` against `golden`. Returns the worst relative error seen across
// TolerantDouble columns, so a caller can report how much headroom the tolerance had.
inline double compare_table_to_golden(cudf::table_view const& table,
                               std::vector<std::vector<std::string>> const& golden,
                               std::vector<ColSpec> const& spec,
                               const char* qtag) {
  double worst_rel = 0.0;

  // ROW COUNT — a result with more or fewer rows than the golden must fail here, not
  // silently compare whichever rows it happens to have.
  EXPECT_EQ(static_cast<int>(golden.size()), table.num_rows())
      << qtag << ": row count differs from golden";
  if (static_cast<int>(golden.size()) != table.num_rows()) return worst_rel;
  EXPECT_EQ(static_cast<int>(spec.size()), table.num_columns())
      << qtag << ": spec describes a different column count than the result has";

  for (int r = 0; r < table.num_rows(); ++r) {
    const auto& g = golden[r];
    // FIELD COUNT — this guard caught a real off-by-one (Q1 asserted 11 fields for a
    // 10-column result). Keep it: a short golden row would otherwise be compared against
    // empty strings and could pass.
    // (ADD_FAILURE + continue rather than ASSERT_*: this function returns a value, so a
    // void-returning ASSERT cannot be used here. Same guarantee — the run fails — while
    // skipping the malformed row instead of reading past its end.)
    if (static_cast<int>(g.size()) != static_cast<int>(spec.size())) {
      ADD_FAILURE() << qtag << ": golden row " << r << " has " << g.size()
                    << " fields, expected " << spec.size();
      continue;
    }

    for (size_t c = 0; c < spec.size(); ++c) {
      const auto& sp = spec[c];
      const auto col = table.column(static_cast<cudf::size_type>(c));
      const std::string where =
          std::string(qtag) + " row " + std::to_string(r) + " col '" + sp.name + "'";

      switch (sp.cmp) {
        case Cmp::ExactString:
          EXPECT_EQ(string_at(col, r), g[c]) << where;
          break;
        case Cmp::ExactDecimal: {
          const Decimal got = decimal_at(col, r);
          const Decimal want = parse_decimal(g[c]);
          if (!want.ok) {
            ADD_FAILURE() << where << ": golden field '" << g[c]
                          << "' is not a plain decimal — refusing to guess at its value";
            break;
          }
          EXPECT_TRUE(decimal_values_equal(got, want))
              << where << " EXACT decimal mismatch\n"
              << "  cudf   : " << decimal_to_string(got) << "\n"
              << "  duckdb : " << decimal_to_string(want);
          break;
        }
        case Cmp::ExactInt: {
          const int64_t got = int64_at(col, r);
          // strtoll returns 0 on garbage, so check the WHOLE field was consumed — an
          // unparseable golden must fail, not silently compare against zero.
          char* endp = nullptr;
          const int64_t want = std::strtoll(g[c].c_str(), &endp, 10);
          if (endp == g[c].c_str() || (endp && *endp != '\0')) {
            ADD_FAILURE() << where << ": golden field '" << g[c] << "' is not an integer";
            break;
          }
          EXPECT_EQ(got, want) << where << " EXACT int mismatch\n"
                               << "  cudf   : " << got << "\n"
                               << "  duckdb : " << want;
          break;
        }
        case Cmp::ExactDate: {
          // parse the golden YYYY-MM-DD to days-since-epoch; the COLUMN stays a timestamp,
          // so no string formatting and no cast of the column is involved
          const int y = std::atoi(g[c].substr(0, 4).c_str());
          const int mo = std::atoi(g[c].substr(5, 2).c_str());
          const int dy = std::atoi(g[c].substr(8, 2).c_str());
          const int64_t got = days_at(col, r);
          const int64_t want = days_since_epoch(y, mo, dy);
          EXPECT_EQ(got, want) << where << " EXACT date mismatch\n"
                               << "  cudf   : " << got << " days\n"
                               << "  duckdb : " << want << " days (" << g[c] << ")";
          break;
        }
        case Cmp::TolerantDouble: {
          const double got = double_at(col, r);
          // same full-consumption check; strtod accepts scientific notation, which is
          // legitimate for a double column, but must still consume the entire field
          char* dendp = nullptr;
          const double want = std::strtod(g[c].c_str(), &dendp);
          if (dendp == g[c].c_str() || (dendp && *dendp != '\0')) {
            ADD_FAILURE() << where << ": golden field '" << g[c] << "' is not a number";
            break;
          }
          const double rel =
              want != 0.0 ? std::abs(got - want) / std::abs(want) : std::abs(got);
          if (rel > worst_rel) worst_rel = rel;
          EXPECT_LE(rel, sp.rel_tol)
              << where << " outside relative tolerance " << sp.rel_tol << "\n"
              << "  cudf   : " << got << "\n"
              << "  duckdb : " << want << "\n"
              << "  rel err: " << rel;
          break;
        }
      }
    }
  }
  return worst_rel;
}


}  // namespace peacock_test
