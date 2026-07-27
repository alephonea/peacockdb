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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
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


// ---------------------------------------------------------------------------
// EXECUTE-TIME BENCHMARK  (task: cuDF execute-only walltime, for comparison against the
// DuckDB VM numbers)
//
// SINGLE SOURCE OF OPERATORS: this times the SAME `execute` closure the enclosing test
// verifies against its golden. The test calls execute() once and compares; the benchmark
// calls it `runs` times and reports the second-smallest. There is no second copy of any
// query plan — a benchmark that timed different code than we verified would be worthless.
//
// TIMED REGION: t0 at closure entry (the input cudf::tables are already resident in VRAM
// from the enclosing test's one-time load); execute() runs every operator; then a FULL
// DEVICE SYNC before t1. cuDF is async on a stream, so timing without the sync would measure
// kernel LAUNCH, not execution — an absurdly small, wrong number. cudaDeviceSynchronize()
// (not a single-stream sync) is deliberate: the vector queries drive both the default cuDF
// stream and a raft handle stream, and this drains every stream in one call.
// EXCLUDED from the region: the parquet load, the golden comparison, and any device->host
// copy — none of those happen inside execute().
//
// SECOND-SMALLEST OF 6: run 1 pays RMM pool growth + kernel first-touch/JIT; 2nd-min
// discards that warm-up without letting a single slow outlier dominate — identical protocol
// to benchmarks/duckdb_minimal.sh, for symmetry.
//
// RAII / VRAM: each iteration's `result` and every intermediate inside execute() free at the
// closure's scope end, so device memory does not grow across the 6 runs. A cheap leak guard
// samples free memory before and after the loop and warns on material growth.
//
// PEAK VRAM IS NOT MEASURED HERE — deliberately. It is measured by the fixture's note_peak()
// (cudaMemGetInfo high-water, sampled DURING execute at the heavy-allocation points) and
// printed once per test by TearDown. That is the transient peak — the moment the widest
// intermediates are all live. An earlier version of this helper sampled cudaMemGetInfo AFTER
// execute() returned, which measured STEADY-STATE resident memory (inputs + result, with the
// transients already freed) — a different and much smaller number that wrongly contradicted
// the note_peak peaks. So peak reporting stays with note_peak, one definition, and this
// helper reports timing only.
//
// Guarded by PEACOCK_BENCHMARK: unset, this is a no-op and the tests run exactly as before.
inline bool benchmark_enabled() {
  const char* v = std::getenv("PEACOCK_BENCHMARK");
  return v && *v && std::string(v) != "0";
}

// execute: a callable returning any owning cuDF result (unique_ptr<table> or <scalar>); its
// return value is held until the sync so the produced data is real, then freed.
// load_ms / setup_ms: printed beside the execute time (setup_ms < 0 => omitted, for the
// plain queries that have no index-build phase).
template <typename F>
inline void benchmark_execute(const char* qtag, F&& execute, double load_ms,
                              double setup_ms = -1.0) {
  if (!benchmark_enabled()) return;
  const int runs = std::max(2, std::atoi(env_or("PEACOCK_BENCHMARK_RUNS", "6").c_str()));

  size_t total = 0, free_before = 0;
  cudaDeviceSynchronize();
  cudaMemGetInfo(&free_before, &total);

  std::vector<double> ms;
  ms.reserve(runs);
  for (int i = 0; i < runs; ++i) {
    cudaDeviceSynchronize();  // drain the prior iteration so its work never bleeds into t0
    const auto t0 = std::chrono::steady_clock::now();
    auto result = execute();
    cudaDeviceSynchronize();  // THE load-bearing sync — without it we time launch, not run
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    (void)result;  // freed here — RAII returns the intermediates to the pool
  }

  // leak guard (not a peak): free memory should return to roughly where it started once the
  // last result frees. A persistent drop means an intermediate is escaping the closure scope.
  cudaDeviceSynchronize();
  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total);
  if (free_before > free_after + (256UL << 20)) {  // >256 MiB not returned
    std::fprintf(stderr,
                 "[bench] %s WARNING: %.2f GiB of device memory not returned after the runs "
                 "— an intermediate may be leaking across iterations\n",
                 qtag, (free_before - free_after) / 1073741824.0);
  }

  std::sort(ms.begin(), ms.end());
  const double second_min = ms.size() > 1 ? ms[1] : ms[0];

  // machine-readable line the driver greps; all runs included for transparency. Peak VRAM is
  // the fixture's note_peak line, not here — see the comment above.
  std::string all;
  for (size_t i = 0; i < ms.size(); ++i) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%s%.3f", i ? "," : "", ms[i]);
    all += buf;
  }
  if (setup_ms >= 0.0) {
    std::fprintf(stderr, "[bench] %s execute_ms=%.3f load_ms=%.3f setup_ms=%.3f runs=%d all=[%s]\n",
                 qtag, second_min, load_ms, setup_ms, runs, all.c_str());
  } else {
    std::fprintf(stderr, "[bench] %s execute_ms=%.3f load_ms=%.3f runs=%d all=[%s]\n", qtag,
                 second_min, load_ms, runs, all.c_str());
  }
}



// GOLDEN CSV READER — RFC4180 quoting, and it has to be.
//
// This used to split on ',' with a comment saying "no quoted fields in these goldens".
// That was true until q10v, which selects c_comment and c_address: TPC-H comment text
// contains commas, so DuckDB quotes those fields and doubles any embedded quote. A
// comma-splitting reader turns one such row into 9 fields instead of 8 — which the
// field-count guard in compare_table_to_golden would catch as a malformed golden,
// blaming the generator for a defect in the reader.
//
// So: proper quoted-field parsing, and the file is consumed as a CHARACTER STREAM rather
// than line by line, because a quoted field may legally contain a newline. Reading line
// by line would split such a row in half and produce two malformed rows — the same class
// of silent misparse, one level down. TPC-H text does not currently contain newlines;
// this costs nothing and removes the assumption.
//
// Unquoted content parses byte-identically to the old splitter, so every existing golden
// is unaffected.
std::vector<std::vector<std::string>> read_csv_golden(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  std::vector<std::vector<std::string>> rows;
  std::vector<std::string> row;
  std::string cur;
  bool in_quotes = false, row_started = false;

  auto end_field = [&] { row.push_back(cur); cur.clear(); row_started = true; };
  auto end_row = [&] {
    end_field();
    rows.push_back(row);
    row.clear();
    row_started = false;
  };

  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (in_quotes) {
      if (c == '"') {
        // "" inside a quoted field is one literal quote; a lone " closes the field
        if (i + 1 < text.size() && text[i + 1] == '"') { cur += '"'; ++i; }
        else in_quotes = false;
      } else {
        cur += c;
      }
    } else if (c == '"' && cur.empty()) {
      // row_started here too, so a row that is a single empty quoted field ("") is a row
      // with one empty field, not a blank line to skip
      in_quotes = true;
      row_started = true;
    } else if (c == ',') {
      end_field();
    } else if (c == '\n') {
      // a blank line is skipped entirely rather than becoming a one-empty-field row
      if (row_started || !cur.empty()) end_row();
    } else if (c != '\r') {
      cur += c;
    }
  }
  // last line without a trailing newline
  if (row_started || !cur.empty()) end_row();
  return rows;
}

// pull one element back to the host. cudf::get_element returns a scalar, which keeps the
// exact fixed-point representation — no round-trip through double for the decimal columns.
//
// WIDTH: cuDF picks the NARROWEST decimal type a column fits — decimal(15,2) comes back as
// DECIMAL64, not DECIMAL128. Columns this suite builds itself (revenue, value) are forced
// to DECIMAL128, but a decimal read straight from parquet and carried through as a key
// (q10v's c_acctbal) stays DECIMAL64. A cast to decimal128 alone therefore returns null and
// every such value reads as 0 — a silent zero, exactly the failure mode the count and int
// accessors were already hardened against. So accept all three widths, same as int64_at
// reads INT16/32/64. The unscaled value is widened to __int128 either way, so the
// comparison downstream is identical.
inline Decimal decimal_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  Decimal d;
  if (auto* f128 = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(s.get())) {
    d.unscaled = static_cast<__int128>(f128->value());
    d.scale = -f128->type().scale();
    return d;
  }
  if (auto* f64 = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal64>*>(s.get())) {
    d.unscaled = static_cast<__int128>(f64->value());
    d.scale = -f64->type().scale();
    return d;
  }
  if (auto* f32 = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal32>*>(s.get())) {
    d.unscaled = static_cast<__int128>(f32->value());
    d.scale = -f32->type().scale();
    return d;
  }
  ADD_FAILURE() << "expected a decimal column (32/64/128), got type id "
                << static_cast<int>(col.type().id());
  return Decimal{};
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
