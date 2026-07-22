// test_tpch.cpp — TPC-H query plans hand-written in BARE cuDF, checked against DuckDB.
//
// No Rust, no DataFusion, no SQL: every query below is an explicit sequence of libcudf
// calls. The point is to exercise the operator chain peacockdb's executor has to produce,
// against real sf40 data, with an independent oracle (DuckDB over the SAME parquet files)
// deciding whether the answer is right.
//
// TWO VISIBLY SEPARATE PHASES, mirroring peacockdb's load-then-execute model:
//   PHASE 1 (load)     read the needed COLUMNS from parquet into cudf tables. Column
//                      selection is expected; ROW filtering is deliberately NOT pushed
//                      into the reader — no filters, no row-group pruning, no num_rows.
//   PHASE 2 (execute)  the operator chain runs over those in-memory columns.
// If a predicate ever migrates into phase 1 this test stops testing what it exists for.
//
// DATA: read-only, in place, from an EXISTING tpch.sf40 on the GPU host. Never downloaded,
// never regenerated, never written to. Absent data => SKIP loudly (see the fixture): a
// green run that silently tested nothing is the one outcome worse than a red one.
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/groupby.hpp>
// cuDF MOVED THIS HEADER between 25.02 and 26.02: cudf/join.hpp -> cudf/join/join.hpp.
// CI builds both legs, so the source has to satisfy both. Verified against the two local
// installs — join.hpp is the ONLY header this file includes that moved, and the
// inner_join signature is byte-identical in both versions (same parameters, same
// null_equality/stream/mr defaults), so this is purely a path change, not an API change.
#if __has_include(<cudf/join/join.hpp>)
#  include <cudf/join/join.hpp>   // cudf >= 26.02
#else
#  include <cudf/join.hpp>        // cudf 25.02
#endif
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// paths — overridable so the same binary runs on the GPU host and a dev box
// ---------------------------------------------------------------------------
std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : fallback;
}

std::string data_dir() {
  return env_or("PEACOCK_TPCH_SF40_DIR", "/home/info/peacock-datasets/testdata/tpch.sf40");
}
std::string golden_dir() {
  return env_or("PEACOCK_TPCH_GOLDEN_DIR", "testdata/goldens/tpch.sf40");
}

bool file_exists(const std::string& p) {
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
};

Decimal parse_decimal(const std::string& raw) {
  std::string s;
  for (char c : raw) {
    if (!std::isspace(static_cast<unsigned char>(c))) s += c;
  }
  bool neg = !s.empty() && s[0] == '-';
  if (neg || (!s.empty() && s[0] == '+')) s.erase(0, 1);
  Decimal d;
  bool seen_point = false;
  for (char c : s) {
    if (c == '.') {
      seen_point = true;
      continue;
    }
    if (c < '0' || c > '9') continue;
    d.unscaled = d.unscaled * 10 + (c - '0');
    if (seen_point) ++d.scale;
  }
  if (neg) d.unscaled = -d.unscaled;
  return d;
}

__int128 pow10_i128(int n) {
  __int128 r = 1;
  for (int i = 0; i < n; ++i) r *= 10;
  return r;
}

// true iff the two decimals denote the SAME exact value, regardless of scale
bool decimal_values_equal(Decimal a, Decimal b) {
  if (a.scale < b.scale) {
    a.unscaled *= pow10_i128(b.scale - a.scale);
  } else if (b.scale < a.scale) {
    b.unscaled *= pow10_i128(a.scale - b.scale);
  }
  return a.unscaled == b.unscaled;
}

std::string to_string_i128(__int128 v) {
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

std::string decimal_to_string(Decimal d) {
  std::string digits = to_string_i128(d.unscaled < 0 ? -d.unscaled : d.unscaled);
  std::string sign = d.unscaled < 0 ? "-" : "";
  if (d.scale == 0) return sign + digits;
  while (static_cast<int>(digits.size()) <= d.scale) digits.insert(digits.begin(), '0');
  digits.insert(digits.end() - d.scale, '.');
  return sign + digits;
}

// read a single-value golden csv (one field, no header)
std::string read_single_value_golden(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  std::getline(f, line);
  return line;
}

// days since the unix epoch for a calendar date — cudf's timestamp_D is exactly this
int32_t days_since_epoch(int y, unsigned m, unsigned d) {
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

}  // namespace

// ===========================================================================
// Q6  — revenue from a discounted-order filter
//
//   SELECT sum(l_extendedprice * l_discount)
//   FROM lineitem
//   WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
//     AND l_discount BETWEEN 0.05 AND 0.07
//     AND l_quantity < 24
//
// operator chain: scan -> filter(3 predicates) -> project(mul) -> reduce(sum)
//
// REPRESENTATION (exact, no tolerance anywhere):
//   l_quantity / l_extendedprice / l_discount are decimal128(15,2) in the parquet, which
//   cuDF decodes to DECIMAL64 scale -2 (INT64 physical, precision 15). We cast to
//   DECIMAL128 at the same scale purely for headroom, then stay fixed-point end to end:
//   the predicate constants are decimal scalars, so the boundaries 0.05 / 0.07 / 24 have
//   NO float representation error and cannot include or exclude a row by rounding.
//   The product lands at scale -4 and the sum accumulates there.
//   Headroom: worst-case product ~1e7 x 7 = 7e7 scale-4 units, ~1e8 qualifying rows =>
//   ~1e16, against a DECIMAL128 ceiling of ~1.7e38. Cannot overflow.
// ===========================================================================
TEST_F(TpchSf40, Q6ExactDecimal) {
  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path = golden_dir() + "/duckdb_q6.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path
      << " (regenerate with testdata/gen_duckdb_goldens.sh --sf 40)";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD ----------------
  // Columns only. No filter, no row-group selection, no num_rows: every predicate below
  // is an operator in phase 2, so the reader hands us the full 240M rows of these 4
  // columns and the GPU does the rest.
  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{std::vector<std::string>{lineitem_path}})
                  .columns({"l_quantity", "l_extendedprice", "l_discount", "l_shipdate"})
                  .build();
  auto loaded = cudf::io::read_parquet(opts);
  auto tbl = loaded.tbl->view();
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();

  ASSERT_EQ(tbl.num_columns(), 4);
  std::fprintf(stderr, "[q6] loaded %ld rows x 4 cols\n", static_cast<long>(tbl.num_rows()));

  auto quantity_raw = tbl.column(0);
  auto extprice_raw = tbl.column(1);
  auto discount_raw = tbl.column(2);
  auto shipdate = tbl.column(3);

  // Assert the decoded types rather than trusting the mapping — if a future cudf decodes
  // these differently, the arithmetic below changes meaning and we want to know.
  ASSERT_EQ(shipdate.type().id(), cudf::type_id::TIMESTAMP_DAYS);
  for (auto c : {quantity_raw, extprice_raw, discount_raw}) {
    ASSERT_TRUE(c.type().id() == cudf::type_id::DECIMAL64 ||
                c.type().id() == cudf::type_id::DECIMAL128)
        << "expected fixed-point money columns, got type id "
        << static_cast<int>(c.type().id());
    ASSERT_EQ(c.type().scale(), -2);
  }

  // widen to DECIMAL128 (same scale => exact, value-preserving) for accumulation headroom
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  auto quantity = cudf::cast(quantity_raw, dec128_s2);
  auto extprice = cudf::cast(extprice_raw, dec128_s2);
  auto discount = cudf::cast(discount_raw, dec128_s2);

  // ---------------- PHASE 2: EXECUTE ----------------
  // filter: three predicates, AND-ed
  auto lo_date = cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(1994, 1, 1)}}, true);
  auto hi_date = cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(1995, 1, 1)}}, true);
  // decimal scalars at scale -2: 0.05 -> 5, 0.07 -> 7, 24 -> 2400. Exact boundaries.
  auto disc_lo = cudf::fixed_point_scalar<numeric::decimal128>(5, numeric::scale_type{-2});
  auto disc_hi = cudf::fixed_point_scalar<numeric::decimal128>(7, numeric::scale_type{-2});
  auto qty_hi = cudf::fixed_point_scalar<numeric::decimal128>(2400, numeric::scale_type{-2});

  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  auto m1 = cudf::binary_operation(shipdate, lo_date, cudf::binary_operator::GREATER_EQUAL, boolean);
  auto m2 = cudf::binary_operation(shipdate, hi_date, cudf::binary_operator::LESS, boolean);
  auto m3 = cudf::binary_operation(discount->view(), disc_lo, cudf::binary_operator::GREATER_EQUAL, boolean);
  auto m4 = cudf::binary_operation(discount->view(), disc_hi, cudf::binary_operator::LESS_EQUAL, boolean);
  auto m5 = cudf::binary_operation(quantity->view(), qty_hi, cudf::binary_operator::LESS, boolean);

  auto mask = cudf::binary_operation(m1->view(), m2->view(), cudf::binary_operator::LOGICAL_AND, boolean);
  mask = cudf::binary_operation(mask->view(), m3->view(), cudf::binary_operator::LOGICAL_AND, boolean);
  mask = cudf::binary_operation(mask->view(), m4->view(), cudf::binary_operator::LOGICAL_AND, boolean);
  mask = cudf::binary_operation(mask->view(), m5->view(), cudf::binary_operator::LOGICAL_AND, boolean);
  note_peak();

  auto kept = cudf::apply_boolean_mask(
      cudf::table_view{{extprice->view(), discount->view()}}, mask->view());
  std::fprintf(stderr, "[q6] rows surviving filter: %ld\n",
               static_cast<long>(kept->num_rows()));
  ASSERT_GT(kept->num_rows(), 0) << "filter kept no rows — the predicates or the data are wrong";
  note_peak();

  // project: extendedprice * discount  (scale -2 * scale -2 -> scale -4)
  auto revenue_col = cudf::binary_operation(kept->get_column(0).view(),
                                            kept->get_column(1).view(),
                                            cudf::binary_operator::MUL,
                                            cudf::data_type{cudf::type_id::DECIMAL128, -4});
  note_peak();

  // reduce: exact fixed-point sum
  auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto result = cudf::reduce(revenue_col->view(), *sum_agg,
                             cudf::data_type{cudf::type_id::DECIMAL128, -4});
  auto* fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(result.get());
  ASSERT_NE(fp, nullptr) << "sum did not come back as a decimal128 scalar";
  ASSERT_TRUE(fp->is_valid());

  Decimal got;
  got.unscaled = static_cast<__int128>(fp->value());
  got.scale = -fp->type().scale();  // cudf scale is negative: -4 => 4 digits after point

  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (exact, value-normalized) ----------------
  const auto golden_raw = read_single_value_golden(golden_path);
  ASSERT_FALSE(golden_raw.empty()) << "empty golden: " << golden_path;
  const Decimal want = parse_decimal(golden_raw);

  std::fprintf(stderr, "[q6] cudf  = %s (scale %d)\n", decimal_to_string(got).c_str(), got.scale);
  std::fprintf(stderr, "[q6] duckdb= %s (scale %d)\n", decimal_to_string(want).c_str(), want.scale);

  EXPECT_TRUE(decimal_values_equal(got, want))
      << "Q6 mismatch (EXACT decimal comparison, no tolerance)\n"
      << "  cudf   : " << decimal_to_string(got) << "\n"
      << "  duckdb : " << decimal_to_string(want) << "\n"
      << "  golden : " << golden_path;

  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q6] load %ld ms, execute %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_done), ms(t0, t_done));
}

// ===========================================================================
// Q1 — pricing summary report
//
//   SELECT l_returnflag, l_linestatus,
//          sum(l_quantity), sum(l_extendedprice),
//          sum(l_extendedprice*(1-l_discount)),
//          sum(l_extendedprice*(1-l_discount)*(1+l_tax)),
//          avg(l_quantity), avg(l_extendedprice), avg(l_discount), count(*)
//   FROM lineitem WHERE l_shipdate <= DATE '1998-09-02'
//   GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus
//
// operator chain: scan -> filter -> project(2 derived cols) -> groupby(8 aggs) -> sort
//
// REPRESENTATION IS PER COLUMN — a blanket tolerance would hide a real error in the
// columns that CAN be exact:
//   sum_qty, sum_base_price, sum_disc_price, sum_charge : DECIMAL128, compared EXACTLY
//                       (scales -2, -2, -4, -6; DuckDB picks its own scales, and the
//                        comparison normalizes scale before comparing values)
//   count_order       : INT64, compared EXACTLY
//   avg_qty, avg_price, avg_disc : DOUBLE on BOTH sides, RELATIVE TOLERANCE 1e-9.
//                       Forced, not chosen: DuckDB's AVG over a DECIMAL returns DOUBLE,
//                       and cuDF's MEAN likewise produces a floating result, so the two
//                       sums are accumulated in different orders over ~236M values. 1e-9
//                       is ~4 orders of magnitude above the ~1e-13 relative drift double
//                       accumulation actually produces at this size, and ~7 orders below
//                       the smallest error that would indicate a real bug (a wrong row
//                       set moves an average by percent, not by 1e-9).
// ===========================================================================
namespace {

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
Decimal decimal_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  auto* fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(s.get());
  EXPECT_NE(fp, nullptr) << "expected a decimal128 column";
  if (!fp) return Decimal{};
  Decimal d;
  d.unscaled = static_cast<__int128>(fp->value());
  d.scale = -fp->type().scale();
  return d;
}

double double_at(cudf::column_view const& col, cudf::size_type i) {
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
int64_t int64_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  if (auto* n64 = dynamic_cast<cudf::numeric_scalar<int64_t>*>(s.get())) return n64->value();
  if (auto* n32 = dynamic_cast<cudf::numeric_scalar<int32_t>*>(s.get())) return n32->value();
  ADD_FAILURE() << "expected an integer column, got type id "
                << static_cast<int>(col.type().id());
  return 0;
}

// o_orderdate stays a TIMESTAMP_DAYS column through the whole chain (type id 12), so it
// needs its own reader — int64_at cannot see it, and casting the column to an integer just
// to compare would throw away the type information the plan is supposed to preserve.
int64_t days_at(cudf::column_view const& col, cudf::size_type i) {
  auto s = cudf::get_element(col, i);
  auto* ts = dynamic_cast<cudf::timestamp_scalar<cudf::timestamp_D>*>(s.get());
  EXPECT_NE(ts, nullptr) << "expected a timestamp_D column";
  return ts ? static_cast<int64_t>(ts->value().time_since_epoch().count()) : 0;
}

std::string string_at(cudf::column_view const& col, cudf::size_type i) {
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
double compare_table_to_golden(cudf::table_view const& table,
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
          EXPECT_TRUE(decimal_values_equal(got, want))
              << where << " EXACT decimal mismatch\n"
              << "  cudf   : " << decimal_to_string(got) << "\n"
              << "  duckdb : " << decimal_to_string(want);
          break;
        }
        case Cmp::ExactInt: {
          const int64_t got = int64_at(col, r);
          const int64_t want = std::strtoll(g[c].c_str(), nullptr, 10);
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
          const double want = std::strtod(g[c].c_str(), nullptr);
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

}  // namespace


TEST_F(TpchSf40, Q1GroupByAggregates) {
  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path = golden_dir() + "/duckdb_q1.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path
      << " (regenerate with testdata/gen_duckdb_goldens.sh --sf 40)";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD ----------------
  // Columns only — no predicate pushdown, no row-group selection. The l_shipdate filter
  // is an operator in phase 2, below.
  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{std::vector<std::string>{lineitem_path}})
                  .columns({"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice",
                            "l_discount", "l_tax", "l_shipdate"})
                  .build();
  auto loaded = cudf::io::read_parquet(opts);
  auto tbl = loaded.tbl->view();
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  ASSERT_EQ(tbl.num_columns(), 7);
  std::fprintf(stderr, "[q1] loaded %ld rows x 7 cols\n", static_cast<long>(tbl.num_rows()));

  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  auto returnflag = tbl.column(0);
  auto linestatus = tbl.column(1);
  auto quantity = cudf::cast(tbl.column(2), dec128_s2);
  auto extprice = cudf::cast(tbl.column(3), dec128_s2);
  auto discount = cudf::cast(tbl.column(4), dec128_s2);
  auto tax = cudf::cast(tbl.column(5), dec128_s2);
  auto shipdate = tbl.column(6);

  // ---------------- PHASE 2: EXECUTE ----------------
  // filter: l_shipdate <= 1998-09-02
  auto cutoff = cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(1998, 9, 2)}}, true);
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  auto mask = cudf::binary_operation(shipdate, cutoff, cudf::binary_operator::LESS_EQUAL, boolean);
  auto kept = cudf::apply_boolean_mask(
      cudf::table_view{{returnflag, linestatus, quantity->view(), extprice->view(),
                        discount->view(), tax->view()}},
      mask->view());
  note_peak();
  std::fprintf(stderr, "[q1] rows surviving filter: %ld\n",
               static_cast<long>(kept->num_rows()));
  ASSERT_GT(kept->num_rows(), 0);

  auto k_flag = kept->get_column(0).view();
  auto k_status = kept->get_column(1).view();
  auto k_qty = kept->get_column(2).view();
  auto k_price = kept->get_column(3).view();
  auto k_disc = kept->get_column(4).view();
  auto k_tax = kept->get_column(5).view();

  // project: disc_price = extendedprice * (1 - discount)          scale -4
  //          charge     = disc_price     * (1 + tax)              scale -6
  // The literals 1 are DECIMAL scalars at scale -2 (value 100), so these stay exact.
  auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
  auto one_minus_disc = cudf::binary_operation(one_s2, k_disc, cudf::binary_operator::SUB, dec128_s2);
  auto one_plus_tax = cudf::binary_operation(one_s2, k_tax, cudf::binary_operator::ADD, dec128_s2);
  auto disc_price = cudf::binary_operation(k_price, one_minus_disc->view(),
                                           cudf::binary_operator::MUL,
                                           cudf::data_type{cudf::type_id::DECIMAL128, -4});
  auto charge = cudf::binary_operation(disc_price->view(), one_plus_tax->view(),
                                       cudf::binary_operator::MUL,
                                       cudf::data_type{cudf::type_id::DECIMAL128, -6});
  note_peak();

  // AVG: DuckDB's AVG over DECIMAL returns DOUBLE, so the cudf side must also be double —
  // this is the one place a floating representation is unavoidable, and the only place a
  // tolerance is applied.
  const auto f64 = cudf::data_type{cudf::type_id::FLOAT64};
  auto qty_f = cudf::cast(k_qty, f64);
  auto price_f = cudf::cast(k_price, f64);
  auto disc_f = cudf::cast(k_disc, f64);

  // groupby (l_returnflag, l_linestatus) -> 8 aggregates
  cudf::groupby::groupby gb(cudf::table_view{{k_flag, k_status}});
  std::vector<cudf::groupby::aggregation_request> reqs;
  auto add = [&](cudf::column_view v, std::unique_ptr<cudf::groupby_aggregation> a) {
    cudf::groupby::aggregation_request r;
    r.values = v;
    r.aggregations.push_back(std::move(a));
    reqs.push_back(std::move(r));
  };
  add(k_qty, cudf::make_sum_aggregation<cudf::groupby_aggregation>());          // 0 sum_qty
  add(k_price, cudf::make_sum_aggregation<cudf::groupby_aggregation>());        // 1 sum_base_price
  add(disc_price->view(), cudf::make_sum_aggregation<cudf::groupby_aggregation>());  // 2
  add(charge->view(), cudf::make_sum_aggregation<cudf::groupby_aggregation>());      // 3
  add(qty_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());      // 4 avg_qty
  add(price_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());    // 5 avg_price
  add(disc_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());     // 6 avg_disc
  add(k_qty, cudf::make_count_aggregation<cudf::groupby_aggregation>());             // 7 count

  auto [keys_tbl, agg_results] = gb.aggregate(reqs);
  note_peak();

  // sort: cudf's groupby does not promise a group order, so sort by the two group keys —
  // the same ORDER BY the golden uses. The keys are unique per group (4 groups), so the
  // ordering is TOTAL: no tie-breaking is involved and the rows cannot shift relative to
  // the golden.
  auto keys_view = keys_tbl->view();
  std::vector<std::unique_ptr<cudf::column>> agg_cols;
  for (auto& r : agg_results) agg_cols.push_back(std::move(r.results[0]));
  std::vector<cudf::column_view> all_views{keys_view.column(0), keys_view.column(1)};
  for (auto& c : agg_cols) all_views.push_back(c->view());

  auto order = cudf::sorted_order(cudf::table_view{{keys_view.column(0), keys_view.column(1)}},
                                  {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                  {cudf::null_order::AFTER, cudf::null_order::AFTER});
  auto sorted = cudf::gather(cudf::table_view{all_views}, order->view());
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE ----------------
  // Per-column semantics: the four SUMs and the COUNT are EXACT; only the three AVGs are
  // toleranced, because DuckDB's AVG over DECIMAL returns DOUBLE and cuDF's MEAN likewise.
  constexpr double kAvgRelTol = 1e-9;  // see the header comment for why 1e-9
  const std::vector<ColSpec> kQ1Spec = {
      {"l_returnflag", Cmp::ExactString},
      {"l_linestatus", Cmp::ExactString},
      {"sum_qty", Cmp::ExactDecimal},
      {"sum_base_price", Cmp::ExactDecimal},
      {"sum_disc_price", Cmp::ExactDecimal},
      {"sum_charge", Cmp::ExactDecimal},
      {"avg_qty", Cmp::TolerantDouble, kAvgRelTol},
      {"avg_price", Cmp::TolerantDouble, kAvgRelTol},
      {"avg_disc", Cmp::TolerantDouble, kAvgRelTol},
      {"count_order", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  const double worst_rel =
      compare_table_to_golden(sorted->view(), golden, kQ1Spec, "q1");

  std::fprintf(stderr, "[q1] worst AVG relative error %.3e (tolerance %.1e)\n",
               worst_rel, kAvgRelTol);
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q1] load %ld ms, execute %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_done), ms(t0, t_done));
}

// ===========================================================================
// Q3 — shipping priority (the first query needing JOINS)
//
//   SELECT l_orderkey, sum(l_extendedprice*(1-l_discount)) AS revenue,
//          o_orderdate, o_shippriority
//   FROM customer, orders, lineitem
//   WHERE c_mktsegment='BUILDING' AND c_custkey=o_custkey AND l_orderkey=o_orderkey
//     AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
//   GROUP BY l_orderkey, o_orderdate, o_shippriority
//   ORDER BY revenue DESC, o_orderdate, l_orderkey LIMIT 10
//
// operator chain: scan x3 -> filter x3 -> join -> join -> project -> groupby -> sort -> limit
//
// REPRESENTATION: revenue is DECIMAL128 scale -4 throughout and compared EXACTLY — the
// (1 - l_discount) literal is a decimal scalar, so no float enters. o_orderdate is
// compared as a date string, l_orderkey / o_shippriority as integers. NO TOLERANCE
// ANYWHERE in Q3 (unlike Q1, which needs one only because AVG forces a double).
//
// TIE-BREAK: the spec's ORDER BY revenue DESC, o_orderdate is not a total order, so
// l_orderkey is appended here AND in the golden generator. Without it a tie at the LIMIT
// 10 boundary could return different rows run to run — flaky, not wrong, which is worse.
// ===========================================================================
TEST_F(TpchSf40, Q3JoinsGroupByTopN) {
  const auto golden_path = golden_dir() + "/duckdb_q3.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path
      << " (regenerate with testdata/gen_duckdb_goldens.sh --sf 40)";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD (all three tables, columns only) ----------------
  auto read_cols = [](const std::string& path, std::vector<std::string> cols) {
    auto o = cudf::io::parquet_reader_options::builder(
                 cudf::io::source_info{std::vector<std::string>{path}})
                 .columns(std::move(cols))
                 .build();
    return cudf::io::read_parquet(o);
  };
  auto cust_in = read_cols(data_dir() + "/customer.parquet", {"c_custkey", "c_mktsegment"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet",
                          {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"});
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  std::fprintf(stderr, "[q3] loaded customer %ld, orders %ld, lineitem %ld rows\n",
               static_cast<long>(cust_in.tbl->num_rows()),
               static_cast<long>(ord_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()));

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};

  // filter customer: c_mktsegment = 'BUILDING'
  auto seg = cudf::string_scalar(std::string("BUILDING"));
  auto cust_mask = cudf::binary_operation(cust_in.tbl->view().column(1), seg,
                                          cudf::binary_operator::EQUAL, boolean);
  auto cust_f = cudf::apply_boolean_mask(
      cudf::table_view{{cust_in.tbl->view().column(0)}}, cust_mask->view());

  // filter orders: o_orderdate < 1995-03-15
  auto d1995 = cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(1995, 3, 15)}}, true);
  auto ord_mask = cudf::binary_operation(ord_in.tbl->view().column(2), d1995,
                                         cudf::binary_operator::LESS, boolean);
  auto ord_f = cudf::apply_boolean_mask(ord_in.tbl->view(), ord_mask->view());

  // filter lineitem: l_shipdate > 1995-03-15
  auto line_mask = cudf::binary_operation(line_in.tbl->view().column(3), d1995,
                                          cudf::binary_operator::GREATER, boolean);
  auto line_f = cudf::apply_boolean_mask(line_in.tbl->view(), line_mask->view());
  note_peak();
  std::fprintf(stderr, "[q3] after filters: customer %ld, orders %ld, lineitem %ld\n",
               static_cast<long>(cust_f->num_rows()), static_cast<long>(ord_f->num_rows()),
               static_cast<long>(line_f->num_rows()));
  ASSERT_GT(cust_f->num_rows(), 0);
  ASSERT_GT(ord_f->num_rows(), 0);
  ASSERT_GT(line_f->num_rows(), 0);

  // helper: build a column_view over a join gather map
  // cudf::inner_join returns a PAIR OF unique_ptr<device_uvector<size_type>> gather maps,
  // not tables: the join tells you which row indices pair up, and you gather the columns
  // you actually want. That is why only the needed columns are gathered below.
  auto map_view = [](std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
    return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                             static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
  };

  // join 1: customer.c_custkey = orders.o_custkey
  auto [c_map, o_map] = cudf::inner_join(cudf::table_view{{cust_f->get_column(0).view()}},
                                         cudf::table_view{{ord_f->get_column(1).view()}});
  // only orders columns are needed downstream (c_custkey is consumed by the join)
  auto co = cudf::gather(cudf::table_view{{ord_f->get_column(0).view(),    // o_orderkey
                                           ord_f->get_column(2).view(),    // o_orderdate
                                           ord_f->get_column(3).view()}},  // o_shippriority
                         map_view(o_map));
  note_peak();
  std::fprintf(stderr, "[q3] customer|X|orders -> %ld rows\n", static_cast<long>(co->num_rows()));

  // join 2: (customer|X|orders).o_orderkey = lineitem.l_orderkey
  auto [co_map, l_map] = cudf::inner_join(cudf::table_view{{co->get_column(0).view()}},
                                          cudf::table_view{{line_f->get_column(0).view()}});
  auto co_side = cudf::gather(cudf::table_view{{co->get_column(0).view(),    // o_orderkey
                                                co->get_column(1).view(),    // o_orderdate
                                                co->get_column(2).view()}},  // o_shippriority
                              map_view(co_map));
  auto l_side = cudf::gather(cudf::table_view{{line_f->get_column(1).view(),    // extendedprice
                                               line_f->get_column(2).view()}},  // discount
                             map_view(l_map));
  note_peak();
  std::fprintf(stderr, "[q3] |X|lineitem -> %ld rows\n", static_cast<long>(co_side->num_rows()));
  ASSERT_GT(co_side->num_rows(), 0) << "join produced no rows";

  // project: revenue = l_extendedprice * (1 - l_discount), exact decimal
  auto price = cudf::cast(l_side->get_column(0).view(), dec128_s2);
  auto disc = cudf::cast(l_side->get_column(1).view(), dec128_s2);
  auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
  auto one_minus_disc = cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB, dec128_s2);
  auto revenue = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                        cudf::binary_operator::MUL, dec128_s4);
  note_peak();

  // groupby (l_orderkey, o_orderdate, o_shippriority) -> sum(revenue)
  cudf::groupby::groupby gb(cudf::table_view{{co_side->get_column(0).view(),
                                              co_side->get_column(1).view(),
                                              co_side->get_column(2).view()}});
  std::vector<cudf::groupby::aggregation_request> reqs;
  {
    cudf::groupby::aggregation_request r;
    r.values = revenue->view();
    r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(std::move(r));
  }
  auto [gkeys, gaggs] = gb.aggregate(reqs);
  note_peak();
  std::fprintf(stderr, "[q3] groups: %ld\n", static_cast<long>(gkeys->num_rows()));

  // sort: revenue DESC, o_orderdate ASC, l_orderkey ASC (total order — see header)
  auto gk = gkeys->view();
  auto rev_col = std::move(gaggs[0].results[0]);
  auto order = cudf::sorted_order(
      cudf::table_view{{rev_col->view(), gk.column(1), gk.column(0)}},
      {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING},
      {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER});
  auto sorted = cudf::gather(
      cudf::table_view{{gk.column(0), rev_col->view(), gk.column(1), gk.column(2)}},
      order->view());

  // limit 10
  auto top = cudf::slice(sorted->view(), {0, 10})[0];
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (exact, NO tolerance anywhere in Q3) ----------------
  // revenue stays DECIMAL128 scale -4 end to end; o_orderdate stays TIMESTAMP_DAYS and is
  // compared as days, so the column is never cast or formatted just to be checked.
  const std::vector<ColSpec> kQ3Spec = {
      {"l_orderkey", Cmp::ExactInt},
      {"revenue", Cmp::ExactDecimal},
      {"o_orderdate", Cmp::ExactDate},
      {"o_shippriority", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_EQ(static_cast<int>(golden.size()), 10) << "golden should hold 10 rows";
  compare_table_to_golden(top, golden, kQ3Spec, "q3");

  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q3] load %ld ms, execute %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_done), ms(t0, t_done));
}

// Same entry point as the other gtest binaries here (the conda cudf ships no gtest_main).
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
