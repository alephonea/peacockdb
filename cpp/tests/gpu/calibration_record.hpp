// The bare-cuDF half of the cost-model calibration record, and a record of an OLDER
// model than the one now measured.
//
// It was written for a fit over two engines on one set of axes: peacockdb pays a host
// prologue per node that bare cuDF has no analogue for, and the difference between the
// two sources was what `const_peacock` meant. That model is gone. The prologue is now
// measured where it happens — `peacock_host_us` per call on the real C++ path — so a
// second engine is no longer how it is separated from the device time.
//
// What follows from that, and is the reason for the two notes below: the two writers no
// longer share a format. `record.rs` writes per-CALL rows keyed by
// `(node, recipe step, call index)` with three time terms; this one writes per-cuDF-call
// rows carrying `source`, `label`, `cuda_bytes` and `wall_us`. They agree on nothing but
// the file being a TSV, and each has its OWN path variable so that a run with both
// enabled cannot append two formats to one file.
//
// Left standing rather than removed because its only includer, `test_tpch.cpp`, is a
// correctness test that predates this work by a month and belongs to it in no way: it
// checks hand-written bare-cuDF TPC-H against DuckDB goldens, and stripping 93 call
// sites out of it would be a large edit to a foreign test for no gain here. Unset the
// variable and a region is the bare call plus one pointer check.
//
// The category mapping is written by hand at each call site and is the part of this work
// worth reviewing. A cuDF call does not carry its role: the same `binary_operation` is a
// filter predicate in one place and a projection in another, and only the query's shape
// says which. A call left outside a region is invisible rather than wrong, so the
// coverage report below is what makes the mapping falsifiable — see QueryScope.
//
// Timing mirrors the node path: CUDA events per region, collected at scope end rather
// than read per call. Reading elapsed time needs a synchronize, and one per region would
// serialize a chain that is otherwise free to overlap — measuring the instrument's effect
// on the query instead of the query.
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "peacock/rmm_pool.hpp"
#include "plan_executor.h"

namespace peacock_test::calib {

// ---------------------------------------------------------------------------
// the categories, as `cost_model.conf` spells them
// ---------------------------------------------------------------------------
// Named constants rather than string literals at the sites: a mistyped category is then a
// compile error instead of a row binned into a category that does not exist, which the
// fit would report as a thinly-sampled category rather than as a mistake.
//
// Only the seven a bare-cuDF chain can reach are here. The other eight name peacockdb
// plan nodes with no cuDF call to point at — a repartition, a coalesce-batches, a window
// — and inventing a mapping for them would put rows in a category this source cannot
// actually measure.
inline constexpr const char* storage_read_bytes = "storage_read_bytes";
inline constexpr const char* cuda_filter_bytes = "cuda_filter_bytes";
inline constexpr const char* cuda_project_bytes = "cuda_project_bytes";
inline constexpr const char* cuda_hash_join_bytes = "cuda_hash_join_bytes";
inline constexpr const char* cuda_aggregate_bytes = "cuda_aggregate_bytes";
inline constexpr const char* cuda_sort_bytes = "cuda_sort_bytes";
inline constexpr const char* cuda_concat_bytes = "cuda_concat_bytes";

/// Same domain the node path pushes into (`cpp/src/node_session.cpp`), so one nsys export
/// reads both sources through one filter and libcudf's own ranges stay separable.
struct peacock_domain {
  static constexpr char const* name{"peacockdb"};
};
using scoped_range = ::nvtx3::scoped_range_in<peacock_domain>;

/// Set ⇒ rows are appended there. Unset ⇒ a region is the bare call plus one pointer
/// check, which is why no site needs an `#ifdef` and the tests run as they did.
///
/// Deliberately NOT `PEACOCK_RECORD_PATH`, which is the node source's. The two shared it
/// while they shared a format; they no longer do, and a shared variable would mean one
/// file holding two column layouts — appended silently from this side, refused from the
/// other, and in neither case the error it is.
inline const char* record_path() {
  static const char* p = std::getenv("PEACOCK_CUDF_RECORD_PATH");
  return (p && *p) ? p : nullptr;
}

// ---------------------------------------------------------------------------
// bytes: the ONE formula, borrowed rather than re-derived
// ---------------------------------------------------------------------------
// `peacock::logical_size_from_table` models the Rust byte rule and says in its own header
// that it exists to be shared with these tests. Counting bytes differently at the two
// ends would make a fit over both wrong by an unknown factor — the one error a
// calibration cannot notice it has made.
inline uint64_t bytes_of(const cudf::table_view& t) {
  return peacock::logical_size_from_table(t, peacock::varlen_content_bytes(t));
}
inline uint64_t bytes_of(const cudf::column_view& c) {
  return bytes_of(cudf::table_view{{c}});
}

struct Produced {
  uint64_t rows = 0;
  uint64_t bytes = 0;
};

/// What a call produced. Overloaded on the return type of every cuDF entry point the
/// queries use, so an unhandled one is a compile error rather than a zero — a new kind of
/// call cannot enter a chain and record nothing.
inline Produced produced(const std::unique_ptr<cudf::table>& t) {
  return {static_cast<uint64_t>(t->num_rows()), bytes_of(t->view())};
}
inline Produced produced(const std::unique_ptr<cudf::column>& c) {
  return {static_cast<uint64_t>(c->size()), bytes_of(c->view())};
}
inline Produced produced(const cudf::io::table_with_metadata& t) {
  return {static_cast<uint64_t>(t.tbl->num_rows()), bytes_of(t.tbl->view())};
}
/// A scalar is one value. Charged its type width so a reduction does not read as free,
/// while staying obviously not where the bytes are.
inline Produced produced(const std::unique_ptr<cudf::scalar>& s) {
  return {1, static_cast<uint64_t>(cudf::size_of(s->type()))};
}
/// A groupby returns the key table beside a per-request bundle of aggregate columns.
/// Both are output; charging only the keys would make an eight-aggregate groupby look the
/// same size as a one-aggregate one.
inline Produced produced(const std::pair<std::unique_ptr<cudf::table>,
                                         std::vector<cudf::groupby::aggregation_result>>& g) {
  std::vector<cudf::column_view> cols;
  for (auto const& c : g.first->view()) cols.push_back(c);
  for (auto const& r : g.second)
    for (auto const& c : r.results) cols.push_back(c->view());
  return {static_cast<uint64_t>(g.first->num_rows()), bytes_of(cudf::table_view{cols})};
}
/// `inner_join` returns two index vectors, not a joined table. Those vectors are what the
/// join materializes; assembling the table costs a later `gather`, which is its own
/// region and its own category.
inline Produced produced(
    const std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>& j) {
  const uint64_t n = j.first->size();
  return {n, n * 2 * sizeof(cudf::size_type)};
}

// ---------------------------------------------------------------------------
// regions
// ---------------------------------------------------------------------------
struct Pending {
  uint64_t seq;
  /// Hand-written at the call site. No longer a column (see `header_notes`); kept
  /// because it is where this source states what each cuDF call stands for.
  const char* category;
  std::string op;
  uint64_t out_rows, out_bytes, host_us;
  cudaEvent_t start, stop;
};

/// `#expr` up to the opening paren: "cudf::binary_operation" out of a call spanning four
/// lines. The column is `node_type` on the other source, and the cuDF entry point is this
/// source's answer to the same question.
inline std::string op_name(const char* expr) {
  std::string s;
  for (const char* p = expr; *p && *p != '('; ++p) {
    if (*p != ' ' && *p != '\n' && *p != '\t') s += *p;
  }
  return s;
}

/// Per-invocation state. One instance per `execute()` call, so `node_seq` restarts at 0
/// and the benchmark's repeated runs are repeated samples of one numbered chain rather
/// than one chain with ever-growing sequence numbers.
class QueryScope {
 public:
  QueryScope(const char* query, const char* label);
  ~QueryScope() { close(); }

  /// Flush now instead of at the brace. Needed where the recorded calls are separate
  /// statements whose results have to outlive the scope — the three parquet reads of q3 —
  /// and a block around them would free the tables the query is about to use. Idempotent.
  void close();

  static QueryScope* current();
  uint64_t next_seq() { return seq_++; }
  void push(const Pending& p) { pending_.push_back(p); }

 private:
  std::string query_, label_;
  uint64_t seq_ = 0;
  std::vector<Pending> pending_;
  std::chrono::steady_clock::time_point t0_;
  QueryScope* prev_;
  bool closed_ = false;
};

inline QueryScope*& scope_slot() {
  static thread_local QueryScope* s = nullptr;
  return s;
}
inline QueryScope* QueryScope::current() { return scope_slot(); }

inline QueryScope::QueryScope(const char* query, const char* label)
    : query_(query), label_(label), prev_(scope_slot()) {
  scope_slot() = this;
  t0_ = std::chrono::steady_clock::now();
}

/// Times `f`, records what it produced, and hands back its result unchanged.
template <typename F>
auto region(const char* category, const char* expr, F&& f) -> decltype(f()) {
  QueryScope* scope = QueryScope::current();
  if (!scope) return f();

  Pending p{};
  p.seq = scope->next_seq();
  p.category = category;
  p.op = op_name(expr);
  // Same name shape as the node path: seq first, because seq is the column an nsys export
  // and the record join on.
  scoped_range range{(std::to_string(p.seq) + " " + p.op).c_str()};

  const auto stream = cudf::get_default_stream().value();
  cudaEventCreateWithFlags(&p.start, cudaEventDefault);
  cudaEventCreateWithFlags(&p.stop, cudaEventDefault);
  const auto h0 = std::chrono::steady_clock::now();
  cudaEventRecord(p.start, stream);
  auto result = f();
  cudaEventRecord(p.stop, stream);
  const auto h1 = std::chrono::steady_clock::now();

  p.host_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(h1 - h0).count());
  const Produced out = produced(result);
  p.out_rows = out.rows;
  p.out_bytes = out.bytes;
  scope->push(p);
  return result;
}

// ---------------------------------------------------------------------------
// emitting
// ---------------------------------------------------------------------------
inline std::string allocator_text() {
  const auto& s = peacock::install_rmm_pool();
  if (s.state != peacock::RmmPoolStatus::State::Installed) {
    return "rmm-default (pool unavailable)";
  }
  char buf[192];
  std::snprintf(buf, sizeof(buf),
                "rmm-pool initial=%.1fGiB max=%.1fGiB of %.1fGiB free on a%s device",
                s.initial_bytes / 1073741824.0, s.maximum_bytes / 1073741824.0,
                s.free_bytes / 1073741824.0, s.integrated ? "n integrated" : " discrete");
  return buf;
}

#ifndef PEACOCK_CXX_BUILD_PROFILE
#  define PEACOCK_CXX_BUILD_PROFILE "unknown"
#endif

/// The `#` preamble, written once per file. Only what differs from the node source; the
/// column meanings are stated at length in `record.rs`, and restating them here would give
/// the two copies room to drift apart.
inline const char* header_notes() {
  return
      "# peacockdb cost-model calibration record — BARE-cuDF source, OLD format.\n"
      "# Not the format peacockdb-core/tests/common/record.rs writes: that one is per\n"
      "#   cuDF CALL, keyed by (plan node, recipe step, call index), and carries three\n"
      "#   time terms. This one predates it and is kept for the older collections under\n"
      "#   testdata/calibration/. The two are not appendable to one file, which is why\n"
      "#   this source reads PEACOCK_CUDF_RECORD_PATH and that one PEACOCK_RECORD_PATH.\n"
      "# What this source's columns mean:\n"
      "# node_type = the cuDF entry point, not a plan node: this chain has no plan.\n"
      "# node_seq = position in the hand-written chain, restarted per execute() call. The\n"
      "#   benchmark runs the chain several times, so a (query, label, node_seq) has one\n"
      "#   row per run and they are repeated samples, not distinct regions.\n"
      "# label = load or execute. The parquet read happens once, outside the timed chain.\n"
      "# peacock_host_us = 0 by construction, not by measurement. There is no host\n"
      "#   prologue here to measure — that absence is the whole point of this source.\n"
      "# There is one region per node here: one stream, no partitioning.\n"
      "# cuda_bytes = the mapped call's OUTPUT bytes, under the same formula the node\n"
      "#   source uses (peacock::logical_size_from_table).\n"
      "# The call→category mapping is NOT written out. The node source's category is a\n"
      "#   lookup in cost_model.conf and needs no column; this source's is hand-written\n"
      "#   per call and cannot be recovered from node_type, which here is a cuDF entry\n"
      "#   point rather than a plan node. It stays at the call sites as the record of\n"
      "#   what each call was taken to be.\n";
}

inline void QueryScope::close() {
  if (closed_) return;
  closed_ = true;
  scope_slot() = prev_;
  if (pending_.empty()) return;

  // One synchronize for the whole chain. The elapsed times are unreadable before it, and
  // the wall below is only a denominator if the device has actually finished.
  cudaEventSynchronize(pending_.back().stop);
  const auto t1 = std::chrono::steady_clock::now();
  const double chain_us =
      std::chrono::duration<double, std::micro>(t1 - t0_).count();

  double device_total = 0;
  std::string rows;
  for (auto& p : pending_) {
    float ms = 0;
    cudaEventElapsedTime(&ms, p.start, p.stop);
    const uint64_t device_us = static_cast<uint64_t>(ms * 1000.0f);
    device_total += device_us;
    cudaEventDestroy(p.start);
    cudaEventDestroy(p.stop);

    if (!record_path()) continue;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "cudf\ttpch\t40\t%s\t%s\t%llu\t%s\t0\t0\t%llu\t%llu\t%llu"
                  "\t0\t%llu\t%llu\t%llu\n",
                  query_.c_str(), label_.c_str(), (unsigned long long)p.seq, p.op.c_str(),
                  (unsigned long long)p.out_rows, (unsigned long long)p.out_bytes,
                  (unsigned long long)p.out_bytes, (unsigned long long)p.host_us,
                  (unsigned long long)device_us, (unsigned long long)p.host_us);
    rows += buf;
  }

  // The coverage check, and this source's reason to be trusted. The plan's original
  // criterion — "peacock_host_us comes out ≈ 0 here" — turned out to be no check at all:
  // it is 0.3us per region on the peacockdb path too, so both sources pass it and it
  // distinguishes nothing. What can go wrong instead is a cuDF call left outside a
  // region: the row it never wrote is not an error anywhere, and the fit sees a chain
  // that did less work than it did. Σ device against the chain's own wall catches
  // exactly that, and catches double-counting from the other side.
  const double covered = chain_us > 0 ? 100.0 * device_total / chain_us : 0.0;
  std::fprintf(stderr,
               "[calib] %s/%s regions=%zu Sdevice=%.0fus chain=%.0fus coverage=%.1f%%\n",
               query_.c_str(), label_.c_str(), pending_.size(), device_total, chain_us,
               covered);
  EXPECT_GE(covered, 80.0)
      << query_ << "/" << label_ << ": the timed regions account for only " << covered
      << "% of the chain's wall time. Either a cuDF call sits outside a CALIB region, or "
         "something between the regions is doing device work nobody is charged for.";

  if (!record_path()) return;
  std::ifstream probe(record_path(), std::ios::ate | std::ios::binary);
  const bool fresh = !probe.good() || probe.tellg() == 0;
  probe.close();
  std::ofstream out(record_path(), std::ios::app);
  if (!out) {
    ADD_FAILURE() << "cannot append calibration rows to " << record_path();
    return;
  }
  if (fresh) {
    // The `# run:` lines are the run's conditions, which the node source keeps in the
    // heading too and refuses to mix within one file. Same three keys, same order,
    // same spelling — a reader joins the two sources' files and has to see one format.
    out << header_notes()
        << "# run: timing_mode=events\n"
        << "# run: build_profile=" << PEACOCK_CXX_BUILD_PROFILE << "\n"
        << "# run: allocator=" << allocator_text() << "\n"
        << "source\tdataset\tsf\tquery\tlabel\tnode_seq\tnode_type"
           "\tin_rows\tin_bytes\tout_rows\tout_bytes\tcuda_bytes"
           "\tpeacock_host_us\tcudf_host_us\tdevice_us\twall_us\n";
  }
  out << rows;
}

}  // namespace peacock_test::calib

/// `CALIB(cuda_project_bytes, cudf::cast(a, t))` — one token per site, and the expression
/// stays readable as the cuDF call it is.
/// Variadic because the wrapped expression is a call with commas in it, and a two-
/// parameter macro would split on the first one.
#define CALIB(cat, ...)                                                     \
  ::peacock_test::calib::region(::peacock_test::calib::cat, #__VA_ARGS__,   \
                                [&] { return __VA_ARGS__; })
