#pragma once

#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace peacock::plan {
struct Schema;
}
namespace fb = peacock::plan;

namespace peacock {

/// Result of executing a plan node: a cuDF table plus column names.
struct TableResult {
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> column_names;
};

/// Per-node actual costs returned across the FFI. On this path the byte formula still
/// lives only in Rust (no CPU/GPU drift): Rust applies the schema+row-derived `ColAccum`
/// overhead and adds `varlen_content_bytes`, the one data-dependent term only C++ can
/// measure on the resident table. `logical_bytes` below is a second implementation of
/// that formula, not consumed here — see its comment.
struct NodeStats {
  uint64_t rows = 0;
  /// Σ over var-length (string) output columns of content bytes
  /// (offsets[n]-offsets[0]); additive across columns, so one total suffices.
  uint64_t varlen_content_bytes = 0;
  /// Host microseconds this output partition spent before touching the device:
  /// flatbuffer decode, registry lookups, `ExprContext`/AST construction. 0 unless
  /// node timing is on.
  ///
  /// The term the calibration calls `const_peacock`, and the reason the split exists:
  /// bare cuDF has no such prologue, so one number covering both halves cannot be
  /// fitted across the two datasets. Ends at `mark_device_start`, not at the
  /// operator's first line — an operator that interleaves decode with column
  /// materialization charges from its first kernel onward to `host_submit_us`, which
  /// understates the peacockdb-only prologue rather than inflating it.
  uint64_t host_setup_us = 0;
  /// Host microseconds from the first device touch to the end of the region. What it
  /// covers depends on the mode, and the two are NOT comparable:
  ///   - `NodeTiming::Events` — no explicit drain; the device work is `device_us`,
  ///     collected separately. Not launch cost, though: cuDF returns owned columns
  ///     and rmm frees them, both synchronizing internally, so this tracks `device_us`
  ///     closely (within 0.01% on tpch q3).
  ///   - `NodeTiming::Sync`   — the region ends in a stream sync, so this contains the
  ///     device execution outright. The legacy single number.
  uint64_t host_submit_us = 0;
  /// The same total Rust derives with `logical_size_from_schema`, recomputed from
  /// cuDF types. Nothing on the peacockdb path consumes it; it exists to be compared
  /// against Rust's, because the bare-cuDF sf40 tests have no Rust to ask and the
  /// calibration only means anything if both datasets land on one byte axis. See
  /// `logical_size_from_table`.
  uint64_t logical_bytes = 0;
  /// 1 when this partition's columns are one for one the types `output_schema`
  /// declares; 0 when the device materialized something else.
  ///
  /// Scopes the comparison above: whether two implementations of the byte rule agree
  /// is only askable where both look at the same columns. Legitimate divergences —
  /// a Partial AVG under GROUPING SETS emitting one MEAN where DataFusion declares
  /// `[count]`+`[sum]` (aggregate.cpp, `avg_state_2col`), a union branch holding a
  /// decimal literal as FLOAT64 until `execute_union` retypes it (#41),
  /// `__grouping_id` built INT32 against a declared UInt8 (#196) — are shape, not
  /// byte-rule drift, and none can arise on the bare-cuDF sf40 path.
  uint64_t schema_faithful = 1;
};

/// How per-node regions are measured. Off by default: both modes cost the normal path
/// something, and one of them changes how the engine SCHEDULES.
enum class NodeTiming : int {
  Off = 0,
  /// Host clock around the region, closed by `cudaStreamSynchronize`. The sync
  /// serializes what cuDF would pipeline, so measuring changes what is measured;
  /// kept as the baseline the events mode is checked against.
  Sync = 1,
  /// CUDA events around the device work, host clock around the host work, no sync
  /// inside the region. Device times are not known at region close and are read
  /// afterwards by `collect_node_times`.
  Events = 2,
};

/// Set the timing mode (process-global; `Off` by default).
///
/// Opt-in because `Sync` drains the default stream at every boundary — right for a
/// benchmark, wrong for everything else — and `Events`, though cheap, still allocates
/// an event pair per region and holds it until collection.
///
/// Neither mode removes every sync: `varlen_content_bytes` reads `chars_size` back, so
/// a node with STRING outputs synchronizes regardless.
void set_node_timing(NodeTiming mode);

/// The current timing mode (see `set_node_timing`).
NodeTiming node_timing();

/// True unless the mode is `Off`.
bool node_timing_enabled();

/// Mark where the current timed region begins touching the device — after the decode,
/// the registry lookups and any `ExprContext`/AST construction, immediately before
/// issuing device work.
///
/// First call in a region wins, the rest are a predictable branch: put the call at
/// every point that could be first and let idempotence sort out which one was.
///
/// Placement is the point of the split, not a detail. `cudaEventRecord` on an idle
/// stream timestamps when the stream REACHES the event, so a mark at the top of a node
/// bills the host prologue as device work — exactly what `host_setup_us` isolates.
///
/// No-op when timing is off or no region is open (the recursive `execute_plan` path),
/// so operators can call it unconditionally.
void mark_device_start();

/// One collected region: which node output partition it belongs to, and what the
/// device spent on it.
struct NodeDeviceTime {
  uint64_t seq = 0;
  uint64_t partition = 0;
  /// Microseconds between the region's start and stop events. Present only for
  /// regions that recorded BOTH — see `NodeSession::collect_node_times`.
  uint64_t device_us = 0;
};

/// Cost of the measurement itself under `NodeTiming::Sync`, in microseconds: that mode's
/// region around no work at all (two `steady_clock` reads plus `cudaStreamSynchronize`
/// on an idle stream).
///
/// A `Sync` node time is real work plus one of these, and the sync's return latency is
/// not small next to a cheap node — without the floor beside them, "cheap" and "below
/// what the method resolves" look identical.
///
/// Measures the SYNC mode even when the caller runs with events: it is the number that
/// says what events bought. An events-mode floor is a different quantity.
///
/// Returns the second-smallest of `samples` (min 2, forced), matching how the benchmark
/// picks a run. Not subtracted from node times anywhere — node measurements are
/// individually noisier than the floor, so subtracting manufactures zeros.
///
/// PRECONDITION: no concurrent execution on the default stream (it synchronizes, and
/// flips the global timing switch for the duration).
uint64_t measure_timing_floor_us(unsigned samples);

/// Execute a FlatBuffer-encoded GPU plan and return the result table.
/// Thin recursive wrapper over the single-node executor — the production fast path.
///
/// @throws std::runtime_error on parse or execution errors.
TableResult execute_plan(const uint8_t* plan_bytes, uint64_t plan_len);

/// Σ var-length content bytes over a table's columns (see `NodeStats`).
uint64_t varlen_content_bytes(const cudf::table_view& table);

/// The half-open row range `[offset, offset+length)` names in a table of `num_rows`,
/// with `length == UINT64_MAX` meaning to the end.
///
/// An offset at or past the end gives an empty range, and a range running past the end
/// clamps to it — neither throws, because the caller is a limit interval whose fetch
/// legitimately overruns the batch it straddles.
std::pair<cudf::size_type, cudf::size_type> clamp_row_range(uint64_t offset, uint64_t length,
                                                            cudf::size_type num_rows);
/// A table's `output_bytes` under the RUST byte formula
/// (`peacockdb-core/src/memory.rs::logical_size_from_schema`), given its already
/// measured var-length content total.
///
/// A second implementation of a rule the codebase otherwise keeps in one place, so the
/// reason has to be stated: the bare-cuDF sf40 tests never enter Rust, and a calibration
/// fitting sf1 and sf40 on one line is wrong by an unknown factor if the two ends count
/// bytes differently. On the peacockdb path it exists only to be compared.
///
/// Models the RUST formula, not cuDF's physical layout, and the two differ: BOOL8 is a
/// byte per row on the device and a bit per row here, and the validity bitmap is charged
/// to every column whether nullable or not.
///
/// Unhandled type ids throw rather than contributing zero, matching the Rust-side panic,
/// so a newly supported type breaks both ends loudly instead of silently disagreeing.
///
/// One ambiguity is irreducible and left to the comparison: `fb_to_type_id` (expr.cpp)
/// collapses `Utf8`, `LargeUtf8` and `Utf8View` onto one cuDF STRING, which Rust widths
/// at 4-byte offsets for the first and third and 8 for the second, and nothing on the
/// device recovers which it was. This assumes 4, what the corpus produces; a `LargeUtf8`
/// column surfaces as a mismatch against Rust, which is the intended outcome.
uint64_t logical_size_from_table(const cudf::table_view& table, uint64_t varlen_content);

/// Build a finished output partition's `NodeStats`. The one place both byte fields are
/// derived, so the measured term and the modelled total cannot drift apart and
/// `chars_size` is read back once rather than twice.
///
/// Takes the whole `TableResult` rather than a view because it needs the column NAMES:
/// `execute_project` synthesizes a `__rowcount__` column for an empty projection (cuDF
/// has no 0-column table with rows), which is absent from `output_schema` and excluded
/// from both byte fields — a device representation detail must not reach the logical
/// byte axis — while `rows` still comes from the full table.
///
/// `declared` is the node's `output_schema`, used only for `schema_faithful`; nullptr
/// reports 1 (nothing contradicts it) rather than 0.
NodeStats node_stats_for(const TableResult& result, uint64_t host_setup_us,
                         uint64_t host_submit_us, const fb::Schema* declared);

/// Node-by-node execution session: parses a plan once and drives ONE node at a
/// time given already-resident child inputs, keeping intermediates resident in a
/// handle registry. Used by the unified CPU/GPU node-executor interface; the
/// all-at-once `execute_plan` remains the production fast path.
///
/// Nodes are addressed by canonical POST-ORDER sequence (children left-to-right,
/// then the node) — the SAME order the Rust walk uses, so the caller's child
/// handles align with each node's inputs.
class NodeSession {
 public:
  /// Parse + verify the plan and index its nodes in post-order.
  NodeSession(const uint8_t* plan_bytes, uint64_t plan_len);
  ~NodeSession();
  NodeSession(const NodeSession&) = delete;
  NodeSession& operator=(const NodeSession&) = delete;

  /// Number of plan nodes (post-order positions 0..count-1).
  size_t node_count() const;

  /// Execute the node at post-order `seq`. Each child contributes a VECTOR of
  /// partition handles: `input_handles` is the flattened concatenation grouped by
  /// child, `input_child_counts[c]` = child c's partition count. Output handles go
  /// to `out_handles[0..*out_count]` (caller buffer of `out_cap`) and
  /// `out_stats[0..*out_count]` is filled PER PARTITION, so Rust can sum the
  /// ColAccum overhead per partition: Σ_p ColAccum(rows_p), NOT ColAccum(Σ rows).
  /// Input handles are CONSUMED.
  void execute_node(uint64_t seq, const uint64_t* input_handles,
                    const uint64_t* input_child_counts, size_t n_children,
                    uint64_t* out_handles, size_t out_cap, size_t* out_count,
                    NodeStats* out_stats);

  /// Execute the `CudfScan` at post-order `seq` reading exactly `row_groups` rather
  /// than the list the node carries, and register its one output table. Throws naming
  /// the kind when `seq` is any other node — this entry point has no generic arm to
  /// fall back to. `out_stats` may be null.
  uint64_t execute_scan_rowgroups(uint64_t seq, cudf::host_span<const uint32_t> row_groups,
                                  NodeStats* out_stats);

  /// Rows `[offset, offset+length)` of `handle` copied into a new owning handle
  /// (`clamp_row_range` for the edges). The input handle is CONSUMED, as every
  /// operation on a resident table is.
  uint64_t slice_handle(uint64_t handle, uint64_t offset, uint64_t length);
  /// Drain every event pair recorded since the last call, one entry per region in
  /// execution order. Empty unless the mode was `NodeTiming::Events`.
  ///
  /// Separate from `execute_node` because the answer does not exist when a node returns
  /// — the point of events — and separate from session destruction because that
  /// destroys the events, so a caller reading only at `end_plan` reads nothing. Call it
  /// after the root `materialize`.
  ///
  /// Collected regions are released, so a second call does not double-report and a long
  /// session does not accumulate events forever.
  ///
  /// Incomplete pairs are dropped, not reported as zero: a node that threw leaves a
  /// start and no stop, and `cudaEventElapsedTime` on such a pair fails with
  /// `cudaErrorInvalidResourceHandle`, taking the whole collection down with it. A
  /// region that never touched the device recorded neither and is equally absent; its
  /// host halves are still in `NodeStats`.
  std::vector<NodeDeviceTime> collect_node_times();

  /// Borrow the resident table behind `handle` (for materialization at root).
  const TableResult& table_for(uint64_t handle) const;

  /// Release a resident handle (idempotent — already-consumed handles are no-ops).
  void release(uint64_t handle);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace peacock
