#pragma once

#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace peacock {

/// Result of executing a plan node: a cuDF table plus column names.
struct TableResult {
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> column_names;
  /// The decimal precision each column was DECLARED with, or 0 where it is not a decimal.
  /// cuDF stores a decimal's scale and not its precision, so the export would otherwise
  /// hand back the widest the type can hold (38) for every one — which is what the plan
  /// node's `output_schema` is read here to prevent.
  ///
  /// Empty where a path has no node to read, and then the export defaults as it always did.
  std::vector<int32_t> column_precisions;
};

/// Per-node actual costs returned across the FFI. The byte formula lives ONLY in
/// Rust (no CPU/GPU drift): Rust applies the schema+row-derived `ColAccum`
/// overhead and adds `varlen_content_bytes`, the one data-dependent term that only
/// C++ can measure on the resident table.
struct NodeStats {
  uint64_t rows = 0;
  /// Σ over var-length (string) output columns of content bytes
  /// (offsets[n]-offsets[0]); additive across columns, so one total suffices.
  uint64_t varlen_content_bytes = 0;
  /// Wall-clock microseconds this OUTPUT PARTITION's work took, measured only
  /// when node timing is enabled (see `set_node_timing`); 0 otherwise. A node's
  /// time is Σ over its partitions, so the caller can sum without knowing which
  /// arm of `execute_node` produced them.
  uint64_t time_us = 0;
};

/// Enable/disable per-node timing. OFF by default, and deliberately so: measuring
/// device work requires SYNCHRONIZING the default stream at every measurement
/// boundary, which serializes what cuDF would otherwise pipeline. That is the right
/// trade for a benchmark and the wrong one for everything else.
///
/// Without the sync a host-side timer around a cuDF call measures kernel
/// SUBMISSION, not execution — and the node-by-node path (`NodeSession::execute_node`)
/// has no sync of its own: `debug_sync` is only reached from `run_op`, i.e. the
/// recursive all-at-once path. The one incidental sync here is
/// `varlen_content_bytes`, which reads `chars_size` back to the host, and only for
/// STRING columns — so timings taken without this flag would be skewed by whether a
/// node happens to output strings.
void set_node_timing(bool enabled);

/// Current state of the timing switch (see `set_node_timing`).
bool node_timing_enabled();

/// Cost of the MEASUREMENT ITSELF, in microseconds: the same timed region every
/// node pays, wrapped around no work at all (two `steady_clock` reads plus
/// `cudaStreamSynchronize` on an already-idle stream).
///
/// Why a caller wants this. A node's reported `time_us` is real work PLUS one of
/// these, and the sync's return latency is not small next to a cheap node. Without
/// the floor printed alongside them, a reader cannot tell "this node is cheap" from
/// "this node is below what the method can resolve" — the two look identical.
///
/// Returns the SECOND-smallest of `samples` (min 2, forced), matching how the
/// benchmark picks a run: the outright minimum is the one most likely to be a
/// scheduling accident. Deliberately NOT subtracted from node times anywhere —
/// subtracting a floor from numbers that are individually noisier than it would
/// manufacture zeros and hide exactly what it claims to expose.
///
/// PRECONDITION: no concurrent execution on the default stream (it synchronizes,
/// and it flips the global timing switch for the duration).
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

  /// Borrow the resident table behind `handle` (for materialization at root).
  const TableResult& table_for(uint64_t handle) const;

  /// Release a resident handle (idempotent — already-consumed handles are no-ops).
  void release(uint64_t handle);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace peacock
