#pragma once

#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace peacock {

/// Result of executing a plan node: a cuDF table plus column names.
struct TableResult {
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> column_names;
};

/// Per-node actual costs returned across the FFI. Rust applies the single-source
/// `ColAccum` overhead (validity bitmap + fixed-width + var-length OFFSET buffers,
/// all schema+row derived) and ADDS `varlen_content_bytes` — the byte formula
/// lives ONLY in Rust (no CPU/GPU drift); C++ supplies just the data-dependent
/// term it alone can measure on the resident table.
struct NodeStats {
  uint64_t rows = 0;
  /// Σ over var-length (string) output columns of content bytes
  /// (offsets[n]-offsets[0]); additive across columns, so one total suffices.
  uint64_t varlen_content_bytes = 0;
};

/// Execute a FlatBuffer-encoded GPU plan and return the result table.
/// Thin recursive wrapper over the single-node executor — the production fast path.
///
/// @throws std::runtime_error on parse or execution errors.
TableResult execute_plan(const uint8_t* plan_bytes, uint64_t plan_len);

/// Σ var-length content bytes over a table's columns (see `NodeStats`).
uint64_t varlen_content_bytes(const cudf::table_view& table);

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

  /// Execute the node at post-order `seq` (multi-handle model, Phase 2). Each
  /// child contributes a VECTOR of partition handles: `input_handles` is the
  /// flattened concatenation grouped by child, `input_child_counts[c]` = child
  /// c's partition count, `n_children` = number of children. The node's output
  /// partition handles are written to `out_handles[0..*out_count]` (caller buffer
  /// of `out_cap`; partition count is bounded by target_partitions), and `*out`
  /// is filled with the Σ-over-partitions stats. Input handles are CONSUMED.
  void execute_node(uint64_t seq, const uint64_t* input_handles,
                    const uint64_t* input_child_counts, size_t n_children,
                    uint64_t* out_handles, size_t out_cap, size_t* out_count,
                    NodeStats* out);

  /// Borrow the resident table behind `handle` (for materialization at root).
  const TableResult& table_for(uint64_t handle) const;

  /// Release a resident handle (idempotent — already-consumed handles are no-ops).
  void release(uint64_t handle);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace peacock
