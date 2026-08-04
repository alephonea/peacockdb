// Split out of the former src/plan_executor.cpp monolith.
//
// NodeSession: node-by-node execution over a parsed plan, keeping intermediates
// resident in a handle registry. `node_children` stays static here -- it is the
// session's own notion of child order and nothing else needs it.

#include "peacock/operators.h"
#include "peacock/expr.h"
#include "peacock/partitioning.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace peacock {

// ============================================================================
// Node-by-node session (unified CPU/GPU node-executor interface)
// ============================================================================

// Children of a plan node in canonical order — MUST match the Rust walk's child
// order so the caller's input handles line up with each node's inputs.
static std::vector<const fb::PlanNode*> node_children(const fb::PlanNode* node) {
  switch (node->node_type()) {
    case fb::PlanNodeKind_GpuScan:
      return {};
    case fb::PlanNodeKind_GpuFilter:
      return {node->node_as_GpuFilter()->input()};
    case fb::PlanNodeKind_GpuProject:
      return {node->node_as_GpuProject()->input()};
    case fb::PlanNodeKind_GpuAggregate:
      return {node->node_as_GpuAggregate()->input()};
    case fb::PlanNodeKind_GpuHashJoin:
      return {node->node_as_GpuHashJoin()->left(), node->node_as_GpuHashJoin()->right()};
    case fb::PlanNodeKind_GpuCrossJoin:
      return {node->node_as_GpuCrossJoin()->left(), node->node_as_GpuCrossJoin()->right()};
    case fb::PlanNodeKind_GpuNestedLoopJoin:
      return {node->node_as_GpuNestedLoopJoin()->left(),
              node->node_as_GpuNestedLoopJoin()->right()};
    case fb::PlanNodeKind_GpuSort:
      return {node->node_as_GpuSort()->input()};
    case fb::PlanNodeKind_GpuCoalesceBatches:
      return {node->node_as_GpuCoalesceBatches()->input()};
    case fb::PlanNodeKind_GpuCoalescePartitions:
      return {node->node_as_GpuCoalescePartitions()->input()};
    case fb::PlanNodeKind_GpuRepartition:
      return {node->node_as_GpuRepartition()->input()};
    case fb::PlanNodeKind_GpuSortPreservingMerge:
      return {node->node_as_GpuSortPreservingMerge()->input()};
    case fb::PlanNodeKind_GpuUnion: {
      std::vector<const fb::PlanNode*> kids;
      if (auto* in = node->node_as_GpuUnion()->inputs()) {
        for (flatbuffers::uoffset_t i = 0; i < in->size(); ++i) kids.push_back(in->Get(i));
      }
      return kids;
    }
    case fb::PlanNodeKind_GpuLimit:
      return {node->node_as_GpuLimit()->input()};
    case fb::PlanNodeKind_GpuWindow:
      return {node->node_as_GpuWindow()->input()};
    default:
      throw std::runtime_error("node_children: unsupported PlanNodeKind: " +
                               std::to_string(node->node_type()));
  }
}

struct NodeSession::Impl {
  std::vector<uint8_t> buf;  // own the plan bytes so fb pointers stay valid
  const fb::GpuPlan* plan = nullptr;
  std::vector<const fb::PlanNode*> post_order;
  std::unordered_map<uint64_t, TableResult> registry;
  uint64_t next_handle = 1;

  void index_post_order(const fb::PlanNode* node) {
    for (auto* child : node_children(node)) index_post_order(child);
    post_order.push_back(node);
  }
};

NodeSession::NodeSession(const uint8_t* plan_bytes, uint64_t plan_len)
    : impl_(std::make_unique<Impl>()) {
  impl_->buf.assign(plan_bytes, plan_bytes + plan_len);
  impl_->plan = fb::GetGpuPlan(impl_->buf.data());
  if (!impl_->plan) throw std::runtime_error("failed to parse FlatBuffer GpuPlan");
  flatbuffers::Verifier verifier(impl_->buf.data(), impl_->buf.size(), /*max_depth=*/1024);
  if (!impl_->plan->Verify(verifier))
    throw std::runtime_error("FlatBuffer verification failed");
  auto* root = impl_->plan->root();
  if (!root) throw std::runtime_error("GpuPlan has no root node");
  impl_->index_post_order(root);
}

NodeSession::~NodeSession() = default;

size_t NodeSession::node_count() const { return impl_->post_order.size(); }

void NodeSession::execute_node(uint64_t seq, const uint64_t* input_handles,
                               const uint64_t* input_child_counts, size_t n_children,
                               uint64_t* out_handles, size_t out_cap, size_t* out_count,
                               NodeStats* out_stats) {
  if (seq >= impl_->post_order.size())
    throw std::runtime_error("NodeSession::execute_node: seq out of range");
  const fb::PlanNode* node = impl_->post_order[seq];

  // Each child contributes a VECTOR of partition handles (multi-handle model,
  // Phase 2). The flat `input_handles` is grouped by child via `input_child_counts`.
  std::vector<std::vector<uint64_t>> child(n_children);
  size_t off = 0;
  for (size_t c = 0; c < n_children; ++c) {
    size_t cnt = input_child_counts ? static_cast<size_t>(input_child_counts[c]) : 0;
    child[c].assign(input_handles + off, input_handles + off + cnt);
    off += cnt;
  }

  // (iii) GpuScan with an explicit RG→batch→partition MAP → emit N partitions, one
  // per ScanBatch, each a set_row_groups read of that entry's row groups. This is
  // the SAME map the Rust CpuNodeExecutor / golden generator replay, so per-partition
  // row counts match by construction. EMPTY map => fall through to the generic path
  // (single-partition read of `row_groups`, tp1 byte-identical).
  if (node->node_type() == fb::PlanNodeKind_GpuScan) {
    const fb::GpuScan* scan = node->node_as_GpuScan();
    if (scan->batches() && scan->batches()->size() > 0) {
      size_t n = scan->batches()->size();
      if (n > out_cap)
        throw std::runtime_error("NodeSession::execute_node: out_handles buffer too small");
      for (size_t p = 0; p < n; ++p) {
        const fb::ScanBatch* b = scan->batches()->Get(static_cast<flatbuffers::uoffset_t>(p));
        TableResult result = execute_scan(scan, b->row_groups());
        auto tv = result.table->view();
        if (out_stats)
          out_stats[p] = NodeStats{static_cast<uint64_t>(tv.num_rows()), varlen_content_bytes(tv)};
        uint64_t handle = impl_->next_handle++;
        impl_->registry.emplace(handle, std::move(result));
        out_handles[p] = handle;  // map entries are stored in partition order 0..n-1
      }
      *out_count = n;
      return;
    }
  }

  // Partition-COLLAPSING nodes → concatenate ALL M child partitions into ONE output
  // (BUFFERING: the full concatenated table is resident), in partition-index order to
  // match the Rust CpuNodeExecutor's `collapses_partitions` concat. NOT a per-partition
  // passthrough.
  //   - GpuCoalescePartitions: the explicit M→1 concat before a Hash repartition.
  //   - GpuSortPreservingMerge: merges N sorted partitions into one (q1's top ORDER BY
  //     node). A plain concat suffices HERE: the node-by-node result is compared
  //     order-independently (batches_to_sorted_str) and the per-node cost is schema-
  //     driven (rows + schema), so partitions=1 + the right row count match the #13
  //     golden by construction. (The production fast path keeps its own sort-merge.)
  if (node->node_type() == fb::PlanNodeKind_GpuCoalescePartitions ||
      node->node_type() == fb::PlanNodeKind_GpuSortPreservingMerge) {
    if (out_cap < 1)
      throw std::runtime_error("NodeSession::execute_node: out_handles buffer too small");
    std::vector<TableResult> owned;
    std::vector<cudf::table_view> views;
    owned.reserve(child[0].size());
    views.reserve(child[0].size());
    for (uint64_t h : child[0]) {
      auto it = impl_->registry.find(h);
      if (it == impl_->registry.end())
        throw std::runtime_error("NodeSession::execute_node: unknown input handle");
      owned.push_back(std::move(it->second));
      impl_->registry.erase(it);
      views.push_back(owned.back().table->view());
    }
    TableResult result;
    result.column_names = owned.empty() ? std::vector<std::string>{} : owned[0].column_names;

    const fb::GpuSortPreservingMerge* spm =
        (node->node_type() == fb::PlanNodeKind_GpuSortPreservingMerge)
            ? node->node_as_GpuSortPreservingMerge()
            : nullptr;
    if (spm && spm->exprs() && spm->exprs()->size() > 0 && views.size() > 1) {
      // (#99) SortPreservingMerge = K-WAY MERGE the N already-sorted partitions by the
      // SPM's sort keys, NOT a plain concat. Concat leaves the output globally
      // UNSORTED (only per-partition-sorted), so a downstream LIMIT/fetch picks the
      // wrong top-N. The N inputs are each sorted upstream by the SAME GpuSort spec
      // (execute_sort reads the identical SortExprNode fields), which is cudf::merge's
      // precondition. Column-ref keys only (post-aggregate ORDER BY, the whole top-N
      // corpus); an EXPRESSION sort key would need per-partition materialization
      // (none present) — throw loudly rather than silently mis-merge.
      std::vector<cudf::size_type> key_cols;
      std::vector<cudf::order> orders;
      std::vector<cudf::null_order> null_orders;
      for (flatbuffers::uoffset_t i = 0; i < spm->exprs()->size(); ++i) {
        auto* se = spm->exprs()->Get(i);
        auto* expr = se->expr();
        if (!expr || expr->node_type() != fb::ExprNode_ColumnRef)
          throw std::runtime_error(
              "GpuSortPreservingMerge: expression sort key not supported by the k-way "
              "merge (needs per-partition materialization) — file an increment");
        key_cols.push_back(
            static_cast<cudf::size_type>(expr->node_as_ColumnRef()->index()));
        orders.push_back(se->asc() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
        null_orders.push_back(se->nulls_first() ? cudf::null_order::BEFORE
                                                : cudf::null_order::AFTER);
      }
      result.table = cudf::merge(views, key_cols, orders, null_orders);
      // Apply the SPM's own fetch (top-N) AFTER the global merge (-1 = unlimited).
      if (spm->fetch() >= 0) {
        auto n = std::min(static_cast<cudf::size_type>(spm->fetch()),
                          result.table->view().num_rows());
        std::vector<cudf::size_type> slice_indices{0, n};
        auto sliced = cudf::slice(result.table->view(), slice_indices);
        result.table = std::make_unique<cudf::table>(sliced[0]);
      }
    } else {
      // GpuCoalescePartitions, or an SPM with no sort keys / a single partition:
      // a plain in-order concat is the correct collapse.
      result.table = cudf::concatenate(views);
    }
    auto tv = result.table->view();
    if (out_stats)
      out_stats[0] = NodeStats{static_cast<uint64_t>(tv.num_rows()), varlen_content_bytes(tv)};
    uint64_t handle = impl_->next_handle++;
    impl_->registry.emplace(handle, std::move(result));
    out_handles[0] = handle;
    *out_count = 1;
    return;
  }

  // (Inc2) GpuRepartition Hash → scatter the ONE input table into N partitions by
  // Spark-murmur3 (comet-identical) hash of the key columns, so the per-partition
  // row counts match the #13 CPU twin (and the golden) by construction — the live
  // conformance gate proves the kernel is bit-equal to comet. Post-lowering the child
  // is a GpuCoalescePartitions (single handle); we defensively concat whatever input
  // partitions arrive into one table, then spark_hash_partition + slice into N.
  if (node->node_type() == fb::PlanNodeKind_GpuRepartition &&
      node->node_as_GpuRepartition()->kind() == fb::PartitioningKind_Hash) {
    const fb::GpuRepartition* rp = node->node_as_GpuRepartition();
    size_t n = static_cast<size_t>(rp->num_partitions());
    if (n == 0 || n > out_cap)
      throw std::runtime_error("NodeSession::execute_node: bad Hash repartition out count");

    // Gather + concat the child partitions into one table (matches the CPU concat).
    std::vector<TableResult> owned;
    std::vector<cudf::table_view> views;
    owned.reserve(child[0].size());
    views.reserve(child[0].size());
    for (uint64_t h : child[0]) {
      auto it = impl_->registry.find(h);
      if (it == impl_->registry.end())
        throw std::runtime_error("NodeSession::execute_node: unknown input handle");
      owned.push_back(std::move(it->second));
      impl_->registry.erase(it);
      views.push_back(owned.back().table->view());
    }
    std::vector<std::string> column_names =
        owned.empty() ? std::vector<std::string>{} : owned[0].column_names;
    std::unique_ptr<cudf::table> combined =
        (owned.size() == 1) ? std::move(owned[0].table) : cudf::concatenate(views);

    // Hash keys: ColumnRef indices into the (partial-agg output) table. Inc2 scope =
    // ColumnRef keys only (the group-by columns); other exprs arrive at Inc6/7.
    std::vector<cudf::size_type> key_cols;
    if (auto* exprs = rp->hash_exprs()) {
      for (flatbuffers::uoffset_t i = 0; i < exprs->size(); ++i) {
        const fb::Expr* e = exprs->Get(i);
        if (e->node_type() != fb::ExprNode_ColumnRef)
          throw std::runtime_error("GpuRepartition: only ColumnRef hash keys supported (Inc2)");
        key_cols.push_back(static_cast<cudf::size_type>(e->node_as_ColumnRef()->index()));
      }
    }

    auto tv = combined->view();
    auto [parted, offsets] = peacock::partitioning::spark_hash_partition(
        tv, key_cols, static_cast<cudf::size_type>(n));
    const cudf::size_type total = parted->num_rows();
    const cudf::table_view pv = parted->view();
    for (size_t p = 0; p < n; ++p) {
      cudf::size_type start = offsets[p];
      cudf::size_type end = (p + 1 < n) ? offsets[p + 1] : total;
      // One owning table per partition (slice → deep copy so each handle owns memory).
      cudf::table_view slice = cudf::slice(pv, {start, end}).front();
      TableResult part;
      part.column_names = column_names;
      part.table = std::make_unique<cudf::table>(slice);
      auto ptv = part.table->view();
      if (out_stats)
        out_stats[p] = NodeStats{static_cast<uint64_t>(ptv.num_rows()), varlen_content_bytes(ptv)};
      uint64_t handle = impl_->next_handle++;
      impl_->registry.emplace(handle, std::move(part));
      out_handles[p] = handle;
    }
    *out_count = n;
    return;
  }

  // Output partition count. Ordinary ops MAP over their children's partitions
  // (all children carry the same count), so n_out = child[0]'s count. Partition-
  // collapsing ops (GpuScan map, GpuCoalescePartitions) are handled above; every
  // remaining node maps 1:1 (tp1 empty-map scans give n_out == 1 = byte-identical
  // to the pre-multi-handle path).
  size_t n_out = (n_children > 0) ? child[0].size() : 1;
  if (n_out == 0) n_out = 1;
  if (n_out > out_cap)
    throw std::runtime_error("NodeSession::execute_node: out_handles buffer too small");
  // The per-partition MAP arm requires every child to carry the same partition
  // count (child[c][p] is read for all c). Joins with mismatched per-child counts
  // arrive at Inc2 (partitioned joins); until then fail LOUDLY rather than read OOB.
  for (size_t c = 1; c < n_children; ++c) {
    if (child[c].size() != n_out)
      throw std::runtime_error(
          "NodeSession::execute_node: children have mismatched partition counts "
          "(multi-partition joins are not implemented yet)");
  }

  for (size_t p = 0; p < n_out; ++p) {
    std::vector<TableResult> inputs;
    inputs.reserve(n_children);
    for (size_t c = 0; c < n_children; ++c) {
      uint64_t h = child[c][p];  // partition p of child c (ordinary op maps per partition)
      auto it = impl_->registry.find(h);
      if (it == impl_->registry.end())
        throw std::runtime_error("NodeSession::execute_node: unknown input handle");
      inputs.push_back(std::move(it->second));
      impl_->registry.erase(it);
    }
    TableResult result = execute_one(node, std::move(inputs));
    auto tv = result.table->view();
    if (out_stats)
      out_stats[p] = NodeStats{static_cast<uint64_t>(tv.num_rows()), varlen_content_bytes(tv)};
    uint64_t handle = impl_->next_handle++;
    impl_->registry.emplace(handle, std::move(result));
    out_handles[p] = handle;
  }
  *out_count = n_out;
}

const TableResult& NodeSession::table_for(uint64_t handle) const {
  auto it = impl_->registry.find(handle);
  if (it == impl_->registry.end())
    throw std::runtime_error("NodeSession::table_for: unknown handle");
  return it->second;
}

void NodeSession::release(uint64_t handle) { impl_->registry.erase(handle); }


}  // namespace peacock
