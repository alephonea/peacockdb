// The plan-node dispatch switch (run_op) and its two drivers, execute_node
// (recursive) and execute_one (single-node). Co-located deliberately: they are one
// dispatch mechanism.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/table/table.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace peacock {

static const char* plan_node_kind_name(fb::PlanNodeKind k) {
  switch (k) {
    case fb::PlanNodeKind_GpuScan:                return "GpuScan";
    case fb::PlanNodeKind_GpuFilter:              return "GpuFilter";
    case fb::PlanNodeKind_GpuProject:             return "GpuProject";
    case fb::PlanNodeKind_GpuAggregate:           return "GpuAggregate";
    case fb::PlanNodeKind_GpuHashJoin:            return "GpuHashJoin";
    case fb::PlanNodeKind_GpuCrossJoin:           return "GpuCrossJoin";
    case fb::PlanNodeKind_GpuNestedLoopJoin:      return "GpuNestedLoopJoin";
    case fb::PlanNodeKind_GpuSort:                return "GpuSort";
    case fb::PlanNodeKind_GpuCoalesceBatches:     return "GpuCoalesceBatches";
    case fb::PlanNodeKind_GpuCoalescePartitions:  return "GpuCoalescePartitions";
    case fb::PlanNodeKind_GpuRepartition:         return "GpuRepartition";
    case fb::PlanNodeKind_GpuSortPreservingMerge: return "GpuSortPreservingMerge";
    case fb::PlanNodeKind_GpuUnion:               return "GpuUnion";
    case fb::PlanNodeKind_GpuLimit:               return "GpuLimit";
    case fb::PlanNodeKind_GpuWindow:              return "GpuWindow";
    default:                                       return "Unknown";
  }
}


// Run one node's op (the dispatch switch). In recursive mode each op's
// `execute_node(child)` recurses here; in single-node mode it returns inputs.
static TableResult run_op(const fb::PlanNode* node, NodeInputs* in) {
  if (!node) throw std::runtime_error("null PlanNode");

  const char* kind = plan_node_kind_name(node->node_type());
  PCK_TRACE("enter %s", kind);

  TableResult result;
  try {
    switch (node->node_type()) {
      case fb::PlanNodeKind_GpuScan:
        result = execute_scan(node->node_as_GpuScan()); break;
      case fb::PlanNodeKind_GpuFilter:
        result = execute_filter(node->node_as_GpuFilter(), in); break;
      case fb::PlanNodeKind_GpuProject:
        result = execute_project(node->node_as_GpuProject(), in); break;
      case fb::PlanNodeKind_GpuAggregate:
        result = execute_aggregate(node->node_as_GpuAggregate(), in); break;
      case fb::PlanNodeKind_GpuHashJoin:
        result = execute_hash_join(node->node_as_GpuHashJoin(), in); break;
      case fb::PlanNodeKind_GpuCrossJoin:
        result = execute_cross_join(node->node_as_GpuCrossJoin(), in); break;
      case fb::PlanNodeKind_GpuNestedLoopJoin:
        result = execute_nested_loop_join(node->node_as_GpuNestedLoopJoin(), in); break;
      case fb::PlanNodeKind_GpuSort:
        result = execute_sort(node->node_as_GpuSort(), in); break;
      case fb::PlanNodeKind_GpuCoalesceBatches:
        result = execute_passthrough(node->node_as_GpuCoalesceBatches()->input(), in); break;
      case fb::PlanNodeKind_GpuCoalescePartitions:
        result = execute_passthrough(node->node_as_GpuCoalescePartitions()->input(), in); break;
      case fb::PlanNodeKind_GpuRepartition:
        result = execute_passthrough(node->node_as_GpuRepartition()->input(), in); break;
      case fb::PlanNodeKind_GpuSortPreservingMerge:
        result = execute_passthrough(node->node_as_GpuSortPreservingMerge()->input(), in); break;
      case fb::PlanNodeKind_GpuUnion:
        result = execute_union(node->node_as_GpuUnion(), in); break;
      case fb::PlanNodeKind_GpuLimit:
        result = execute_limit(node->node_as_GpuLimit(), in); break;
      case fb::PlanNodeKind_GpuWindow:
        result = execute_window(node->node_as_GpuWindow(), in); break;
      default:
        throw std::runtime_error(
            "unsupported PlanNodeKind: " + std::to_string(node->node_type()));
    }
  } catch (const std::exception& e) {
    std::string msg = e.what();
    if (msg.find("[in ") == std::string::npos) {
      throw std::runtime_error(std::string("[in ") + kind + "] " + msg);
    }
    throw;
  }

  debug_sync(kind);
  if (debug_enabled()) {
    auto tv = result.table->view();
    PCK_TRACE("leave %s rows=%d cols=%d", kind, tv.num_rows(), tv.num_columns());
  }
  return result;
}

// Recursive driver (production fast path) OR single-node child resolver.
TableResult execute_node(const fb::PlanNode* node, NodeInputs* in) {
  if (in && in->items) {
    if (in->idx >= in->items->size()) {
      throw std::runtime_error("execute_one: not enough input handles for node");
    }
    return std::move((*in->items)[in->idx++]);
  }
  return run_op(node, in);
}

TableResult execute_one(const fb::PlanNode* node, std::vector<TableResult> inputs) {
  const size_t provided = inputs.size();
  NodeInputs in{&inputs, 0};
  TableResult result = run_op(node, &in);

  // INVARIANT: a node handed inputs must consume ALL of them.
  //
  // Under-consumption means execute_node did not see the caller's inputs, fell
  // through to run_op, and re-executed that child subtree from parquet — CORRECT
  // ANSWERS at exponential cost, so goldens, byte digests and result comparison
  // all still pass. This check is the only detector.
  //
  // Safe for every dispatch case: scans are leaves (provided == 0) and every other
  // op resolves its children unconditionally.
  //
  // Deliberately `!= provided`, NOT `> 0 && idx == 0`: the weaker form covers only
  // single-child ops. A hash join threading `in` to its left child but nullptr to
  // its right consumes 1 of 2, passes the weak check, and re-executes the whole
  // right subtree — the costliest form of exactly this bug.
  if (in.idx != provided) {
    throw std::runtime_error(
        "execute_one: node was given " + std::to_string(provided) +
        " input(s) but consumed " + std::to_string(in.idx) +
        " — the unconsumed children were re-executed instead of reused. This is the "
        "silent re-execution bug: correct results, exponential cost.");
  }
  return result;
}

}  // namespace peacock
