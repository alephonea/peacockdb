#pragma once
// PRIVATE header -- must stay under src/, NOT include/ (see plan_types.h).
//
// The per-operator entry points, non-static so each operator can live in its own
// translation unit under src/operators/.

#include "peacock/plan_types.h"
#include "plan_executor.h"

#include <vector>

namespace peacock {

// Explicit input channel, threaded through the call chain. Must stay a parameter,
// never ambient state: a per-TU global would read null after a split and silently
// re-execute the whole subtree from parquet — correct answers at exponential cost,
// invisible to every correctness test. A parameter also nests without save/restore.
//
// CONTRACT: `items == nullptr` means RECURSIVE mode — execute_node runs the child
// itself. Non-null means SINGLE-NODE mode — children are already resident and are
// consumed positionally, in the same post-order the caller pushed them.
struct NodeInputs {
  std::vector<TableResult>* items = nullptr;
  size_t idx = 0;
};

// GpuScan is the one LEAF: it reads Parquet and takes no NodeInputs, which is why
// execute_one's consume-all invariant is trivially satisfied for zero-input nodes.
//
// Default argument on the DECLARATION only -- repeating it on the definition is a
// hard error.
TableResult execute_scan(const fb::GpuScan* scan,
                         const flatbuffers::Vector<uint32_t>* row_groups_override = nullptr);
TableResult execute_filter(const fb::GpuFilter* filter, NodeInputs* in);
TableResult execute_project(const fb::GpuProject* proj, NodeInputs* in);
TableResult execute_aggregate(const fb::GpuAggregate* agg, NodeInputs* in);
TableResult execute_hash_join(const fb::GpuHashJoin* join, NodeInputs* in);
TableResult execute_cross_join(const fb::GpuCrossJoin* join, NodeInputs* in);
TableResult execute_nested_loop_join(const fb::GpuNestedLoopJoin* join, NodeInputs* in);
TableResult execute_sort(const fb::GpuSort* sort, NodeInputs* in);
TableResult execute_union(const fb::GpuUnion* u, NodeInputs* in);
TableResult execute_limit(const fb::GpuLimit* limit, NodeInputs* in);
TableResult execute_window(const fb::GpuWindow* win, NodeInputs* in);

// The recursive driver. Every operator TU calls it to resolve its children, as do
// execute_plan.cpp and node_session.cpp, so it has to be header-declared. Its
// companions run_op and plan_node_kind_name stay STATIC in dispatch.cpp -- nothing
// outside that TU calls them, and they are one dispatch mechanism with it.
TableResult execute_node(const fb::PlanNode* node, NodeInputs* in);

// `inputs` BY VALUE: execute_one owns them for the duration of the call and hands
// a NodeInputs pointing at that local down the chain.
TableResult execute_one(const fb::PlanNode* node, std::vector<TableResult> inputs);

// Kept `inline` so it does not become a real call through the .so; per-node
// overhead is measured in llm-wiki/reports/benchmark-minimal.md.
inline TableResult execute_passthrough(const fb::PlanNode* input_node, NodeInputs* in) {
  return execute_node(input_node, in);
}

}  // namespace peacock
