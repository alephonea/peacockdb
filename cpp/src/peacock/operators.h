#pragma once
// PRIVATE header -- deliberately under src/, NOT include/. CMakeLists ships
// include/ wholesale via install(DIRECTORY include/), so anything placed there
// becomes public API sitting next to the stable C FFI surface (peacock_gpu.h).
// These are internal executor guts and are not part of that contract.
//
// The per-operator entry points, pulled non-static so each operator can live in
// its own translation unit under src/operators/.

#include "peacock/plan_types.h"
#include "plan_executor.h"

#include <vector>

namespace peacock {

// Explicit input channel, threaded through the call chain. This REPLACES a pair of
// anonymous-namespace thread_locals: those were per-translation-unit, so splitting
// execute_one from execute_node would have made execute_node read a permanently-null
// copy and silently re-execute the whole subtree from parquet — correct answers,
// exponential cost, invisible to every correctness test. A parameter cannot fail that
// way, and it makes nesting natural (no RAII save/restore).
//
// `items == nullptr` means RECURSIVE mode: execute_node runs the child itself.
// Non-null means SINGLE-NODE mode: children are already resident and are consumed
// positionally, in the same post-order the caller pushed them.
struct NodeInputs {
  std::vector<TableResult>* items = nullptr;
  size_t idx = 0;
};

// GpuScan is a LEAF: it reads Parquet and takes no NodeInputs at all. (Every other
// operator does -- the dispatcher hands scans nothing to consume, which is exactly
// why execute_one's consume-at-least-one invariant excludes zero-input nodes.)
//
// Default argument on the DECLARATION only -- it sat on the definition before the
// split, which becomes a hard error once the declaration carries it too.
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

// The recursive driver. Every operator TU calls it to resolve its children, and so
// do execute_plan.cpp and node_session.cpp -- so this one genuinely has to be
// header-declared.
//
// Its companions run_op (the dispatch switch) and plan_node_kind_name stay STATIC in
// dispatch.cpp: nothing outside that TU calls them. Inc4a removed the thread_local
// pair run_op/execute_node/execute_one used to share, so they are no longer
// technically inseparable, but they remain co-located there on purpose -- they are
// one dispatch mechanism. Co-locating them never required exporting them.
TableResult execute_node(const fb::PlanNode* node, NodeInputs* in);

// `inputs` BY VALUE: execute_one owns them for the duration of the call and hands
// a NodeInputs pointing at that local down the chain.
TableResult execute_one(const fb::PlanNode* node, std::vector<TableResult> inputs);

// plan_node_kind_name is NOT declared here on purpose -- it is used only by run_op's
// trace inside dispatch.cpp, so it stays static there rather than being exported.

// Was a static one-liner inlined into run_op's passthrough branches. Kept `inline`
// so the split does not turn it into a real call through the .so --
// benchmarks/minimal.md measures per-node overhead.
inline TableResult execute_passthrough(const fb::PlanNode* input_node, NodeInputs* in) {
  return execute_node(input_node, in);
}

}  // namespace peacock
