// Split out of the former src/plan_executor.cpp monolith.
//
// GpuCoalescePartitions -- k-way merge / concat of partitions.
//
// INTENTIONALLY (almost) EMPTY. Coalesce is not a free-standing operator: it is
// handled inline inside NodeSession::execute_node in src/node_session.cpp, where
// it needs the session's partition registry and handle bookkeeping. Lifting that
// block out to here would change the call structure rather than just move code,
// which is out of scope for a pure file split.
//
// This file exists so the operator set is discoverable by name -- looking for
// "where does Coalesce live" lands here and gets pointed at the real code.

// see src/node_session.cpp, NodeSession::execute_node -- case fb::PlanNodeKind_GpuCoalescePartitions
