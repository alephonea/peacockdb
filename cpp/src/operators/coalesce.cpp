// CudfCoalescePartitions -- k-way merge / concat of partitions.
//
// INTENTIONALLY EMPTY. Coalesce is not a free-standing operator: it is handled
// inline in NodeSession::execute_node (src/node_session.cpp, case
// fb::PlanNodeKind_CudfCoalescePartitions), which is where the session's partition
// registry and handle bookkeeping live. This file exists so the operator set
// stays discoverable by name.
