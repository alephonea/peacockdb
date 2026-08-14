// CudfRepartition -- hash repartition across partitions.
//
// INTENTIONALLY EMPTY. Repartition is not a free-standing operator: it is handled
// inline in NodeSession::execute_node (src/node_session.cpp, case
// fb::PlanNodeKind_CudfRepartition), which is where the session's partition
// registry and handle bookkeeping live. This file exists so the operator set
// stays discoverable by name.
