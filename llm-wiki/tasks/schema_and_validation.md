# schema registry and validation (T7/T8 remainder)

**Goal.** Finish the two tasks whose implementations landed early on
`ENS-bp-plan-skeleton` but whose test surfaces did not: prove the `Schema` carried on every
node is right, and make `validate_schemas_and_partitions` a check that can go red for the
reasons it claims rather than only for the ones a real plan happens to hit.

**What already landed, so nobody rebuilds it.** `Schema` exists with `group_keys` and
`agg_state`, every node carries a populated one, the plan goldens print the declared type per
column, and node-local validation is called from `plan_batch_partitioned` with all ten
goldens passing it node by node. That half was pulled forward by a review finding — ten
guards existed and none ran on the live path.

## T7 — what the schema tests must show

- A hand-built plan carrying a project, an aggregate and a union produces the expected types
  and the expected semantics annotations, asserted on the tree rather than on rendered text.
- The annotations survive the aggregate sequence: `agg_state` is right at the init, at the
  per-lane merge and at the finalizing merge, which are three schemas for one logical
  aggregate.
- **Decimal precision and scale through project, aggregate and union-cast.** This is the
  one that earns the task: `avg`'s state columns were once typed backwards and per-node
  bytes could not show it, because both engines derive them from the same plan schema — so
  CPU and GPU agreed on the same wrong number and only a real divide would have diverged.

## T8 — what validation still owes

- The generic structural pass, over and above the per-node checks that exist.
- Manually constructed wrong combinations: each rule turned red by an input built to break
  it, per the reviewer's anchor that a guard which cannot go red is not a guard.
- Validation run over every canonized corpus plan as a standing check, not as a one-off.
- Defects in the checks themselves, including any the reviewer reported against
  `ENS-bp-plan-skeleton` and I deferred here.

**Constraints.** No plan changes. The ten goldens stay byte-identical: this task adds tests
and checks, and if a new check rejects a canonized plan that is a finding — either the check
or the plan is wrong — which stops and reports rather than regenerating. A test that only
passes is worth less than one shown to fail on the defect it guards.

**Verification bar.** Every rule in `validate_schemas_and_partitions` has an input that turns
it red. Decimal fidelity is asserted at each step of project → aggregate → union-cast. The
goldens do not move, and `test_ci_coverage` names whatever targets appear.
