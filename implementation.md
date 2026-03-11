# PeacockDB Implementation Plan

GPU-accelerated analytical database using Apache DataFusion as the SQL frontend and NVIDIA cuDF (libcudf) as the GPU execution engine.

---

## Phase 1: Project Scaffolding & Build System

### 1.1 DONE Rust Project Setup
- Initialize a Cargo workspace with the following crates:
  - `peacockdb` — the main binary crate (CLI entrypoint)
  - `peacockdb-core` — core library: plan translation, GPU executor orchestration
  - `peacockdb-ffi` — C++/Rust FFI boundary (CXX or raw `extern "C"`)

### 1.2 C++ Build Integration
- Add cuDF as a git submodule (or use CMake `FetchContent` / conda packages)
- Write a `CMakeLists.txt` for the C++ side that:
  - Links against `libcudf.so`, `librmm`, CUDA runtime
  - Builds a shared library `libpeacock_gpu.so` containing the GPU executor
- Use `build.rs` (with `cmake` crate or `cc` crate) to compile the C++ component and link it into the Rust binary
- Ensure the binary can be built with `cargo build` end-to-end

### 1.3 CLI Binary
- Use `clap` for argument parsing:
  ```
  peacockdb --data-dir /path/to/parquet/files --query "SELECT ..."
  ```
- Flags: `--data-dir` (directory of parquet files, each file = one table named after the file), `--query` (SQL string), optional `--gpu-memory-limit` (bytes)

### 1.4 CI: GitHub Actions + Verda GPU Instances

Two CI tiers: GitHub-hosted runners for build/CPU tests (free, every PR), and on-demand Verda GPU instances for GPU tests (paid, on merge or manual trigger).

#### 1.4.1 Tier 1: GitHub-Hosted Runners (Build + CPU Tests)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  build-and-test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Build (CPU-only, no CUDA)
        run: cargo build --workspace --features cpu-only
        # cpu-only feature flag compiles peacockdb-core and peacockdb
        # without peacockdb-ffi / C++ / CUDA dependencies

      - name: Run CPU tests
        run: cargo test --workspace --features cpu-only
        # Tests the plan IR round-trip, CPU executor, expression evaluation,
        # all scalar functions via DataFusion's built-in implementations
```

This runs on every push and PR. The `cpu-only` Cargo feature excludes `peacockdb-ffi` from compilation, so no CUDA toolkit or libcudf is needed. All tests use `CpuExecutor` (Phase 8).

#### 1.4.2 Tier 2: Verda GPU Instances (GPU Tests)

GPU tests run on a Verda instance provisioned on-demand. The critical design constraint: **Verda GPU instances bill per-minute, so the instance must be destroyed promptly, even if the test job crashes, times out, or is cancelled.**

```yaml
# .github/workflows/gpu-tests.yml
name: GPU Tests
on:
  push:
    branches: [master]
  workflow_dispatch:       # manual trigger for PRs that need GPU testing

env:
  VERDA_CLIENT_ID: ${{ secrets.VERDA_CLIENT_ID }}
  VERDA_CLIENT_SECRET: ${{ secrets.VERDA_CLIENT_SECRET }}
  VERDA_INSTANCE_TYPE: "1A100.8V"     # single A100, 8 vCPUs
  VERDA_IMAGE: "ubuntu-24.04-cuda-12.8-open-docker"
  MAX_GPU_TEST_MINUTES: 30            # hard timeout for the entire GPU job

jobs:
  gpu-test:
    runs-on: ubuntu-latest
    timeout-minutes: 45               # GitHub-level hard timeout (includes setup/teardown)
    steps:
      - uses: actions/checkout@v4

      - name: Install Verda SDK
        run: pip install verda

      - name: Provision GPU instance
        id: provision
        run: |
          python3 scripts/ci/verda_provision.py \
            --instance-type "$VERDA_INSTANCE_TYPE" \
            --image "$VERDA_IMAGE" \
            --timeout "$MAX_GPU_TEST_MINUTES" \
            | tee provision_output.txt
          echo "instance_id=$(cat provision_output.txt | grep INSTANCE_ID | cut -d= -f2)" >> $GITHUB_OUTPUT
          echo "instance_ip=$(cat provision_output.txt | grep INSTANCE_IP | cut -d= -f2)" >> $GITHUB_OUTPUT

      - name: Run GPU tests on remote instance
        timeout-minutes: 30
        run: |
          ssh -o StrictHostKeyChecking=no root@${{ steps.provision.outputs.instance_ip }} << 'REMOTE_EOF'
            set -e
            git clone --recursive ${{ github.server_url }}/${{ github.repository }} /workspace/peacockdb
            cd /workspace/peacockdb
            git checkout ${{ github.sha }}
            cargo build --workspace --release
            cargo test --workspace --release -- --test-threads=1
          REMOTE_EOF

      - name: Destroy GPU instance
        if: always()                   # CRITICAL: runs even if tests fail/cancel/timeout
        run: |
          python3 scripts/ci/verda_destroy.py \
            --instance-id "${{ steps.provision.outputs.instance_id }}"
```

#### 1.4.3 Verda Provisioning Script with Safety Guards

```python
# scripts/ci/verda_provision.py
"""
Provision a Verda GPU instance with multiple safety mechanisms
to prevent runaway billing.
"""
import os, sys, time, argparse
from verda import VerdaClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-type', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--timeout', type=int, default=30, help='Max lifetime in minutes')
    args = parser.parse_args()

    client = VerdaClient(
        os.environ['VERDA_CLIENT_ID'],
        os.environ['VERDA_CLIENT_SECRET']
    )

    # Upload CI SSH key if not already present
    ssh_keys = client.ssh_keys.get()
    ci_key_ids = [k.id for k in ssh_keys if k.name == 'peacockdb-ci']
    if not ci_key_ids:
        ci_key = client.ssh_keys.create(
            name='peacockdb-ci',
            public_key=os.environ['CI_SSH_PUBLIC_KEY']
        )
        ci_key_ids = [ci_key.id]

    # ── SAFETY GUARD 1: Startup script that self-destructs ──
    # The instance runs a background watchdog that kills itself
    # after --timeout minutes, regardless of what the CI job does.
    startup_script = f"""#!/bin/bash
    # Self-destruct timer: shutdown after {args.timeout} minutes
    (sleep {args.timeout * 60} && shutdown -h now) &
    echo $! > /tmp/watchdog.pid
    echo "Watchdog PID $(cat /tmp/watchdog.pid): will shutdown in {args.timeout}m"
    """

    instance = client.instances.create(
        instance_type=args.instance_type,
        image=args.image,
        ssh_key_ids=ci_key_ids,
        hostname=f'peacockdb-ci-{os.environ.get("GITHUB_RUN_ID", "manual")}',
        description=f'CI run, auto-destroy after {args.timeout}m',
        startup_script=startup_script,
    )

    # Wait for instance to be running
    for _ in range(60):  # max 5 minutes to boot
        status = client.instances.get_by_id(instance.id)
        if status.status == 'running':
            break
        time.sleep(5)
    else:
        # Boot timeout — destroy and fail
        client.instances.action(instance.id, client.actions.DELETE)
        sys.exit("ERROR: Instance failed to boot within 5 minutes")

    ip = status.ip
    print(f"INSTANCE_ID={instance.id}")
    print(f"INSTANCE_IP={ip}")

if __name__ == '__main__':
    main()
```

```python
# scripts/ci/verda_destroy.py
"""
Destroy a Verda instance. Called with `if: always()` so it runs
even when the test job fails, is cancelled, or times out.
"""
import os, sys, argparse
from verda import VerdaClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-id', required=True)
    args = parser.parse_args()

    client = VerdaClient(
        os.environ['VERDA_CLIENT_ID'],
        os.environ['VERDA_CLIENT_SECRET']
    )

    try:
        client.instances.action(args.instance_id, client.actions.DELETE)
        print(f"Instance {args.instance_id} destroyed successfully")
    except Exception as e:
        # Log but don't fail — the instance may already be gone
        # (self-destruct watchdog or manual cleanup)
        print(f"WARNING: Failed to destroy instance {args.instance_id}: {e}")

if __name__ == '__main__':
    main()
```

#### 1.4.4 Five Layers of GPU Billing Protection

Runaway Verda instances are the biggest CI cost risk. Five independent safety mechanisms ensure instances are always destroyed:

| Layer | Mechanism | Catches |
|---|---|---|
| **1. `if: always()` step** | `verda_destroy.py` runs after tests, unconditionally | Test failures, assertion errors, build failures |
| **2. GitHub `timeout-minutes`** | GitHub kills the job after 45 minutes, then `if: always()` still runs | Hung tests, infinite loops, SSH hangs |
| **3. Instance startup watchdog** | `shutdown -h now` after N minutes, baked into the instance's startup script | GitHub runner crashes, network partition, GitHub outage — the instance kills itself independently |
| **4. Scheduled reaper workflow** | Separate cron workflow that lists all Verda instances and deletes any older than 1 hour | All of the above fail simultaneously, orphaned instances from cancelled workflows |
| **5. Verda spending alert** | Configure Verda account billing alert at a monthly threshold (e.g., $500) | Sustained leak over many runs |

#### 1.4.5 Scheduled Reaper (Belt-and-Suspenders)

```yaml
# .github/workflows/verda-reaper.yml
name: Verda Instance Reaper
on:
  schedule:
    - cron: '*/15 * * * *'   # every 15 minutes

jobs:
  reap:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Verda SDK
        run: pip install verda

      - name: Destroy stale instances
        env:
          VERDA_CLIENT_ID: ${{ secrets.VERDA_CLIENT_ID }}
          VERDA_CLIENT_SECRET: ${{ secrets.VERDA_CLIENT_SECRET }}
        run: python3 scripts/ci/verda_reaper.py --max-age-minutes 60
```

```python
# scripts/ci/verda_reaper.py
"""
Kill any Verda instances older than --max-age-minutes.
Runs on a 15-minute cron schedule as a safety net.
"""
import os, argparse
from datetime import datetime, timezone, timedelta
from verda import VerdaClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-age-minutes', type=int, default=60)
    args = parser.parse_args()

    client = VerdaClient(
        os.environ['VERDA_CLIENT_ID'],
        os.environ['VERDA_CLIENT_SECRET']
    )

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.max_age_minutes)
    instances = client.instances.get()

    for inst in instances:
        # Only reap instances created by CI (hostname prefix)
        if not inst.hostname.startswith('peacockdb-ci-'):
            continue
        if inst.created_at < cutoff:
            print(f"REAPING stale instance {inst.id} "
                  f"(hostname={inst.hostname}, age={datetime.now(timezone.utc) - inst.created_at})")
            try:
                client.instances.action(inst.id, client.actions.DELETE)
            except Exception as e:
                print(f"WARNING: Failed to reap {inst.id}: {e}")

if __name__ == '__main__':
    main()
```

#### 1.4.6 Cost Control Summary

| Control | Value |
|---|---|
| Instance type | `1A100.8V` (single GPU, cheapest A100 option) |
| Max test duration | 30 minutes per run |
| Instance hard timeout | Startup watchdog kills after 30 minutes |
| Reaper frequency | Every 15 minutes |
| Reaper max age | 60 minutes |
| GPU tests trigger | `master` push, manual `workflow_dispatch`, or `/gpu-test` PR comment (see 1.5) |
| Monthly billing alert | Configure at $500 on Verda dashboard |
| Estimated cost per run | ~$1-2 (30 min × A100 spot rate) |

### 1.5 Code Review Workflow

Use **[Reviewable.io](https://reviewable.io)** for code review. It layers Gerrit-like review discipline on top of GitHub PRs — no self-hosted infrastructure needed. Free for public repositories.

#### 1.5.1 Why Reviewable

| GitHub PRs alone | Reviewable adds |
|---|---|
| Comments disappear when code changes ("outdated") | Comments track across revisions — stays on the logical line even after rebases/pushes |
| No way to know which revision a reviewer actually saw | File matrix shows exactly which revision each reviewer has reviewed |
| "Approve" is binary — no nuance | Disposition system: each discussion is explicitly `Resolved`, `Working`, `Pondering`, or `Blocking` |
| No enforcement that all discussions are resolved | Custom completion condition can block merge until all discussions are resolved and all files reviewed at latest revision |
| Reviewer has to re-read entire diff after each push | Reviewable shows incremental diffs between any two revisions (r2→r3, not just base→head) |

#### 1.5.2 Setup

1. **Install the Reviewable GitHub App** on the peacockdb repository (one-click from reviewable.io)
2. Reviewable automatically activates on every PR — a "Review on Reviewable" button appears
3. Configure branch protection rules on GitHub:
   - Require 1 approving review
   - Require `ci / build-and-test-cpu` status check to pass
   - Optionally require the Reviewable completion status check (see 1.5.3)

#### 1.5.3 Custom Completion Condition

Reviewable supports a programmable completion condition (JavaScript) that controls when the "review complete" status check passes. Configure in Reviewable's repository settings:

```javascript
// Reviewable completion condition
// Stored in repository settings on reviewable.io

// Require: at least 1 approval, all discussions resolved, all files reviewed at latest revision
const dominated = review.dominated;
const approvals = review.sentiments.filter(s => s === 'approved');
const pendingDiscussions = review.discussions.filter(
  d => d.disposition !== 'resolved' && d.disposition !== 'accepted'
);
const unreviewedFiles = review.files.filter(f => !f.reviewed);

return {
  completed: approvals.length >= 1
    && pendingDiscussions.length === 0
    && unreviewedFiles.length === 0,
  description: [
    approvals.length >= 1 ? null : 'Need at least 1 approval',
    pendingDiscussions.length === 0 ? null : `${pendingDiscussions.length} unresolved discussions`,
    unreviewedFiles.length === 0 ? null : `${unreviewedFiles.length} files not reviewed at latest revision`,
  ].filter(Boolean).join(', ') || 'Ready to merge',
  pendingReviewers: unreviewedFiles.length > 0
    ? review.assignees
    : [],
};
```

This gives Gerrit-like rigor: merge is blocked until every file has been reviewed at the latest revision and every discussion thread is marked resolved.

#### 1.5.4 `/gpu-test` Command: Trigger GPU Tests from a PR

A reviewer or author comments `/gpu-test` on the PR to trigger GPU tests for the current head commit. Implemented via GitHub Actions `issue_comment` trigger:

```yaml
# .github/workflows/gpu-tests.yml
name: GPU Tests
on:
  push:
    branches: [master]
  workflow_dispatch:
    inputs:
      commit_sha:
        description: 'Git commit SHA to test'
        required: true
  issue_comment:
    types: [created]

env:
  VERDA_CLIENT_ID: ${{ secrets.VERDA_CLIENT_ID }}
  VERDA_CLIENT_SECRET: ${{ secrets.VERDA_CLIENT_SECRET }}
  VERDA_INSTANCE_TYPE: "1A100.8V"
  VERDA_IMAGE: "ubuntu-24.04-cuda-12.8-open-docker"
  MAX_GPU_TEST_MINUTES: 30

jobs:
  gpu-test:
    # Only run for: master push, manual dispatch, or /gpu-test PR comment
    if: |
      github.event_name == 'push' ||
      github.event_name == 'workflow_dispatch' ||
      (github.event_name == 'issue_comment' &&
       github.event.issue.pull_request &&
       contains(github.event.comment.body, '/gpu-test'))
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - name: Determine commit to test
        id: commit
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          if [ "${{ github.event_name }}" = "issue_comment" ]; then
            # Get the PR head SHA at the time /gpu-test was posted
            PR_NUM="${{ github.event.issue.number }}"
            SHA=$(gh pr view "$PR_NUM" --repo "${{ github.repository }}" --json headRefOid -q .headRefOid)
            echo "sha=$SHA" >> $GITHUB_OUTPUT
            echo "pr_number=$PR_NUM" >> $GITHUB_OUTPUT
          elif [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            echo "sha=${{ github.event.inputs.commit_sha }}" >> $GITHUB_OUTPUT
          else
            echo "sha=${{ github.sha }}" >> $GITHUB_OUTPUT
          fi

      - name: Acknowledge /gpu-test request
        if: github.event_name == 'issue_comment'
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh pr comment "${{ steps.commit.outputs.pr_number }}" \
            --repo "${{ github.repository }}" \
            --body "GPU tests started for commit \`${{ steps.commit.outputs.sha }}\`. [Watch the run](${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }})"

      - uses: actions/checkout@v4
        with:
          ref: ${{ steps.commit.outputs.sha }}
          submodules: recursive

      - name: Install Verda SDK
        run: pip install verda

      - name: Provision GPU instance
        id: provision
        run: |
          python3 scripts/ci/verda_provision.py \
            --instance-type "$VERDA_INSTANCE_TYPE" \
            --image "$VERDA_IMAGE" \
            --timeout "$MAX_GPU_TEST_MINUTES" \
            | tee provision_output.txt
          echo "instance_id=$(grep INSTANCE_ID provision_output.txt | cut -d= -f2)" >> $GITHUB_OUTPUT
          echo "instance_ip=$(grep INSTANCE_IP provision_output.txt | cut -d= -f2)" >> $GITHUB_OUTPUT

      - name: Run GPU tests on remote instance
        id: tests
        timeout-minutes: 30
        run: |
          ssh -o StrictHostKeyChecking=no root@${{ steps.provision.outputs.instance_ip }} << 'REMOTE_EOF'
            set -e
            git clone --recursive ${{ github.server_url }}/${{ github.repository }} /workspace/peacockdb
            cd /workspace/peacockdb
            git checkout ${{ steps.commit.outputs.sha }}
            cargo build --workspace --release
            cargo test --workspace --release -- --test-threads=1
          REMOTE_EOF

      - name: Report results to PR
        if: always() && steps.commit.outputs.pr_number != ''
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          SHA="${{ steps.commit.outputs.sha }}"
          RUN_URL="${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"

          if [ "${{ steps.tests.outcome }}" = "success" ]; then
            BODY="GPU tests **PASSED** for commit \`${SHA}\`. [View run](${RUN_URL})"
            STATE="success"
          else
            BODY="GPU tests **FAILED** for commit \`${SHA}\`. [View run](${RUN_URL})"
            STATE="failure"
          fi

          # Post result as PR comment (visible in Reviewable)
          gh pr comment "${{ steps.commit.outputs.pr_number }}" \
            --repo "${{ github.repository }}" \
            --body "$BODY"

          # Also set a commit status so it shows in the PR checks
          gh api repos/${{ github.repository }}/statuses/${SHA} \
            -f state="$STATE" \
            -f target_url="$RUN_URL" \
            -f context="ci/gpu-tests" \
            -f description="GPU tests on Verda"

      - name: Destroy GPU instance
        if: always()
        run: |
          python3 scripts/ci/verda_destroy.py \
            --instance-id "${{ steps.provision.outputs.instance_id }}"
```

Key details:
- `/gpu-test` in any PR comment triggers the workflow
- The workflow posts an acknowledgement comment immediately ("GPU tests started for commit `abc123`...")
- On completion, posts a result comment with pass/fail and a link to the run — **this comment is visible in Reviewable** alongside code review discussions
- Also sets a `ci/gpu-tests` commit status on the exact SHA, so the check appears in the PR status section
- All five Verda billing safety layers (1.4.4) still apply

#### 1.5.5 Review Flow Summary

```
1. Developer opens PR on GitHub
   → CPU tests run automatically (GitHub Actions)
   → Reviewable activates, shows "Review on Reviewable" button

2. Reviewer opens PR in Reviewable
   → Sees file-by-file diff, leaves inline comments with dispositions
   → Marks files as reviewed (tracked per-revision)

3. Author pushes new commits addressing feedback
   → Reviewable shows incremental diff (r1→r2) so reviewer only re-reads what changed
   → Unreviewed files at new revision are highlighted

4. Reviewer wants GPU validation → comments /gpu-test on the PR
   → GitHub Actions provisions Verda A100, runs GPU tests at exact PR head SHA
   → Result posted as PR comment (visible in Reviewable) + commit status check

5. All files reviewed at latest revision, all discussions resolved, 1+ approval, CI green
   → Reviewable completion check passes → PR can be merged
```

#### 1.5.6 Reviewable Configuration Checklist

- [ ] Install Reviewable GitHub App on the repository
- [ ] Set custom completion condition (1.5.3) in Reviewable repo settings
- [ ] Enable GitHub branch protection: require `ci / build-and-test-cpu` + Reviewable status check
- [ ] Add `CODEOWNERS` file to enforce reviewer assignment
- [ ] Optionally require `ci/gpu-tests` status check for changes touching `peacockdb-ffi/` or C++ code

---

## Phase 2: DataFusion Frontend Integration

### 2.1 Table Registration
- Scan `--data-dir` for `.parquet` files
- For each file, register a DataFusion `TableProvider`:
  - Use DataFusion's built-in `ListingTable` / `ParquetFormat` to register tables
  - Table name derived from filename (e.g., `orders.parquet` → table `orders`)
- Create a `SessionContext` with all tables registered

### 2.2 Physical Plan Rewriting via `PhysicalOptimizerRule`

Rather than replacing the entire `QueryPlanner`, let DataFusion build its default physical plan (with `FilterExec`, `AggregateExec`, `HashJoinExec`, etc.), then use a custom `PhysicalOptimizerRule` to walk the tree bottom-up and replace CPU nodes with GPU equivalents. This approach is simpler and more maintainable — it reuses DataFusion's physical planning logic (partitioning, distribution enforcement, etc.) and only substitutes the execution kernels.

```rust
#[derive(Debug)]
struct GpuExecutionRule;

impl PhysicalOptimizerRule for GpuExecutionRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Walk tree bottom-up, replace nodes:
        //   FilterExec       → GpuFilterExec
        //   ProjectionExec   → GpuProjectExec
        //   AggregateExec    → GpuAggregateExec
        //   HashJoinExec        → GpuHashJoinExec / GpuSemiJoinExec
        //   SortMergeJoinExec  → GpuSortMergeJoinExec
        //   NestedLoopJoinExec → GpuNestedLoopJoinExec
        //   CrossJoinExec      → GpuCrossJoinExec
        //   SortExec         → GpuSortExec
        //   ParquetExec      → GpuScanExec (reads via cudf::io::read_parquet)
        plan.transform_up(|node| { /* match & replace */ })
    }

    fn name(&self) -> &str { "gpu_execution" }
    fn schema_check(&self) -> bool { true }
}
```

Register via:
```rust
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_physical_optimizer_rule(Arc::new(GpuExecutionRule))
    .build();
let ctx = SessionContext::new_with_state(state);
```

Each `GpuXxxExec` node implements the `ExecutionPlan` trait. In `execute()`, it serializes its subtree across the FFI boundary and invokes cuDF operations on the GPU.

### 2.3 Query Flow
1. Parse SQL → `LogicalPlan` (DataFusion)
2. Optimize `LogicalPlan` (DataFusion optimizer rules — reuse as-is)
3. Physical plan: `LogicalPlan` → default `ExecutionPlan` tree (DataFusion `DefaultPhysicalPlanner`)
4. GPU rewrite: `GpuExecutionRule` walks tree, replaces CPU nodes with `GpuXxxExec` nodes
4. Execute GPU plan → `RecordBatch` stream (Phase 4)
5. Print results to stdout (pretty-printed Arrow table)

---

## Phase 3: Physical Plan Translation (Rust → C++)

### 3.1 GPU Physical Plan IR
Define a serializable intermediate representation that crosses the Rust→C++ FFI boundary. This can be either:
- **Option A (recommended):** A Flatbuffers/Protobuf schema describing the plan tree, serialized on the Rust side, deserialized on the C++ side
- **Option B:** A C-compatible struct tree passed via FFI (more fragile but simpler for initial prototype)

The IR plan node types:

| IR Node | DataFusion LogicalPlan Node | cuDF Operation |
|---|---|---|
| `GpuScan` | `TableScan` | `cudf::io::read_parquet()` |
| `GpuFilter` | `Filter` | Evaluate expression → `cudf::apply_boolean_mask()` |
| `GpuProject` | `Projection` | Column selection via `table::select()` + `cudf::unary`/`binaryop` for computed columns |
| `GpuHashJoin` | `HashJoinExec` | `cudf::hash_join` (build) + `.inner_join()/.left_join()` (probe) + `cudf::gather()` to materialize |
| `GpuSortMergeJoin` | `SortMergeJoinExec` | Sort both sides (`cudf::sorted_order` + `gather`), then `cudf::merge()` + conditional scan for matching keys |
| `GpuNestedLoopJoin` | `NestedLoopJoinExec` | `cudf::conditional_inner_join()` / `conditional_left_join()` with AST predicate |
| `GpuCrossJoin` | `CrossJoinExec` | `cudf::cross_join()` |
| `GpuSemiJoin` | `HashJoinExec` (semi) | `cudf::left_semi_join()` / `cudf::left_anti_join()` + `cudf::gather()` |
| `GpuAggregate` | `Aggregate` | `cudf::groupby` + `aggregation_request` with `make_sum/min/max/count/mean_aggregation()` |
| `GpuSort` | `Sort` | `cudf::sorted_order()` + `cudf::gather()` |
| `GpuLimit` | `Limit` | `cudf::slice()` or limit row count during scan |
| `GpuUnion` | `Union` | `cudf::concatenate()` |

### 3.2 Expression Translation
DataFusion `Expr` → GPU expression representation:
- **Column references** → column index in the cudf table
- **Literals** → `cudf::scalar`
- **Binary ops** (`+`, `-`, `*`, `/`, `=`, `<`, `>`, `AND`, `OR`) → `cudf::binary_operation()` with appropriate `binary_operator` enum
- **Unary ops** (`NOT`, `IS NULL`, `IS NOT NULL`) → `cudf::unary_operation()`
- **Cast** → `cudf::cast()`
- **Aggregation functions** → `cudf::make_<agg>_aggregation()` (sum, min, max, count, mean, count_distinct→nunique, etc.)

### 3.3 FFI Layer
- Define C-compatible function:
  ```cpp
  // C++ side
  extern "C" ArrowDeviceArray* execute_gpu_plan(
      const uint8_t* plan_buf, size_t plan_len,
      GpuExecutorConfig config
  );
  ```
- Rust calls this via FFI, passing the serialized plan
- C++ deserializes the plan, walks the tree, executes cuDF ops bottom-up
- Result returned as `ArrowDeviceArray` (via cuDF's `to_arrow_host()`) for zero-copy (or minimal-copy) transfer back to Rust Arrow `RecordBatch`
- Use cuDF's Arrow interop: `cudf::to_arrow_host()` produces host-resident Arrow data that Rust can consume directly

### 3.4 Arrow Interop (Result Path)
- C++ produces `cudf::table` as final result
- Convert via `cudf::to_arrow_host()` → `ArrowDeviceArray` + `ArrowSchema`
- Rust consumes via `arrow-rs` `ffi` module: `ArrowArray::from_raw()` → `RecordBatch`
- Wrap in a `RecordBatchStream` to satisfy DataFusion's `ExecutionPlan::execute()` contract

---

## Phase 4: C++ GPU Executor with Pipelined I/O

### 4.1 Execution Model: Pipelined Chunks

The executor processes data in chunks. The critical insight is that loading the next chunk from disk and computing on the current chunk are independent — disk I/O doesn't use the GPU, and GPU compute doesn't use the disk. By overlapping them we keep both resources busy.

```
Time ──────────────────────────────────────────────────►

CPU/IO thread:  ┌─Load C0─┐ ┌─Load C1─┐ ┌─Load C2─┐ ┌─Load C3─┐
                │ parquet  │ │ parquet  │ │ parquet  │ │ parquet  │
                └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
                     │            │            │            │
                     ▼ H→D       ▼ H→D       ▼ H→D       ▼ H→D
GPU stream:          ┌──Exec C0──┐ ┌──Exec C1──┐ ┌──Exec C2──┐ ┌──Exec C3──┐
                     │ filter/   │ │ filter/   │ │ filter/   │ │           │
                     │ project/  │ │ project/  │ │ project/  │ │           │
                     │ agg       │ │ agg       │ │ agg       │ │           │
                     └───────────┘ └───────────┘ └───────────┘ └───────────┘

Without pipelining:  [Load C0][Exec C0][Load C1][Exec C1][Load C2][Exec C2] ...
With pipelining:     [Load C0][Load C1][Load C2][Load C3]
                              [Exec C0][Exec C1][Exec C2][Exec C3]
                     ← saves ~one load latency per chunk after the first ─→
```

### 4.2 Pipeline Architecture

```cpp
class PipelinedExecutor {
    /// Background I/O: reads parquet chunks, uploads to GPU staging buffer
    std::thread io_thread_;

    /// Double-buffer: while GPU processes buffer[i], I/O fills buffer[1-i]
    std::array<std::unique_ptr<cudf::table>, 2> staging_buffers_;
    int active_buf_ = 0;

    /// Synchronization
    std::mutex mtx_;
    std::condition_variable chunk_ready_;   // I/O → GPU: "next chunk loaded"
    std::condition_variable chunk_consumed_; // GPU → I/O: "buffer is free, load next"
    bool io_done_ = false;

    /// CUDA stream for async H→D transfers
    rmm::cuda_stream compute_stream_;
};
```

#### 4.2.1 I/O Thread

The I/O thread runs on CPU, reading parquet chunks and preparing them for GPU consumption:

```cpp
void io_thread_func(PipelinedExecutor* exec, const ScanNode& scan) {
    auto reader = cudf::io::chunked_parquet_reader(
        scan.chunk_size_bytes,
        cudf::io::parquet_reader_options::builder(source_info(scan.path))
            .columns(scan.projected_columns)
            .build()
    );

    while (reader.has_next()) {
        auto chunk = reader.read_chunk();  // CPU: decompress parquet → host memory

        // Wait until the staging buffer is free
        {
            std::unique_lock lock(exec->mtx_);
            exec->chunk_consumed_.wait(lock, [&]{
                return exec->staging_buffers_[1 - exec->active_buf_] == nullptr;
            });
        }

        // Place chunk in the inactive staging buffer
        {
            std::lock_guard lock(exec->mtx_);
            exec->staging_buffers_[1 - exec->active_buf_] = std::move(chunk.tbl);
        }
        exec->chunk_ready_.notify_one();
    }

    // Signal completion
    {
        std::lock_guard lock(exec->mtx_);
        exec->io_done_ = true;
    }
    exec->chunk_ready_.notify_one();
}
```

#### 4.2.2 GPU Execution Loop

The main thread consumes chunks from the staging buffer and runs the operator pipeline:

```cpp
std::unique_ptr<cudf::table> PipelinedExecutor::execute(const PlanNode& plan) {
    // Start I/O thread for the leaf scan
    io_thread_ = std::thread(io_thread_func, this, plan.find_scan());

    std::vector<std::unique_ptr<cudf::table>> partial_results;

    while (true) {
        // Wait for next chunk
        std::unique_ptr<cudf::table> chunk;
        {
            std::unique_lock lock(mtx_);
            chunk_ready_.wait(lock, [&]{
                return staging_buffers_[1 - active_buf_] != nullptr || io_done_;
            });
            if (staging_buffers_[1 - active_buf_] == nullptr && io_done_) break;

            // Swap buffers: GPU takes the loaded chunk, I/O can fill the other
            active_buf_ = 1 - active_buf_;
            chunk = std::move(staging_buffers_[active_buf_]);
        }
        chunk_consumed_.notify_one();

        // Execute the pipeline of stateless operators on this chunk
        auto result = execute_pipeline(plan, std::move(chunk));
        partial_results.push_back(std::move(result));
    }

    io_thread_.join();
    return merge_partial_results(plan, std::move(partial_results));
}
```

### 4.3 Operator Classification: Pipelineable vs. Pipeline-Breaking

Not all operators can process data chunk-by-chunk. The pipeline must be broken at certain boundaries:

| Operator | Pipelineable? | Behavior |
|---|---|---|
| **Scan** | Source | Produces chunks — this is the pipeline source |
| **Filter** | Yes | Stateless: apply predicate to each chunk independently |
| **Project** | Yes | Stateless: evaluate expressions per chunk |
| **Limit** | Yes | Track running row count, stop when limit reached, signal I/O to stop early |
| **Aggregate** | Partial | Partial aggregation per chunk (pipelineable), final merge at end (pipeline-breaking) |
| **Hash Join (probe side)** | Yes | Build side must be fully materialized first; then probe chunks stream through |
| **Hash Join (build side)** | Breaking | Must accumulate all build-side chunks before probe begins |
| **Sort-Merge Join** | Breaking | Both sides must be fully sorted first; then merge-scan is pipelineable but depends on sorted input |
| **Nested Loop Join** | Breaking | Must materialize the right (inner) side; left side can stream chunk-by-chunk calling `conditional_*_join` per chunk |
| **Cross Join** | Breaking | Must materialize one side; other side streams through `cudf::cross_join()` per chunk |
| **Semi/Anti Join** | Mixed | Build hash set from right side (breaking), then `left_semi_join`/`left_anti_join` per left chunk (pipelineable) |
| **Sort** | Breaking | Needs all data before sorting. Each chunk sorted independently, then k-way merge |
| **Union** | Yes | Chunks from each input concatenated in sequence |

#### Pipeline Segments
The executor splits the plan tree into **pipeline segments** at breaking boundaries:

```
Example: SELECT * FROM A JOIN B ON ... WHERE ... ORDER BY ...

Pipeline 1 (build side):  Scan(B) ──chunk──► [accumulate into hash table]
Pipeline 2 (probe side):  Scan(A) ──chunk──► Filter ──► HashJoin(probe) ──► [accumulate]
Pipeline 3 (sort):        [accumulated] ──► Sort ──► output

Pipeline 1 runs first (I/O for B pipelined with hash table building).
Pipeline 2 runs next (I/O for A pipelined with filter + probe).
Pipeline 3 runs last on the accumulated join result.
```

### 4.4 Pipeline Construction

```cpp
/// A pipeline segment: a chain of pipelineable operators fed by a source.
struct PipelineSegment {
    ScanSource source;                      // chunked_parquet_reader or materialized table
    std::vector<PipelineableOp*> operators; // filter, project, probe-side join, ...
    PipelineSink sink;                      // accumulate, partial-aggregate, or output
};

/// Split the plan tree into pipeline segments at breaking boundaries.
std::vector<PipelineSegment> build_pipelines(const PlanNode& root) {
    // Walk plan bottom-up.
    // At each pipeline-breaking node, emit a segment for everything below it,
    // then start a new segment with the breaking node's output as source.
}
```

### 4.5 CUDA Stream Overlap

For even finer-grained pipelining within a single chunk, use separate CUDA streams:

```cpp
rmm::cuda_stream io_stream;      // for H→D transfers (cudf::from_arrow_host)
rmm::cuda_stream compute_stream; // for cuDF compute kernels

// While GPU computes on chunk N, the H→D transfer for chunk N+1 runs concurrently:
// 1. I/O thread reads parquet → host memory (CPU, no stream)
// 2. I/O thread enqueues H→D copy on io_stream
// 3. GPU executes operators on compute_stream
// 4. Synchronize io_stream before compute_stream starts on chunk N+1
```

This gives three-stage pipelining: disk→host (CPU), host→device (io_stream), compute (compute_stream).

### 4.6 Operator Implementations

#### Scan (`GpuScan`)
No longer a standalone operator — it becomes the `PipelineSegment::source`. The I/O thread drives `chunked_parquet_reader` and feeds chunks into the pipeline.

For small tables that fit in GPU memory, a single non-chunked `read_parquet()` is used as an optimization.

#### Filter (`GpuFilter`)
```cpp
// Pipelineable: processes one chunk at a time
std::unique_ptr<cudf::table> execute(cudf::table_view chunk) {
    auto mask = evaluate_expression(chunk, predicate_expr);
    return cudf::apply_boolean_mask(chunk, mask);
}
```

#### Projection (`GpuProject`)
- Simple column selection: `table.select(column_indices)`
- Computed columns: evaluate expression tree using `cudf::binary_operation()`, `cudf::unary_operation()`, then assemble new table from result columns

#### Join Strategies

DataFusion selects the join strategy during physical planning based on join type, predicate structure, and table statistics. The `GpuExecutionRule` (Phase 2.2) maps each DataFusion join node to the corresponding GPU join operator. We support four strategies:

##### Hash Join (`GpuHashJoin`)
**When used:** Equi-joins (`a.key = b.key`) — the most common case. DataFusion emits `HashJoinExec`.

Supports all join types: `Inner`, `Left`, `Right`, `Full`, `Semi`, `Anti`.

Splits into two pipeline phases:
```cpp
// Phase 1: Build side — pipeline-breaking, accumulates into hash table
void build(std::vector<std::unique_ptr<cudf::table>> build_chunks) {
    auto build_table = cudf::concatenate(build_chunks);
    hasher_ = std::make_unique<cudf::hash_join>(
        build_table.select(key_indices), null_equality::EQUAL);
    build_table_ = std::move(build_table);
}

// Phase 2: Probe side — pipelineable, each probe chunk processed independently
std::unique_ptr<cudf::table> probe(cudf::table_view probe_chunk) {
    auto probe_keys = probe_chunk.select(key_indices);
    auto [left_map, right_map] = hasher_->inner_join(probe_keys);
    auto left_result = cudf::gather(build_table_, left_map);
    auto right_result = cudf::gather(probe_chunk, right_map);
    // combine columns into output
}
```

For **semi/anti joins** (`WHERE x IN (subquery)`, `WHERE NOT EXISTS ...`), the probe phase uses:
```cpp
// Semi join: return left rows that have a match
auto left_map = cudf::left_semi_join(probe_keys, build_keys);
auto result = cudf::gather(probe_chunk, left_map);

// Anti join: return left rows that have NO match
auto left_map = cudf::left_anti_join(probe_keys, build_keys);
auto result = cudf::gather(probe_chunk, left_map);
```

When the build side has unique keys (e.g., joining on a primary key), use `cudf::distinct_hash_join` for a faster code path.

##### Sort-Merge Join (`GpuSortMergeJoin`)
**When used:** DataFusion emits `SortMergeJoinExec` when both inputs are already sorted on the join key (e.g., from an upstream `ORDER BY` or index), or when the optimizer estimates that sort-merge is cheaper than hash join (both sides very large, no good build-side candidate).

```cpp
// 1. Ensure both sides are sorted on join keys
//    (skip if inputs are already sorted — check PlanProperties)
auto left_sorted = ensure_sorted(left_table, left_key_indices);
auto right_sorted = ensure_sorted(right_table, right_key_indices);

// 2. Merge-join scan: walk both sorted tables with two cursors
//    cuDF doesn't have a native merge-join, so implement as:
//    a) Concatenate keys from both sides with a side tag column
//    b) cudf::merge() the two sorted key sets
//    c) Identify matching key groups from the merged output
//    d) cudf::gather() to materialize matching rows

// Alternative: use cudf::conditional_inner_join() with an equality AST
//    expression — cuDF's conditional join handles arbitrary predicates
//    and may be competitive for sorted inputs.
```

Pipeline behavior: both sides must be sorted first (pipeline-breaking). Once sorted, the merge scan itself is sequential.

##### Nested Loop Join (`GpuNestedLoopJoin`)
**When used:** DataFusion emits `NestedLoopJoinExec` for non-equi joins where the predicate can't be decomposed into equality conditions (e.g., `a.x < b.y`, range joins, theta joins). Also used for very small tables where hash overhead isn't worth it.

```cpp
// cuDF's conditional join API handles arbitrary predicates via AST expressions
auto predicate = translate_to_cudf_ast(join_predicate);

auto [left_map, right_map] = cudf::conditional_inner_join(
    left_table, right_table, predicate);
auto left_result = cudf::gather(left_table, left_map);
auto right_result = cudf::gather(right_table, right_map);
```

cuDF variants: `conditional_left_join`, `conditional_full_join`, `conditional_left_semi_join`, `conditional_left_anti_join`.

For joins with **both** equality and non-equality predicates (e.g., `a.key = b.key AND a.ts < b.ts`), use cuDF's **mixed join** API which combines hash lookup with conditional evaluation:
```cpp
auto [left_map, right_map] = cudf::mixed_inner_join(
    left_equality_keys, right_equality_keys,
    left_conditional_cols, right_conditional_cols,
    inequality_predicate);
```

Pipeline behavior: materialize the right (inner) side, then stream left chunks through `conditional_*_join()` per chunk.

##### Cross Join (`GpuCrossJoin`)
**When used:** DataFusion emits `CrossJoinExec` for `CROSS JOIN` or implicit cartesian products (comma-separated `FROM` without `WHERE`).

```cpp
auto result = cudf::cross_join(left_table, right_table);
```

Pipeline behavior: materialize one side (the smaller one). Stream the other side through `cudf::cross_join()` per chunk. **Warning:** output size = left_rows × right_rows — the memory estimator (Phase 5) must be especially conservative here.

##### Strategy Selection Summary

| DataFusion node | GPU operator | cuDF API | Pipeline behavior |
|---|---|---|---|
| `HashJoinExec` (equi, inner/left/right/full) | `GpuHashJoin` | `cudf::hash_join` | Build=breaking, probe=pipelineable |
| `HashJoinExec` (semi/anti) | `GpuSemiJoin` | `cudf::left_semi_join` / `left_anti_join` | Build=breaking, probe=pipelineable |
| `SortMergeJoinExec` | `GpuSortMergeJoin` | Sort + merge scan (or `conditional_*_join`) | Both sides breaking (sort), then sequential |
| `NestedLoopJoinExec` (non-equi) | `GpuNestedLoopJoin` | `cudf::conditional_*_join` | Right=breaking, left=pipelineable |
| `NestedLoopJoinExec` (mixed predicates) | `GpuNestedLoopJoin` | `cudf::mixed_*_join` | Right=breaking, left=pipelineable |
| `CrossJoinExec` | `GpuCrossJoin` | `cudf::cross_join` | One side breaking, other pipelineable |

#### Aggregate (`GpuAggregate`)
Two-phase pipeline:
```cpp
// Pipelineable: partial aggregation per chunk
std::unique_ptr<cudf::table> partial_aggregate(cudf::table_view chunk) {
    cudf::groupby::groupby gb(chunk.select(group_key_indices));
    auto [keys, results] = gb.aggregate(requests);
    return combine_keys_and_results(keys, results);
}

// Pipeline-breaking: merge all partial results
std::unique_ptr<cudf::table> final_merge(
    std::vector<std::unique_ptr<cudf::table>> partials) {
    auto combined = cudf::concatenate(partials);
    cudf::groupby::groupby gb(combined.select(group_key_indices));
    auto [keys, results] = gb.aggregate(merge_requests); // MERGE_SUM, MERGE_M2, etc.
    return combine_keys_and_results(keys, results);
}
```

#### Sort (`GpuSort`)
Pipeline-breaking. Accumulates all input, then:
```cpp
auto sorted_indices = cudf::sorted_order(input, column_order, null_precedence);
auto sorted_table = cudf::gather(input, sorted_indices);
```
When input is chunked: sort each chunk independently, then `cudf::merge()` the sorted chunks.

#### Limit (`GpuLimit`)
```cpp
// Pipelineable with early termination
std::unique_ptr<cudf::table> execute(cudf::table_view chunk) {
    if (rows_emitted_ >= limit_) {
        signal_io_stop();  // tell I/O thread to stop reading
        return nullptr;
    }
    size_t take = std::min(chunk.num_rows(), limit_ - rows_emitted_);
    rows_emitted_ += take;
    return std::make_unique<cudf::table>(cudf::slice(chunk, {0, take})[0]);
}
```

### 4.7 Multi-Scan Pipelining

Queries with multiple table scans (e.g., joins) require coordinating multiple I/O streams:

```
Query: SELECT ... FROM orders o JOIN lineitem l ON ...

Phase 1 — Build pipeline (I/O for smaller table):
  I/O thread 1: reads lineitem.parquet chunk by chunk
  GPU: accumulates chunks into hash_join build table

Phase 2 — Probe pipeline (I/O for larger table):
  I/O thread 1: reads orders.parquet chunk by chunk
  GPU: filter → hash_join probe per chunk

Both phases internally pipeline I/O with GPU via double-buffering.
```

For joins where both sides are large, the build side itself may need to be partitioned (Phase 5.4). Each partition's build+probe pair is a separate pipeline run.

### 4.8 Interaction with Memory Management

The pipelining directly helps memory management:
- Only 2 chunks are on GPU at any time (double-buffer), not the full table
- Partial aggregation reduces data volume inside the pipeline before accumulation
- `Limit` with early termination avoids reading the entire table
- The chunk size is controlled by the memory budget (Phase 5) and adaptive estimator (Phase 5.5.3)

---

## Phase 5: GPU Memory Management, Chunking & OOM Recovery

### 5.1 Memory Budget Tracker
- Query available GPU memory via `cudaMemGetInfo()`
- Maintain a running estimate of memory usage across all live `cudf::table` objects
- Configurable memory limit (default: 80% of free GPU memory)
- Track actual vs. estimated sizes for every intermediate result (feeds the adaptive estimator in 5.5)

### 5.2 Cardinality Estimation
For each intermediate relation, estimate output size to determine if chunking is needed:

- **Scan**: Row count from parquet metadata (no GPU needed), column widths from schema
- **Filter**: Selectivity estimation from column statistics (min/max/null_count) + HLL for distinct counts
- **Join**: Use `hash_join::inner_join_size()` / `left_join_size()` / `full_join_size()` for exact output size estimation before materializing
- **Aggregate**: HLL on group-by keys estimates number of output groups
- **Sort/Limit**: Output size = input size (sort), or min(input, N) (limit)

### 5.3 Statistics Collection
After producing each intermediate table on the GPU, compute statistics:
```cpp
struct TableStats {
    size_t row_count;
    size_t memory_bytes;          // actual GPU memory footprint
    per_column: {
        scalar min, max;
        size_t null_count;
        HyperLogLog hll;          // for distinct count estimation
    }
};
```
- Use `cudf::reduction` with `make_min/max_aggregation()` for min/max
- Null count from column's `null_count()`
- HLL: implement via cuDF's `cudf::make_nunique_aggregation()` or a custom HLL kernel
- Feed stats back into cardinality estimator for downstream operators

### 5.4 Chunking Strategy (Proactive)
When estimated intermediate size exceeds memory budget:

1. **Scan chunking**: Use `chunked_parquet_reader` to read N rows at a time
2. **Filter/Project**: Process chunk-by-chunk (stateless — each chunk independent)
3. **Hash Join**:
   - Partition both sides by join key hash into K partitions
   - Process each partition pair sequentially on GPU
   - Concatenate results
4. **Aggregate**:
   - Partial aggregation per chunk (produces partial results)
   - Final merge aggregation across partial results
   - cuDF supports merge aggregations: `MERGE_LISTS`, `MERGE_SETS`, `MERGE_M2` etc.
5. **Sort**:
   - Sort each chunk independently
   - K-way merge sorted chunks (use `cudf::merge()`)

### 5.5 OOM Recovery & Adaptive Re-estimation

Estimates can be wrong. When a cuDF call triggers a CUDA OOM (`cudaErrorMemoryAllocation` or RMM `out_of_memory` exception), the executor must recover rather than crash.

#### 5.5.1 OOM Detection
- Wrap every cuDF call that allocates GPU memory in a try/catch for `rmm::out_of_memory` and `cudf::cuda_error`
- On catch, the operator enters **recovery mode** — it does NOT propagate the error

#### 5.5.2 Recovery Protocol
When OOM is caught at operator `Op_i` processing input chunk `C_j`:

1. **Free the failed allocation** — the exception already guarantees no partial result was produced
2. **Spill live intermediates**: If there are other intermediate `cudf::table` objects on GPU that are not currently needed by `Op_i`, convert them to host memory via `cudf::to_arrow_host()` and release the GPU copies. Mark them as spilled; they will be re-uploaded on demand via `cudf::from_arrow_host()`.
3. **Halve the chunk size**: Set `chunk_rows = chunk_rows / 2` for the current operator's input
4. **Retry** with the smaller chunk
5. If OOM recurs even at the minimum chunk size (e.g., a single row group), escalate: spill ALL non-essential GPU state, then retry once more. If that still fails, propagate the error to the user with a diagnostic message showing the memory breakdown.

```
┌──────────────────────────────────────────────────┐
│               Operator executes                  │
│                     │                            │
│              cuDF call fails OOM                 │
│                     │                            │
│         ┌───────────▼──────────────┐             │
│         │  Spill other intermediates│             │
│         │  to host memory          │             │
│         └───────────┬──────────────┘             │
│                     │                            │
│         ┌───────────▼──────────────┐             │
│         │  Halve chunk_rows        │             │
│         │  for this operator       │             │
│         └───────────┬──────────────┘             │
│                     │                            │
│              Retry cuDF call                     │
│                     │                            │
│              ┌──────┴──────┐                     │
│           Success        OOM again               │
│              │              │                    │
│     Record actual size   At min chunk?           │
│     Update estimates     ┌──┴──┐                 │
│                        No     Yes                │
│                         │      │                 │
│                   Halve again  Propagate error    │
└──────────────────────────────────────────────────┘
```

#### 5.5.3 Adaptive Estimation from Partial Output

The key insight: when processing data in chunks, early chunks produce actual output that can correct estimates for remaining chunks.

For each operator, after successfully producing output for chunk `C_j`:

1. **Record the expansion ratio**:
   ```
   actual_ratio_j = output_rows_j / input_rows_j       (for filter: selectivity)
   actual_ratio_j = output_bytes_j / input_bytes_j     (for join: byte expansion)
   ```

2. **Update the running estimate** using an exponential moving average over chunks processed so far:
   ```
   estimated_ratio = α * actual_ratio_j + (1 - α) * estimated_ratio_prev
   ```
   Use `α = 0.3` — recent chunks weighted more since data distributions may be localized, but don't overfit to a single chunk.

3. **Re-derive chunk size** for remaining chunks:
   ```
   max_output_bytes = gpu_memory_budget - currently_live_bytes
   safe_input_rows = max_output_bytes / (avg_row_width * estimated_ratio)
   next_chunk_rows = safe_input_rows * safety_factor    (safety_factor = 0.8)
   ```

4. **Per-operator specifics**:

   | Operator | What to measure from early chunks | How it adjusts |
   |---|---|---|
   | **Filter** | Selectivity (`output_rows / input_rows`) | Higher selectivity → output nearly as large as input → smaller chunks needed |
   | **Join** | Row expansion ratio, or use `hash_join::inner_join_size()` on chunk keys before materializing | Fanout > 1× means output bigger than input → reduce chunk size proportionally |
   | **Aggregate** | Group count vs. input rows (reduction ratio) | Low reduction (many groups) → output nearly as large → smaller chunks; high reduction → can use bigger chunks |
   | **Sort** | No expansion, but sort needs ~2× memory (input + sorted copy + index array) | If OOM on sort, halve input; the 2× overhead is fixed |
   | **Expression eval** | Number of intermediate columns materialized | Deep expression trees create many temp columns; track peak column count and adjust |

#### 5.5.4 Spill Manager
Manages host-memory copies of GPU intermediates:

```cpp
class SpillManager {
    struct SpilledTable {
        ArrowDeviceArray host_data;   // host-resident Arrow data
        ArrowSchema schema;
        size_t gpu_bytes;             // how much GPU memory was freed
    };

    std::unordered_map<TableId, SpilledTable> spilled_;
    size_t total_spilled_bytes_ = 0;

    // Spill a table from GPU to host, freeing GPU memory
    void spill(TableId id, cudf::table&& gpu_table);

    // Reload a spilled table back to GPU
    std::unique_ptr<cudf::table> reload(TableId id);

    // Spill the largest non-active table to free GPU memory
    size_t spill_largest(TableId exclude);
};
```

- Each live intermediate `cudf::table` in the executor gets a `TableId`
- The executor marks which tables are "active" (currently being read by an operator)
- On OOM, the spill manager picks the largest non-active table, calls `cudf::to_arrow_host()`, releases the GPU table, and stores the host copy
- When that table is needed again, `reload()` calls `cudf::from_arrow_host()` to bring it back

#### 5.5.5 Chunk Size Floor & CPU Fallback Escalation
- Minimum chunk size: 1 parquet row group (cannot split further without re-encoding)
- If a single row group causes OOM even after spilling everything else, **fall back to the CPU executor** (Phase 8) for that operator subtree rather than failing the query:
  1. The failing operator and its inputs are re-executed on CPU using the same plan IR
  2. The result is produced as an Arrow `RecordBatch` on host memory
  3. Downstream operators can continue on GPU (the CPU result is uploaded via `cudf::from_arrow_host()`) or also fall back to CPU if needed
  4. Log a warning: `"Operator X fell back to CPU execution: row group N of table T requires X MB but only Y MB available on GPU"`
- A query can run in **mixed mode**: some operators on GPU, some on CPU, as long as the plan IR is the same
- Log all OOM events and the adaptive estimate corrections for debugging

### 5.6 Disk Overflow Execution

When intermediate data exceeds both GPU memory and host memory — e.g., a join between two 100GB tables on a machine with 16GB GPU / 64GB RAM — the executor must spill to disk. This extends the spill hierarchy to four tiers:

```
Tier 0: GPU memory        (fast, limited — 8-80 GB)
Tier 1: Host memory       (medium, larger — 32-512 GB)
Tier 2: Local disk (SSD)  (slow, very large — 1+ TB)
Tier 3: CPU fallback      (last resort — uses DataFusion, can also spill to disk via its own mechanisms)
```

#### 5.6.1 Spill File Format

All disk-spilled data is written as **temporary IPC (Arrow Feather v2) files** in a configurable scratch directory (`--spill-dir`, default: `$TMPDIR/peacockdb-spill/`):

```cpp
struct SpillFile {
    std::filesystem::path path;        // e.g., /tmp/peacockdb-spill/q42_part_07.arrow
    size_t num_rows;
    size_t num_bytes_on_disk;          // compressed size
    ArrowSchema schema;                // remembers column types
    PartitionId partition;             // which hash partition this belongs to (for joins/aggs)
};
```

- Arrow IPC is used because it's self-describing, supports zero-copy reads via memory mapping, and both cuDF (`cudf::io::read_arrow_ipc`) and arrow-rs can read it directly
- Optional LZ4 compression reduces I/O volume at minimal CPU cost
- Files are cleaned up when the query completes (or on crash, via a reaper on next startup)

```cpp
class DiskSpillManager {
    std::filesystem::path spill_dir_;
    size_t total_spilled_bytes_ = 0;
    std::vector<SpillFile> spill_files_;

    /// Write a cudf::table to disk as Arrow IPC, freeing GPU/host memory
    SpillFile spill_to_disk(TableId id, cudf::table&& gpu_table);

    /// Write host-resident Arrow data to disk
    SpillFile spill_to_disk(TableId id, ArrowDeviceArray&& host_data, ArrowSchema schema);

    /// Read a spill file back into GPU memory
    std::unique_ptr<cudf::table> reload_to_gpu(const SpillFile& file);

    /// Read a spill file into host memory (for CPU fallback)
    std::vector<RecordBatch> reload_to_host(const SpillFile& file);

    /// Cleanup all spill files
    void cleanup();
};
```

#### 5.6.2 Disk-Overflow Hash Join (Grace Hash Join)

When the build side of a hash join exceeds GPU + host memory, use **Grace hash join** — a classic disk-based join algorithm adapted for GPU:

```
1. PARTITION phase (both sides):
   Read input in chunks → hash partition each chunk into K buckets → write each bucket to a spill file

2. JOIN phase (per partition):
   For partition i: read build_i from disk → load to GPU → build hash table
                    read probe_i from disk → stream chunks through GPU probe
                    write output to result (or next pipeline stage)
```

Implementation:

```cpp
class GraceHashJoin {
    size_t num_partitions_;    // K — chosen so each partition fits in GPU memory
    std::vector<SpillFile> build_partitions_;
    std::vector<SpillFile> probe_partitions_;

    /// Phase 1: Partition both inputs to disk
    void partition_to_disk(PipelineSource& build_source,
                           PipelineSource& probe_source,
                           const std::vector<size_type>& key_indices) {
        // For each input chunk:
        //   1. Compute hash of join keys: cudf::hash(chunk.select(key_indices))
        //   2. Compute partition assignment: hash_col % K
        //   3. cudf::partition(chunk, partition_assignment, K)
        //   4. Append each partition slice to its spill file
    }

    /// Phase 2: Join each partition pair on GPU
    std::vector<std::unique_ptr<cudf::table>> join_partitions() {
        std::vector<std::unique_ptr<cudf::table>> results;
        for (size_t i = 0; i < num_partitions_; i++) {
            auto build_table = spill_mgr_.reload_to_gpu(build_partitions_[i]);
            cudf::hash_join hasher(build_table.select(key_indices), ...);

            // Stream probe partition through in chunks (pipelined I/O)
            auto probe_reader = open_ipc_reader(probe_partitions_[i]);
            while (probe_reader.has_next()) {
                auto probe_chunk = probe_reader.read_chunk();  // disk → host
                auto gpu_chunk = cudf::from_arrow_host(probe_chunk); // host → GPU
                auto result = probe_with(hasher, gpu_chunk);
                results.push_back(std::move(result));
            }
        }
        return results;
    }
};
```

**Partition count selection**:
```
K = ceil(estimated_build_size / (gpu_memory_budget * 0.6))
```
The 0.6 factor reserves space for the hash table overhead and the probe chunk. If a single partition is still too large (skewed keys), recursively re-partition that partition with a different hash seed.

**Partition function**: Use `cudf::hash(keys, cudf::hash_id::MURMURHASH3)` followed by `cudf::partition()` which physically splits the table by partition index. This is the same hash function cuDF uses internally for its hash joins, ensuring consistent key distribution.

#### 5.6.3 Disk-Overflow Aggregation (Hash Aggregation with Spill)

When partial aggregation results across all chunks don't reduce enough and accumulate beyond memory:

```
1. PARTITION phase:
   Read input in chunks → partial aggregate each chunk on GPU
   → hash-partition the partial results by group keys into K buckets
   → spill each bucket to disk

2. MERGE phase (per partition):
   For partition i: read all partial results for partition i from disk
   → load to GPU → final merge aggregation
   → output
```

```cpp
class SpillingAggregation {
    size_t num_partitions_;
    std::vector<std::vector<SpillFile>> partition_files_; // partition_files_[i] = partial results for partition i

    /// Phase 1: Partial aggregate + partition + spill
    void ingest(cudf::table_view chunk) {
        // 1. Partial aggregation on GPU
        cudf::groupby::groupby gb(chunk.select(group_key_indices));
        auto [keys, results] = gb.aggregate(requests);
        auto partial = combine_keys_and_results(keys, results);

        // 2. Hash-partition partial result by group keys
        auto hashes = cudf::hash(partial.select(group_key_indices));
        auto [partitioned, offsets] = cudf::partition(partial, hashes % K, K);

        // 3. Spill each partition to disk
        for (int i = 0; i < K; i++) {
            auto slice = cudf::slice(partitioned, {offsets[i], offsets[i+1]});
            partition_files_[i].push_back(
                spill_mgr_.spill_to_disk(slice));
        }
    }

    /// Phase 2: Final merge per partition
    std::unique_ptr<cudf::table> finalize() {
        std::vector<std::unique_ptr<cudf::table>> final_results;
        for (int i = 0; i < K; i++) {
            // Load all partial results for this partition
            auto partials = reload_and_concatenate(partition_files_[i]);
            // Final aggregation
            cudf::groupby::groupby gb(partials.select(group_key_indices));
            auto [keys, results] = gb.aggregate(merge_requests);
            final_results.push_back(combine(keys, results));
        }
        return cudf::concatenate(final_results);
    }
};
```

**When to trigger**: Monitor accumulated partial results size. When total partials exceed `host_memory_budget * 0.5`, switch from in-memory accumulation to disk-spill mode. This is a one-way switch per operator instance — once spilling starts, all subsequent partials go to disk.

#### 5.6.4 Disk-Overflow Sort (External Sort)

When the input to sort exceeds GPU + host memory:

```
1. SORT-RUN phase:
   Read chunks → sort each chunk on GPU → write sorted run to disk

2. MERGE phase:
   K-way merge of sorted runs:
   - Open all run files
   - Read first block from each run into GPU
   - Repeatedly: find minimum across run heads, emit to output,
     refill from the run whose head was consumed
   - cudf::merge() handles merging pre-sorted tables
```

```cpp
class ExternalSort {
    std::vector<SpillFile> sorted_runs_;

    /// Phase 1: Produce sorted runs
    void add_chunk(cudf::table_view chunk) {
        auto sorted = cudf::sort(chunk, column_order, null_precedence);
        sorted_runs_.push_back(spill_mgr_.spill_to_disk(std::move(sorted)));
    }

    /// Phase 2: K-way merge
    std::unique_ptr<cudf::table> merge_runs() {
        // If all runs fit in GPU memory at once, merge directly
        if (total_run_size() <= gpu_budget_) {
            auto tables = reload_all_runs();
            return cudf::merge(tables, key_indices, column_order, null_precedence);
        }

        // Otherwise, multi-pass merge:
        // Merge M runs at a time (M chosen so M runs fit in GPU memory)
        // Each merge pass produces fewer, larger runs
        // Repeat until 1 run remains
        while (sorted_runs_.size() > 1) {
            std::vector<SpillFile> next_level;
            for (size_t i = 0; i < sorted_runs_.size(); i += merge_fan_in_) {
                auto batch = load_runs(i, std::min(i + merge_fan_in_, sorted_runs_.size()));
                auto merged = cudf::merge(batch, key_indices, column_order, null_precedence);
                next_level.push_back(spill_mgr_.spill_to_disk(std::move(merged)));
            }
            sorted_runs_ = std::move(next_level);
        }
        return spill_mgr_.reload_to_gpu(sorted_runs_[0]);
    }
};
```

#### 5.6.5 Spill Hierarchy Integration

The `SpillManager` (5.5.4) is extended to manage all three tiers. The escalation order when memory is tight:

```
1. GPU OOM → spill non-active GPU tables to host memory (existing 5.5.4)
2. Host memory pressure → spill host-resident tables to disk (new)
3. Disk overflow for pipeline-breaking operators (join build, aggregation accumulation, sort)
4. CPU fallback as last resort (existing 5.5.5)
```

```cpp
class SpillManager {
    // ... existing fields from 5.5.4 ...

    DiskSpillManager disk_mgr_;
    size_t host_memory_budget_;          // configurable via --host-memory-limit
    size_t host_memory_used_ = 0;

    /// Enhanced spill: GPU → host → disk cascade
    void spill(TableId id, cudf::table&& gpu_table) {
        if (host_memory_used_ + table_size < host_memory_budget_) {
            // Tier 1: spill to host
            spill_to_host(id, std::move(gpu_table));
        } else {
            // Tier 2: spill to disk
            disk_mgr_.spill_to_disk(id, std::move(gpu_table));
        }
    }

    /// When host memory is also under pressure, demote host-resident tables to disk
    void spill_host_to_disk(TableId id) {
        auto host_data = std::move(host_spilled_[id]);
        disk_mgr_.spill_to_disk(id, std::move(host_data));
        host_memory_used_ -= host_data.size;
    }
};
```

#### 5.6.6 CLI Flags

```
--spill-dir /path/to/ssd         # scratch directory for spill files (default: $TMPDIR)
--host-memory-limit 32G          # max host memory for spilled intermediates (default: 50% of system RAM)
--disk-spill-limit 500G          # max disk usage for spill files (default: unlimited)
```

#### 5.6.7 Spill File Lifecycle

```
Query starts
  └── Operator needs to spill
        └── SpillManager writes Arrow IPC to --spill-dir
              └── Files named: {query_id}_{operator_id}_{partition}_{sequence}.arrow
  └── Operator reads back spill files during merge/finalize
  └── Query completes (success or error)
        └── SpillManager::cleanup() deletes all files for this query

On startup:
  └── Scan --spill-dir for orphaned files from crashed queries
  └── Delete any files older than 1 hour
```

---

## Phase 6: Expression Evaluation Engine (C++)

### 6.1 Expression Tree
```cpp
struct GpuExpr {
    enum Kind { ColumnRef, Literal, BinaryOp, UnaryOp, Cast, Function };
    // ... variant fields
};
```

### 6.2 Expression Evaluator
```cpp
std::unique_ptr<cudf::column> evaluate(
    cudf::table_view const& input,
    GpuExpr const& expr
);
```
- Recursively evaluate sub-expressions
- `ColumnRef` → extract column from table view
- `Literal` → create `cudf::scalar`, broadcast to column
- `BinaryOp` → `cudf::binary_operation(lhs_col, rhs_col, op, output_type)`
- `UnaryOp` → `cudf::unary_operation(col, op)`
- `Cast` → `cudf::cast(col, target_type)`

---

## Phase 7: Built-in SQL Scalar Functions

DataFusion supports ~120 built-in scalar functions. The GPU executor must map each one to a cuDF C++ call. Functions that have no cuDF equivalent must fall back to CPU evaluation. The plan IR `Function` expression node carries the function name and arguments.

### 7.1 Function Dispatch Architecture

Extend the C++ expression evaluator (Phase 6) with a function dispatch table:

```cpp
using ScalarFn = std::function<std::unique_ptr<cudf::column>(
    std::vector<cudf::column_view> const& args,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)>;

std::unordered_map<std::string, ScalarFn> gpu_function_registry;
```

On the Rust side, the plan IR `Function` node:
```rust
struct PlanFunction {
    name: String,              // e.g. "substr", "date_trunc", "regexp_replace"
    args: Vec<PlanExpr>,       // evaluated recursively
    return_type: DataType,
}
```

During GPU expression evaluation, if `name` is in the GPU registry, dispatch to cuDF. Otherwise, mark the enclosing operator for CPU fallback (Phase 8).

### 7.2 Math Functions

All map cleanly to cuDF — either via `cudf::unary_operation()` or `cudf::binary_operation()`.

| DataFusion function | cuDF mapping |
|---|---|
| `abs(x)` | `cudf::unary_operation(x, ABS)` |
| `ceil(x)` | `cudf::unary_operation(x, CEIL)` |
| `floor(x)` | `cudf::unary_operation(x, FLOOR)` |
| `sqrt(x)` | `cudf::unary_operation(x, SQRT)` |
| `cbrt(x)` | `cudf::unary_operation(x, CBRT)` |
| `exp(x)` | `cudf::unary_operation(x, EXP)` |
| `ln(x)` | `cudf::unary_operation(x, LOG)` |
| `sin/cos/tan(x)` | `cudf::unary_operation(x, SIN/COS/TAN)` |
| `asin/acos/atan(x)` | `cudf::unary_operation(x, ARCSIN/ARCCOS/ARCTAN)` |
| `sinh/cosh/tanh(x)` | `cudf::unary_operation(x, SINH/COSH/TANH)` |
| `asinh/acosh/atanh(x)` | `cudf::unary_operation(x, ARCSINH/ARCCOSH/ARCTANH)` |
| `round(x, d)` | `cudf::round(x, d, HALF_UP)` |
| `trunc(x, d)` | `cudf::round(x, d)` with truncation (floor toward zero) |
| `signum(x)` | Binary ops: `(x > 0) - (x < 0)` cast to output type |
| `pow(x, y)` | `cudf::binary_operation(x, y, POW)` |
| `atan2(y, x)` | `cudf::binary_operation(y, x, ATAN2)` |
| `log(base, x)` | `cudf::binary_operation(x, base, LOG_BASE)` |
| `log2(x)` | `cudf::binary_operation(x, scalar(2), LOG_BASE)` |
| `log10(x)` | `cudf::binary_operation(x, scalar(10), LOG_BASE)` |
| `mod(x, y)` | `cudf::binary_operation(x, y, MOD)` |
| `pi()` | Constant scalar broadcast |
| `random()` | Custom CUDA kernel or `curand` fill |
| `isnan(x)` | `cudf::is_nan(x)` |
| `nanvl(x, y)` | `is_nan(x)` → if true use y, else x (conditional via bitmask) |

**Coverage: ~35/37 DataFusion math functions have direct cuDF mappings.**

Missing on GPU (fall back to CPU): `factorial`, `gcd`, `lcm`, `degrees`, `radians` (trivial: multiply by `180/π` or `π/180` — can implement via `binary_operation(x, scalar, MUL)`).

### 7.3 String Functions

cuDF's `cudf::strings` namespace covers most SQL string operations.

| DataFusion function | cuDF mapping |
|---|---|
| `length(s)` / `char_length(s)` / `character_length(s)` | `cudf::strings::count_characters(s)` |
| `octet_length(s)` / `bit_length(s)` | `cudf::strings::count_bytes(s)` (× 8 for bit_length) |
| `upper(s)` | `cudf::strings::to_upper(s)` |
| `lower(s)` | `cudf::strings::to_lower(s)` |
| `trim(s)` / `btrim(s)` | `cudf::strings::strip(s, BOTH)` |
| `ltrim(s)` | `cudf::strings::strip(s, LEFT)` |
| `rtrim(s)` | `cudf::strings::strip(s, RIGHT)` |
| `lpad(s, len, fill)` | `cudf::strings::pad(s, len, LEFT, fill)` |
| `rpad(s, len, fill)` | `cudf::strings::pad(s, len, RIGHT, fill)` |
| `substr(s, start, len)` / `substring(s, start, len)` | `cudf::strings::slice_strings(s, start, start+len)` |
| `left(s, n)` | `cudf::strings::slice_strings(s, 0, n)` |
| `right(s, n)` | `cudf::strings::slice_strings(s, -n)` |
| `replace(s, from, to)` | `cudf::strings::replace(s, from, to)` |
| `reverse(s)` | `cudf::strings::reverse(s)` |
| `repeat(s, n)` | `cudf::strings::repeat_strings(s, n)` |
| `concat(a, b, ...)` | `cudf::strings::concatenate(table_of_cols, "")` |
| `concat_ws(sep, a, b, ...)` | `cudf::strings::concatenate(table_of_cols, sep)` |
| `starts_with(s, prefix)` | `cudf::strings::starts_with(s, prefix)` |
| `ends_with(s, suffix)` | `cudf::strings::ends_with(s, suffix)` |
| `contains(s, substr)` | `cudf::strings::contains(s, substr)` |
| `strpos(s, substr)` / `position(substr IN s)` / `instr(s, substr)` | `cudf::strings::find(s, substr)` + 1 (SQL is 1-indexed) |
| `ascii(s)` | `cudf::strings::code_points(s)` (first element) |
| `chr(n)` | Custom: cast int to char |
| `initcap(s)` | `cudf::strings::capitalize(s)` |
| `translate(s, from, to)` | `cudf::strings::translate(s, from_chars, to_chars)` |
| `split_part(s, delim, n)` | `cudf::strings::split::split(s, delim)` → extract nth element |
| `overlay(s, replacement, start, len)` | `cudf::strings::replace_slice(s, replacement, start, start+len)` |
| `to_hex(n)` | `cudf::strings::convert::integers_to_hex(n)` |

**Coverage: ~30/38 DataFusion string functions have direct cuDF mappings.**

Missing on GPU (fall back to CPU): `levenshtein`, `find_in_set`, `substr_index`/`substring_index`, `uuid`, `encode`/`decode`.

### 7.4 Regular Expression Functions

cuDF has full GPU-accelerated regex support via `cudf::strings::regex_program`.

| DataFusion function | cuDF mapping |
|---|---|
| `regexp_like(s, pattern)` | `cudf::strings::contains_re(s, regex_program(pattern))` |
| `regexp_match(s, pattern)` | `cudf::strings::extract(s, regex_program(pattern))` |
| `regexp_replace(s, pattern, replacement)` | `cudf::strings::replace_re(s, regex_program(pattern), replacement)` |
| `regexp_count(s, pattern)` | `cudf::strings::count_re(s, regex_program(pattern))` |
| SQL `LIKE` pattern | `cudf::strings::like(s, pattern, escape)` (native LIKE support, no regex conversion needed) |

**Coverage: 5/5 — all regex functions have cuDF mappings.**

Note: cuDF `regex_program` objects should be cached and reused when the same pattern appears in multiple rows/batches, since regex compilation is expensive.

### 7.5 Date/Time Functions

cuDF's datetime support covers extraction and arithmetic but has gaps in formatting and construction.

| DataFusion function | cuDF mapping |
|---|---|
| `date_part('year', ts)` / `datepart` | `cudf::datetime::extract_datetime_component(ts, YEAR)` |
| `date_part('month', ts)` | `cudf::datetime::extract_datetime_component(ts, MONTH)` |
| `date_part('day', ts)` | `cudf::datetime::extract_datetime_component(ts, DAY)` |
| `date_part('hour', ts)` | `cudf::datetime::extract_datetime_component(ts, HOUR)` |
| `date_part('minute', ts)` | `cudf::datetime::extract_datetime_component(ts, MINUTE)` |
| `date_part('second', ts)` | `cudf::datetime::extract_datetime_component(ts, SECOND)` |
| `date_part('dow', ts)` | `cudf::datetime::extract_datetime_component(ts, WEEKDAY)` |
| `date_part('doy', ts)` | `cudf::datetime::day_of_year(ts)` |
| `date_part('quarter', ts)` | `cudf::datetime::extract_quarter(ts)` |
| `date_trunc('month', ts)` / `datetrunc` | `cudf::datetime::floor_datetimes(ts, MONTH)` |
| `date_trunc('day', ts)` | `cudf::datetime::floor_datetimes(ts, DAY)` |
| `date_trunc('hour', ts)` | `cudf::datetime::floor_datetimes(ts, HOUR)` |
| `date_bin(interval, ts, origin)` | `cudf::datetime::floor_datetimes()` with custom frequency, offset by origin |
| `to_timestamp(s)` | `cudf::strings::convert::to_timestamps(s, format)` |
| `to_date(s)` | `cudf::strings::convert::to_timestamps(s, date_format)` then cast to date |
| `to_char(ts, format)` / `date_format` | `cudf::strings::convert::from_timestamps(ts, format)` |
| `from_unixtime(n)` | Cast integer seconds to timestamp type via `cudf::cast()` |
| `now()` / `current_timestamp` | Constant scalar (evaluated once on Rust side before GPU dispatch) |
| `current_date` / `today` | Constant scalar |
| `current_time` | Constant scalar |

**Date/time arithmetic** (adding intervals to timestamps):
- Adding months: `cudf::datetime::add_calendrical_months(ts, months)`
- Adding days/hours/minutes/seconds: `cudf::binary_operation(ts, duration_scalar, ADD)` — cuDF supports timestamp ± duration via binary ops
- Date difference: `cudf::binary_operation(ts1, ts2, SUB)` → produces duration

**Coverage: ~20/25 DataFusion date/time functions have cuDF mappings.**

Missing on GPU (fall back to CPU): `make_date`, `make_time`, `to_local_time`, `to_unixtime` (trivial: cast timestamp to int64), `to_time`.

### 7.6 Conditional & Null Functions

These are expression-level constructs, not library calls. Implement in the expression evaluator via boolean mask composition.

| DataFusion function | GPU implementation |
|---|---|
| `coalesce(a, b, ...)` | Chain: `is_valid(a)` mask → select `a`, else recurse on `b, ...`. Use `cudf::copy_if_else(a, coalesce(b,...), mask)` |
| `nullif(a, b)` | `cudf::binary_operation(a, b, EQUAL)` → where true, set null. Use `cudf::copy_if_else(null_scalar, a, eq_mask)` |
| `ifnull(a, b)` / `nvl(a, b)` | `coalesce(a, b)` |
| `nvl2(a, b, c)` | `cudf::copy_if_else(b, c, is_valid(a))` |
| `greatest(a, b, ...)` | Chain `cudf::binary_operation(a, b, NULL_MAX)` pairwise |
| `least(a, b, ...)` | Chain `cudf::binary_operation(a, b, NULL_MIN)` pairwise |
| `CASE WHEN` | Evaluate each `WHEN` branch → boolean mask. Apply `cudf::copy_if_else()` in reverse order (last branch first, overlaying earlier branches) |

**Coverage: 7/7 — all implementable on GPU.**

### 7.7 Type Conversion

| DataFusion function | cuDF mapping |
|---|---|
| `CAST(x AS type)` | `cudf::cast(x, target_type)` — supports all numeric, string, timestamp, date, duration conversions |
| `TRY_CAST(x AS type)` | `cudf::cast()` with error → null (wrap in try/catch per-row not possible; use validity mask post-cast) |
| `to_timestamp_seconds/millis/micros/nanos(s)` | `cudf::strings::convert::to_timestamps(s, format)` then cast to target resolution |
| `arrow_typeof(x)` | Constant string (resolved at plan time, not a GPU operation) |

### 7.8 Coverage Summary & Fallback Strategy

| Category | Total in DataFusion | GPU-supported | CPU fallback |
|---|---|---|---|
| Math | 37 | 33 | 4 (`factorial`, `gcd`, `lcm`, `iszero`) |
| String | 38 | 30 | 8 (`levenshtein`, `find_in_set`, `uuid`, etc.) |
| Regex | 5 | 5 | 0 |
| Date/Time | 25 | 20 | 5 (`make_date`, `make_time`, etc.) |
| Conditional/Null | 7 | 7 | 0 |
| Type conversion | 4 | 3 | 1 (`TRY_CAST` partial) |
| **Total** | **~116** | **~98 (84%)** | **~18 (16%)** |

### 7.9 Fallback Mechanics for Unsupported Functions

When the plan IR contains a function not in the GPU registry:

1. **Expression-level fallback**: If only one expression in a `Project` or `Filter` uses an unsupported function, evaluate just that expression on CPU:
   - Pull the input columns needed by that expression to host via `cudf::to_arrow_host()`
   - Evaluate using DataFusion's scalar function implementation on CPU
   - Upload the result column back to GPU via `cudf::from_arrow_host()`
   - Continue the rest of the operator on GPU

2. **This is more granular than operator-level fallback** (Phase 5.5.5 / Phase 8). A query like:
   ```sql
   SELECT upper(name), levenshtein(name, 'target'), quantity * 2
   FROM orders
   WHERE quantity > 10
   ```
   runs filter on GPU, `upper()` on GPU, `quantity * 2` on GPU, and only `levenshtein()` drops to CPU for that single column — then the results are reassembled on GPU.

3. **Detection at plan time**: The `GpuExecutionRule` (Phase 2.2) checks all function references in the plan IR. If a subtree contains only GPU-supported functions, the whole operator runs on GPU. If any function is unsupported, the expression evaluator is configured for per-expression fallback.

### 7.10 Adding New Function Mappings

Adding GPU support for a new function requires:
1. Add entry to `gpu_function_registry` in C++
2. Implement the mapping (usually 1-5 lines calling a cuDF API)
3. Add a test that compares GPU vs CPU results for that function

No changes to the plan IR, FFI layer, or Rust side are needed — the function name string is already passed through.

---

## Phase 8: CPU Executor (Rust)

A pure-Rust executor that converts GPU plan IR back into DataFusion `ExecutionPlan` nodes and runs them on CPU. Serves two purposes: (1) correctness oracle for testing, (2) fallback when GPU execution is impossible.

### 8.1 Architecture

The CPU executor lives in `peacockdb-core` (pure Rust, no C++/CUDA dependency). It implements the same `PlanExecutor` trait as the GPU executor:

```rust
/// Backend-agnostic executor trait. Both GPU and CPU executors implement this.
trait PlanExecutor {
    fn execute(&self, plan: &PlanNode) -> Result<Vec<RecordBatch>>;
}

struct CpuExecutor { session_ctx: SessionContext }
struct GpuExecutor { /* FFI handle to C++ side */ }

impl PlanExecutor for CpuExecutor { /* ... */ }
impl PlanExecutor for GpuExecutor { /* ... */ }
```

### 8.2 Approach: Translate Plan IR → DataFusion ExecutionPlan

Rather than reimplementing operators from scratch, the CPU executor translates each plan IR node back into the corresponding DataFusion physical operator. DataFusion already has correct, well-tested implementations of every operator we need — reusing them avoids duplicating complex logic (hash join, aggregate accumulators, sort, etc.) and guarantees semantic equivalence.

```rust
/// Convert our plan IR back to a DataFusion ExecutionPlan tree.
fn ir_to_datafusion(node: &PlanNode) -> Result<Arc<dyn ExecutionPlan>> {
    match node {
        PlanNode::Scan { path, projection, .. } => {
            // → DataFusion's ParquetExec / DataSourceExec
            Ok(build_parquet_exec(path, projection)?)
        }
        PlanNode::Filter { input, predicate } => {
            let child = ir_to_datafusion(input)?;
            let phys_expr = ir_expr_to_physical(predicate, &child.schema())?;
            // → DataFusion's FilterExec
            Ok(Arc::new(FilterExec::try_new(phys_expr, child)?))
        }
        PlanNode::Project { input, exprs } => {
            let child = ir_to_datafusion(input)?;
            let phys_exprs = exprs.iter()
                .map(|e| ir_expr_to_physical(e, &child.schema()))
                .collect::<Result<Vec<_>>>()?;
            // → DataFusion's ProjectionExec
            Ok(Arc::new(ProjectionExec::try_new(phys_exprs, child)?))
        }
        PlanNode::HashJoin { left, right, on, filter, join_type, .. } => {
            let left_child = ir_to_datafusion(left)?;
            let right_child = ir_to_datafusion(right)?;
            // → DataFusion's HashJoinExec (also handles semi/anti)
            Ok(Arc::new(HashJoinExec::try_new(
                left_child, right_child,
                on.clone(), filter.clone(),
                join_type, /* partition_mode */ ...,
            )?))
        }
        PlanNode::SortMergeJoin { left, right, on, join_type, .. } => {
            let left_child = ir_to_datafusion(left)?;
            let right_child = ir_to_datafusion(right)?;
            // → DataFusion's SortMergeJoinExec
            Ok(Arc::new(SortMergeJoinExec::try_new(
                left_child, right_child,
                on.clone(), None, *join_type,
                vec![], false,
            )?))
        }
        PlanNode::NestedLoopJoin { left, right, filter, join_type, .. } => {
            let left_child = ir_to_datafusion(left)?;
            let right_child = ir_to_datafusion(right)?;
            // → DataFusion's NestedLoopJoinExec
            Ok(Arc::new(NestedLoopJoinExec::try_new(
                left_child, right_child,
                filter.clone(), join_type,
            )?))
        }
        PlanNode::CrossJoin { left, right } => {
            let left_child = ir_to_datafusion(left)?;
            let right_child = ir_to_datafusion(right)?;
            // → DataFusion's CrossJoinExec
            Ok(Arc::new(CrossJoinExec::new(left_child, right_child)))
        }
        PlanNode::Aggregate { input, group_by, aggr_exprs } => {
            let child = ir_to_datafusion(input)?;
            // → DataFusion's AggregateExec
            Ok(Arc::new(AggregateExec::try_new(
                AggregateMode::Single, group_by, aggr_exprs, child,
            )?))
        }
        PlanNode::Sort { input, sort_exprs } => {
            let child = ir_to_datafusion(input)?;
            // → DataFusion's SortExec
            Ok(Arc::new(SortExec::new(sort_exprs, child)))
        }
        PlanNode::Limit { input, count } => {
            let child = ir_to_datafusion(input)?;
            // → DataFusion's GlobalLimitExec
            Ok(Arc::new(GlobalLimitExec::new(child, 0, Some(*count))))
        }
        PlanNode::Union { inputs } => {
            let children = inputs.iter()
                .map(ir_to_datafusion)
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(UnionExec::new(children)))
        }
    }
}
```

Then execution is just DataFusion's standard collect:
```rust
impl PlanExecutor for CpuExecutor {
    fn execute(&self, plan: &PlanNode) -> Result<Vec<RecordBatch>> {
        let df_plan = ir_to_datafusion(plan)?;
        let task_ctx = self.session_ctx.task_ctx();
        let stream = df_plan.execute(0, task_ctx)?;
        block_on(stream.try_collect())
    }
}
```

### 8.3 Expression Translation (Plan IR → DataFusion PhysicalExpr)

The plan IR expressions are translated to DataFusion `PhysicalExpr` nodes, which are the same expression types DataFusion uses internally:

```rust
fn ir_expr_to_physical(
    expr: &PlanExpr,
    schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    match expr {
        PlanExpr::ColumnRef(idx) => {
            Ok(Arc::new(Column::new(schema.field(*idx).name(), *idx)))
        }
        PlanExpr::Literal(value) => {
            Ok(Arc::new(Literal::new(value.clone())))
        }
        PlanExpr::BinaryOp { left, right, op } => {
            let l = ir_expr_to_physical(left, schema)?;
            let r = ir_expr_to_physical(right, schema)?;
            Ok(Arc::new(BinaryExpr::new(l, *op, r)))
        }
        PlanExpr::Cast { input, to_type } => {
            let inner = ir_expr_to_physical(input, schema)?;
            Ok(Arc::new(CastExpr::new(inner, to_type.clone(), None)))
        }
        // ...
    }
}
```

### 8.4 Why This Approach

| Alternative | Problem |
|---|---|
| Reimplement join/aggregate/sort with raw `arrow::compute` | Duplicates thousands of lines of complex, bug-prone logic (hash tables, accumulator state machines, null handling). Must be kept in sync with any plan IR changes. |
| Use DataFusion's default execution (skip plan IR entirely) | Doesn't test the plan IR itself — a bug in IR generation would be invisible. |
| **Translate IR → DataFusion ExecutionPlan (chosen)** | Tests the full IR round-trip. Zero operator logic to maintain. If DataFusion can execute it, so can we. |

The round-trip (LogicalPlan → PhysicalPlan → GpuExecutionRule rewrites to IR → `ir_to_datafusion()` back to DataFusion nodes) also validates that the plan IR faithfully captures all information needed for execution — if a field is lost during IR serialization, the CPU executor will produce wrong results or fail, catching the bug.

### 8.5 Usage Modes

#### 8.5.1 As Test Oracle
```rust
#[test]
fn test_filter_correctness() {
    let plan = parse_and_plan("SELECT * FROM t WHERE x > 10");
    let gpu_result = GpuExecutor::new().execute(&plan).unwrap();
    let cpu_result = CpuExecutor.execute(&plan).unwrap();
    assert_batches_equal(&gpu_result, &cpu_result, /*float_epsilon=*/ 1e-6);
}
```

Every test runs the query through both executors and compares results. This catches GPU-specific bugs (precision, null handling edge cases, off-by-one in gather maps) that would be invisible if only testing against DataFusion's own execution.

#### 8.5.2 As OOM Fallback
When the GPU executor hits unrecoverable OOM (Phase 5.5.5), it delegates to the CPU executor:

```rust
impl GpuExecutor {
    fn execute_node(&self, node: &PlanNode) -> Result<Vec<RecordBatch>> {
        match self.try_execute_on_gpu(node) {
            Ok(result) => Ok(result),
            Err(e) if e.is_gpu_oom() => {
                log::warn!("GPU OOM on {}: falling back to CPU", node.name());
                CpuExecutor.execute(node)
            }
            Err(e) => Err(e),
        }
    }
}
```

The fallback is **per-operator subtree**, not all-or-nothing. A query like:
```
GpuSort
  └── GpuHashJoin        ← OOM here
        ├── GpuScan(A)
        └── GpuScan(B)
```
falls back to:
```
GpuSort                   ← still on GPU (result of join uploaded via from_arrow)
  └── CpuHashJoin         ← CPU fallback for this subtree
        ├── CpuScan(A)    ← CPU reads parquet directly
        └── CpuScan(B)
```

The CPU-produced `RecordBatch` is passed back through the normal Arrow FFI path (`cudf::from_arrow_host()`) so the parent GPU operator can consume it.

#### 8.5.3 As Standalone Mode
For environments without a GPU (CI, development laptops):
```
peacockdb --data-dir ./data --query "SELECT ..." --executor cpu
```
The `--executor` flag selects: `gpu` (default), `cpu`, or `auto` (try GPU, fall back to CPU).

### 8.6 What the CPU Executor is NOT
- It is not a performance-competitive engine — no vectorized execution, no SIMD, no parallelism beyond what `arrow-rs` provides internally
- It is not a reimplementation of DataFusion's full executor (no partitioning, no distribution, no async streaming)
- It processes the plan IR **batch-at-a-time** (all input materialized, then operator applied) — simple and correct, not fast
- It exists purely for correctness verification and graceful degradation

---

## Phase 9: Testing & Validation

### 9.1 Dual-Executor Correctness Testing
Run every test query through both CPU and GPU executors, compare results:

```rust
fn assert_equivalent(sql: &str, data_dir: &str) {
    let plan = plan_query(sql, data_dir);
    let gpu = GpuExecutor::new().execute(&plan).unwrap();
    let cpu = CpuExecutor.execute(&plan).unwrap();

    // Sort both results by all columns (execution order may differ)
    let gpu_sorted = sort_batches(&gpu);
    let cpu_sorted = sort_batches(&cpu);

    assert_batches_equal(&gpu_sorted, &cpu_sorted, 1e-6);
}
```

### 9.2 Test Queries (progressive complexity)
1. `SELECT * FROM lineitem LIMIT 10` — scan + limit
2. `SELECT l_orderkey, l_quantity FROM lineitem WHERE l_quantity > 30` — scan + filter + project
3. `SELECT l_returnflag, SUM(l_quantity) FROM lineitem GROUP BY l_returnflag` — aggregate
4. `SELECT * FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey` — hash join (equi)
5. `SELECT * FROM orders o LEFT JOIN lineitem l ON o.o_orderkey = l.l_orderkey` — hash join (left/right/full)
6. `SELECT * FROM orders WHERE o_custkey IN (SELECT c_custkey FROM customer WHERE ...)` — semi join
7. `SELECT * FROM orders WHERE o_custkey NOT IN (SELECT c_custkey FROM customer WHERE ...)` — anti join
8. `SELECT * FROM a, b WHERE a.x < b.y` — nested loop join (non-equi / theta join)
9. `SELECT * FROM a JOIN b ON a.key = b.key AND a.ts BETWEEN b.start AND b.end` — mixed join (equality + range)
10. `SELECT * FROM small_dim, small_dim2` — cross join
11. Full TPC-H queries (Q1, Q6, Q3, Q5, ...) — complex multi-join + aggregate

### 9.3 Scalar Function Tests
For each function category, test GPU vs CPU equivalence:
- **Math**: `SELECT abs(-1), ceil(1.5), floor(1.5), sqrt(4), ln(exp(1)), round(3.14159, 2), pow(2,10)`
- **String**: `SELECT upper('hello'), substr('abcdef', 2, 3), replace('foo', 'o', 'a'), concat('a','b','c'), length('test'), trim('  x  '), lpad('x', 5, '0')`
- **Regex**: `SELECT regexp_replace('abc123', '[0-9]+', 'N'), regexp_like('hello', 'h.*o'), regexp_count('aaa', 'a')`
- **Date/Time**: `SELECT date_part('year', timestamp '2024-03-15'), date_trunc('month', timestamp '2024-03-15 10:30:00'), now() - interval '1 day'`
- **Conditional**: `SELECT coalesce(NULL, NULL, 3), nullif(1, 1), CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END, greatest(1, 2, 3)`
- **Mixed**: queries combining multiple function categories in a single expression tree
- **NULL propagation**: every function tested with NULL inputs to verify correct null handling
- **Expression-level fallback**: queries mixing GPU-supported and CPU-only functions (e.g., `upper(name)` + `levenshtein(name, 'x')`) to verify per-expression fallback produces correct results

### 9.4 CPU-Only CI
- CI runs all tests with `--executor cpu` (no GPU required in CI)
- Tests verify the plan IR → CPU execution path is correct
- A separate GPU CI job (if available) runs the dual-executor comparison

### 9.5 OOM Fallback Tests
- Set `--gpu-memory-limit` artificially low (e.g., 64MB)
- Run queries that require more than 64MB intermediate memory
- Verify: query completes successfully via CPU fallback, results match full-GPU execution
- Verify: warning log messages indicate which operators fell back

### 9.6 Memory Pressure Tests
- Run queries on datasets larger than GPU memory
- Verify chunking produces correct results
- Monitor GPU memory usage stays within budget

---

## Phase 10: Optimizations (Future)

- **Predicate pushdown to parquet reader**: Use `parquet_reader_options::set_filter()` with cuDF's AST expressions to skip row groups at scan time
- **Projection pushdown**: Only read needed columns from parquet
- **Multi-GPU support**: Partition data across GPUs with `UCX` / `NCCL`
- **Persistent table catalog**: Store metadata so repeated queries skip schema inference
- **Prepared statements / query caching**: Cache physical plans for repeated queries
- **Triple-buffering**: If I/O is consistently faster than compute (or vice versa), a third buffer can absorb the imbalance

---

## Dependency Summary

| Component | Crate / Library |
|---|---|
| SQL parsing & planning | `datafusion` |
| Arrow memory format | `arrow-rs` |
| Arrow FFI | `arrow::ffi` |
| CLI | `clap` |
| Serialization (plan IR) | `flatbuffers` or `prost` (protobuf) |
| C++ build | `cmake` crate in `build.rs` |
| GPU compute | `libcudf` (C++) |
| GPU memory | `librmm` (C++) |
| CUDA runtime | CUDA Toolkit 12.x |

---

## Milestone Summary

| Milestone | Deliverable |
|---|---|
| **M1** | Project builds, links libcudf, CLI parses args |
| **M1b** | CI: GitHub Actions for build+CPU tests, Verda GPU tests with auto-destroy + reaper |
| **M2** | `SELECT * FROM table LIMIT N` works end-to-end (scan → GPU → Arrow → print) |
| **M3** | Pipelined I/O: scan chunks overlap with GPU execution via double-buffering |
| **M4** | Filter + Projection on GPU (pipelineable operators) |
| **M5** | Hash Join on GPU (build/probe pipeline split, inner/left/right/full/semi/anti) |
| **M5b** | Sort-merge join, nested loop join (conditional/mixed), cross join on GPU |
| **M6** | Aggregation (GROUP BY) on GPU (partial agg per chunk + final merge) |
| **M7** | Sort + Limit on GPU |
| **M8** | Math + string + date/time scalar functions on GPU (core ~80 functions) |
| **M9** | Regex functions on GPU, expression-level CPU fallback for unsupported functions |
| **M10** | CPU executor passes same test suite as GPU (no GPU required) |
| **M11** | Dual-executor tests: GPU vs CPU results match on all queries |
| **M12** | Chunked execution for large datasets |
| **M13** | OOM fallback: queries complete via mixed GPU/CPU when memory is tight |
| **M14** | Disk-overflow hash join (Grace hash join) for build sides exceeding GPU+host memory |
| **M15** | Disk-overflow aggregation and external sort |
| **M16** | TPC-H Q1 and Q6 pass correctness tests (both executors) |
| **M17** | Distribution: Docker image published, tarball release with graceful driver mismatch detection |

---

## Phase 11: Distribution

### 11.1 The CUDA Driver Compatibility Problem

`libcuda.so` (the GPU driver) is supplied by the host OS and **cannot be bundled** with the application. Every other CUDA library (`libcudart.so`, `libcudf.so`, `librmm.so`, etc.) can be distributed, but the driver must match or exceed the CUDA toolkit version the libraries were compiled against.

CUDA uses a forward-compatibility model: a driver is guaranteed to support all toolkit versions up to and including the version it shipped with. A driver that is **too old** will refuse to run code compiled against a newer toolkit, producing a cryptic `CUDA_ERROR_INVALID_PTX` or `cudaErrorInsufficientDriver` at runtime.

### 11.2 Docker Image (primary distribution for the server)

Docker + `nvidia-container-toolkit` is the standard deployment model for GPU workloads. The host only needs a compatible NVIDIA driver; the CUDA runtime and all libraries live inside the image.

```dockerfile
FROM nvcr.io/nvidia/cuda:12.6-base-ubuntu22.04
COPY lib/  /opt/peacockdb/lib/
COPY bin/  /opt/peacockdb/bin/
ENV LD_LIBRARY_PATH=/opt/peacockdb/lib
ENTRYPOINT ["/opt/peacockdb/bin/peacockdb"]
```

- Pin the base image to the exact CUDA toolkit version used to build cuDF.
- Publish to a container registry (GitHub Container Registry or Docker Hub) as part of CI on merge to `master`.
- Clients connect over a network socket and need no CUDA at all.

### 11.3 Tarball Release (power users, bare-metal)

A relocatable tarball bundles all `.so` dependencies except `libcuda.so`:

```
peacockdb-<version>-linux-x86_64.tar.gz
  bin/peacockdb
  lib/libpeacock_gpu.so
      libcudf.so
      librmm.so
      libcudart.so   ← redistributable, from CUDA toolkit
      ...            ← all transitive deps collected via ldd
```

**RPATH** in `libpeacock_gpu.so` and the `peacockdb` binary must be set to `$ORIGIN/../lib` so the dynamic linker finds the bundled libraries without requiring `LD_LIBRARY_PATH`.

**Graceful driver mismatch detection**: at startup, before any CUDA call, check driver compatibility explicitly and print a clear error if the driver is too old:

```rust
// In peacockdb/src/main.rs, before initializing the GPU executor:
let (driver_ver, runtime_ver) = peacock_cuda_versions(); // thin FFI to cudaDriverGetVersion / cudaRuntimeGetVersion
if driver_ver < runtime_ver {
    eprintln!(
        "error: CUDA driver version {} is too old for this build (requires >= {}).\n\
         Please update your NVIDIA driver.",
        format_cuda_ver(driver_ver),
        format_cuda_ver(runtime_ver)
    );
    std::process::exit(1);
}
```

This turns a cryptic PTX error deep in cuDF into an actionable message at the very first line of output.

**Packaging script** (`scripts/make-tarball.sh`): collect all shared library dependencies with `ldd`, copy them into `lib/`, patch RPATHs with `patchelf`, and produce the tarball. Run as a CI step after the cmake install step.

### 11.4 CLI Client

The query client (`peacockdb-client`) has no CUDA dependency — it is a pure Rust binary that speaks the wire protocol. Distribute it as a single statically linked binary (build with `--target x86_64-unknown-linux-musl`) available via:

```
curl -L https://github.com/.../releases/latest/download/peacockdb-client-linux-x86_64 -o peacockdb-client
chmod +x peacockdb-client
```
