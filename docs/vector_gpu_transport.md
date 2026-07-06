# Vector-search GPU transport (design note)

Status: **design only, implementation deferred to a GPU-enabled branch.** This
note is Ticket 3 of the vector-search MVP (`ENS1-vector-search-mvp`). It records
the architecture for running the ExactBrute vector top-k on the GPU via cuVS, why
the obvious mappings don't work, and the concrete change list to implement it
where it can actually be compiled and tested (remote GPU CI).

Nothing here changes the shipped behaviour: the CPU fallback in
`peacockdb-core/src/vector/exec.rs` remains the `rust-only` path, and the plan IR
(`GpuVectorSearch`) already round-trips. Only the `default`-feature GPU executor
is affected, and none of it lands until it can be built against cuDF + cuVS + a
device.

---

## 1. The finding: fp16 embeddings cannot ride in the cuDF table

The physical executor is a cuDF-table dataflow: every node's runtime value is a
`TableResult { std::unique_ptr<cudf::table> table; std::vector<std::string>
column_names; }` (`cpp/src/plan_executor.h`), and `run_op` dispatches each
`fb::PlanNodeKind` to an `execute_*` that consumes and produces such tables
(`cpp/src/plan_executor.cpp`). So for `GpuVectorSearch` to score its input, the
embedding would have to be a **column of the input cuDF table**.

It cannot be, in this tree:

- `fb_to_type_id()` (`plan_executor.cpp`) maps `fb::DataType_Float16 -> EMPTY`
  (no case; the `default` returns `cudf::type_id::EMPTY`). fp16 is not a supported
  element type on the C++ side.
- There is **no** `FixedSizeList` / list handling on the C++ side at all: a grep
  of `cpp/src` + `cpp/include` for `FLOAT16`, `__half`, `type_id::LIST`,
  `from_arrow`, `FixedSizeList` returns nothing. cuDF here has no
  `FIXED_SIZE_LIST` type, and importing an Arrow `FixedSizeList<Float16>` would at
  best yield a `LIST<FLOAT16>`, whose element type cuDF still does not support.

Consequences:

- The embedding produced by ticket 1/2 (Arrow `FixedSizeList<Float16, dim>`, see
  `peacockdb-core/src/vector/types.rs`) has no cuDF column representation, so it
  cannot arrive as an input column to a `GpuVectorSearch` op.
- Even the C++ GPU unit test cannot construct a cuDF input table holding fp16
  vectors to check the op against a brute-force reference.
- Independently, cuVS does not want a cuDF column anyway: `brute_force::build`
  takes a `raft::device_matrix_view<const __half, int64_t, row_major>` — a raw,
  contiguous `(n × dim)` device buffer.

## 2. Chosen architecture — "Correct-C": an fp16 side-buffer on `TableResult`

The embedding is **row data** produced by the scan subtree, not plan structure.
It must therefore live in the executor's runtime table value, **not** in the
serialized FlatBuffers plan. (Serializing embeddings into the plan would push
millions of rows of vectors into the plan bytes and conflate plan with data. The
IR only *references* the embedding: `GpuVectorSearch` already carries
`metric / query / dim / scalar / k / strategy`, which is all the plan needs.)

Design:

- `TableResult` gains an **optional raw fp16 device-matrix side-buffer**, carried
  *alongside* the cuDF table (not as a column):

  ```cpp
  struct VectorMatrix {
    rmm::device_buffer data;   // dim * num_rows __half values, row-major, contiguous
    int64_t num_rows;
    int64_t dim;
  };
  struct TableResult {
    std::unique_ptr<cudf::table> table;
    std::vector<std::string>     column_names;
    std::optional<VectorMatrix>  embedding;   // present only when the subtree carries a vector column
  };
  ```

  The matrix is aligned **row-for-row** with `table`: matrix row `i` is the
  embedding of table row `i`. Every row-preserving/row-reordering op that already
  rewrites the table (filter, slice, gather, sort) must apply the *same*
  transform to `embedding` so alignment is preserved. For the MVP the only op
  between scan and `GpuVectorSearch` is `GpuFilter` (the pre-filter pushed down by
  `PushFilterIntoVectorTopK`), so the alignment burden is small: filter must also
  gather the surviving rows out of the embedding matrix.

- **Producer:** a vector-aware `GpuScan` reads the `FixedSizeList<Float16, dim>`
  Parquet/Arrow column into a contiguous `(n × dim)` fp16 device buffer and sets
  `TableResult::embedding`. The vector column is *not* added to the cuDF table
  (cuDF can't hold it); the passthrough columns (id, filter keys, projected
  output) are the cuDF table.

- **Consumer:** `execute_gpu_vector_search` reads `metric/dim/scalar/k` from the
  `GpuVectorSearch` fbs table and the query vector from `GpuVectorSearch.query`
  (`dim` little-endian `__half` values → `device_span<const __half>` of length
  `dim`), takes `input.embedding` as the dataset, runs cuVS
  `brute_force::{build, search}` under L2 to get the top-`k` **row indices**, and
  **gathers those indices against the passthrough cuDF `input.table`** to produce
  the output `TableResult`. Output rows are ordered nearest-first (ascending L2),
  matching the CPU fallback and the `ORDER BY l2_distance … ASC LIMIT k` SQL
  semantics.

This mirrors the CPU fallback exactly (`exec.rs`: score the already-filtered
input, keep the k smallest, gather nearest-first), so CPU and GPU return the same
rows; the only difference is where the scoring happens.

### Why A and B were rejected

- **A — `dim` separate FLOAT32 columns (or one fp32 buffer) in the cuDF table.**
  Doesn't scale to real embedding dims (128–1536+ columns), and forces the
  scan/projection to shred and reassemble the vector, diverging the GPU column
  model from the CPU `FixedSizeList` model. Rejected.
- **B — teach the cuDF mapping fp16 + fixed-size-list.** cuDF has no fp16 element
  type; this is not fixable at the `fb_to_type_id` / `from_arrow` layer. Dead.
- **C (this note)** keeps cuDF out of the fp16 business entirely, matches the raw
  `device_matrix_view<const __half>` cuVS actually wants, and keeps the CPU and
  GPU embedding models identical (both `FixedSizeList<Float16>` at the source).

## 3. Concrete change list (to implement on a GPU-enabled branch)

1. **`TableResult` / node-executor data model** (`cpp/src/plan_executor.h`,
   `plan_executor.cpp`): add the optional `VectorMatrix embedding` field above.
   Audit every `execute_*` that reorders/subsets rows (`execute_filter`,
   `execute_limit`/slice, any gather/sort) to apply the same row transform to
   `embedding` when present, so it stays row-aligned with the table. Ops that
   don't touch a vector-bearing subtree simply leave `embedding` empty.

2. **Vector-aware `GpuScan`** (`execute_scan`): when the scanned schema has a
   `FixedSizeList<Float16, dim>` column, read it into a contiguous `(n × dim)`
   row-major fp16 `rmm::device_buffer` and populate `TableResult::embedding`;
   keep the remaining columns as the cuDF table. **Open item (§4):** confirm how
   the pinned cuDF/Arrow import path exposes the fixed-size-list child's raw fp16
   bytes (Arrow C-Data import of the child buffer vs. a cuDF list column we then
   flatten). The child is contiguous for a fixed-size list, so a `dim`-strided
   copy (or direct buffer take) yields the matrix.

3. **`execute_gpu_vector_search`** (new `cpp/src/ops/vector_search.{h,cpp}`, or
   inline to match the file's convention — the other ops are inline
   `static TableResult execute_*` in `plan_executor.cpp`):
   - Read `GpuVectorSearch` via `node->node_as_GpuVectorSearch()`; pull
     `dim/k/metric/scalar/strategy` and the `query` bytes.
   - `input = execute_node(vs->input())`; require `input.embedding` present with
     matching `dim`.
   - Build `raft::device_matrix_view<const __half>` over `input.embedding.data`
     `(num_rows × dim)`; query = `device_span<const __half>` of length `dim` from
     `vs->query()` (copy the `dim` LE `__half` values H2D).
   - `cuvs::neighbors::brute_force::index idx = build(res, dataset, L2)`;
     `search(res, idx, query(1 × dim), neighbors(1 × k), distances(1 × k))`.
   - Gather the `k` neighbor row indices against `input.table` (`cudf::gather`) →
     output table; drop `embedding` on the result (top-k output doesn't re-expose
     the vector unless a parent needs it — MVP parents don't). Order nearest-first.
   - Add `case fb::PlanNodeKind_GpuVectorSearch` to `run_op`'s switch and to
     `node_children` (`return {node->node_as_GpuVectorSearch()->input()};`) and to
     `plan_node_kind_name`.

4. **cuVS in the build** (`cpp/CMakeLists.txt`): add cuVS **at the same rapids
   minor as the pinned cuDF** so cuVS/cuDF share RAFT + rmm (mismatched RAFT/rmm
   is the classic ODR/ABI break). Order of preference:
   `find_package(cuvs CONFIG)` if the CI conda image ships it (see §4), else
   `rapids_cpm`/`CPMAddPackage(cuvs)` pinned to the cuDF rapids branch. Link
   `peacock_gpu` with `cuvs::cuvs` (add to the existing
   `target_link_libraries(peacock_gpu PRIVATE cudf::cudf flatbuffers
   Arrow::arrow_shared)`), and bundle `libcuvs.so` in the install/rpath logic the
   way Arrow/cuDF already are. **Do not add cuVS to CMake until implementation
   starts** (per coord: design-first).

5. **C++ GPU test** (`cpp/tests/gpu/test_plan_executor.cpp` conventions, label
   `gpu`): build a small deterministic fixture (a handful of fp16 vectors,
   `dim ≈ 4–8`) as a `GpuVectorSearch` plan over a vector-aware scan; assert the
   returned top-k ids/rows equal the exact brute-force reference computed in the
   test. Runs only in the GPU tier.

### Rust side — already done, no further change

- The serializer already emits/reads `GpuVectorSearch`
  (`plan_serializer.rs`: `serialize_gpu_vector_search` / `deserialize_gpu_vector_search`),
  and `GpuVectorSearchExec` is reached the same way other `Gpu*Exec` nodes reach
  C++ under `default` features. The plan IR needs **no** change for Correct-C
  (embedding stays out of the plan bytes).
- The `rust-only` CPU fallback (`exec.rs`) is unchanged and remains the tested
  behaviour off-GPU.

## 4. Open items (must be resolved in the GPU environment)

- **cuVS availability:** does the CI GPU conda image already ship cuVS
  (`find_package(cuvs CONFIG)` works), or must we `CPMAddPackage` it at the pinned
  cuDF's rapids branch? Unverifiable locally (no image, no cuDF submodule checked
  out). Pick `find_package` if present; otherwise CPM at the exact rapids minor.
- **fp16 child extraction:** does the pinned cuDF/Arrow import path expose the
  `FixedSizeList<Float16>` child's raw contiguous fp16 buffer cleanly (Arrow
  C-Data import of the child array vs. a cuDF `LIST` column we flatten)? This is
  the one spot where cuDF's lack of fp16 could still bite at import time; needs a
  device to confirm the buffer can be obtained without a cuDF fp16 column.
- **RAFT/rmm sharing:** confirm the resolved cuVS pulls the **same** RAFT + rmm as
  the pinned cuDF (25.02 on the 25.02 leg, 26.02 on the 26.02 leg). A version skew
  here is the most likely CI build failure.

## What was NOT verified locally

Everything in §§2–4 is unbuildable in this environment: no cuDF submodule, no
cuVS, no GPU. The `cpp/`-side findings in §1 (the `fb_to_type_id` gap, the absence
of fp16/list/`from_arrow` support, the `TableResult` shape, the `run_op` dispatch)
were read directly from source and are solid. The cuVS API names
(`brute_force::{build, search}`, `raft::device_matrix_view<const __half>`) are
from the cuVS docs and must be pinned to the actual header signatures at the
resolved cuVS version when implementation starts.
