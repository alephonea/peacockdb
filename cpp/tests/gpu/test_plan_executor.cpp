/// GPU tests for plan_executor: builds FlatBuffer plans programmatically and
/// executes them against testdata/tpch.minimal/ Parquet files.

#include "plan_executor.h"
#include "generated/gpu_plan_generated.h"

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "rmm_pool.hpp"

namespace fb = peacock::plan;

static std::string testdata_dir() {
  const char* env = std::getenv("PEACOCK_TESTDATA_DIR");
  if (env) return std::string(env);
  return std::string(PEACOCK_TESTDATA_DIR);
}

static std::string parquet_path(const std::string& table) {
  return testdata_dir() + "/tpch.minimal/" + table + ".parquet";
}

/// Build an Expr wrapping a ColumnRef.
static flatbuffers::Offset<fb::Expr> make_col_ref(
    flatbuffers::FlatBufferBuilder& fbb, uint32_t index,
    const char* name = nullptr) {
  auto name_off = name ? fbb.CreateString(name)
                       : flatbuffers::Offset<flatbuffers::String>{};
  auto col = fb::CreateColumnRef(fbb, index, name_off);
  return fb::CreateExpr(fbb, fb::ExprNode_ColumnRef, col.Union());
}

/// Build an Expr wrapping an Int64 literal.
static flatbuffers::Offset<fb::Expr> make_int64_literal(
    flatbuffers::FlatBufferBuilder& fbb, int64_t val) {
  auto sv = fb::CreateScalarValue(fbb, fb::DataType_Int64,
                                  /*bool_val=*/false, /*int_val=*/val);
  auto lit = fb::CreateLiteralExpr(fbb, sv);
  return fb::CreateExpr(fbb, fb::ExprNode_LiteralExpr, lit.Union());
}

/// Build an Expr wrapping a Float64 literal.
static flatbuffers::Offset<fb::Expr> make_float64_literal(
    flatbuffers::FlatBufferBuilder& fbb, double val) {
  auto sv = fb::CreateScalarValue(fbb, fb::DataType_Float64,
                                  /*bool_val=*/false, /*int_val=*/0,
                                  /*uint_val=*/0, /*float_val=*/val);
  auto lit = fb::CreateLiteralExpr(fbb, sv);
  return fb::CreateExpr(fbb, fb::ExprNode_LiteralExpr, lit.Union());
}

/// Build a binary expression: left <op> right.
static flatbuffers::Offset<fb::Expr> make_binary_expr(
    flatbuffers::FlatBufferBuilder& fbb,
    flatbuffers::Offset<fb::Expr> left, fb::BinaryOp op,
    flatbuffers::Offset<fb::Expr> right) {
  auto bin = fb::CreateBinaryExprNode(fbb, left, op, right);
  return fb::CreateExpr(fbb, fb::ExprNode_BinaryExprNode, bin.Union());
}

/// Build a CAST expression.
static flatbuffers::Offset<fb::Expr> make_cast_expr(
    flatbuffers::FlatBufferBuilder& fbb,
    flatbuffers::Offset<fb::Expr> inner, fb::DataType target_type) {
  auto cast = fb::CreateCastExprNode(fbb, inner, target_type);
  return fb::CreateExpr(fbb, fb::ExprNode_CastExprNode, cast.Union());
}

/// Wrap a plan node kind into a PlanNode table.
static flatbuffers::Offset<fb::PlanNode> make_plan_node(
    flatbuffers::FlatBufferBuilder& fbb, fb::PlanNodeKind kind,
    flatbuffers::Offset<void> node,
    flatbuffers::Offset<fb::Schema> schema = {}) {
  return fb::CreatePlanNode(fbb, kind, node, schema);
}

/// Build a Schema from field definitions.
static flatbuffers::Offset<fb::Schema> make_schema(
    flatbuffers::FlatBufferBuilder& fbb,
    const std::vector<std::pair<std::string, fb::DataType>>& fields) {
  std::vector<flatbuffers::Offset<fb::Field>> field_offsets;
  for (auto& [name, dt] : fields) {
    field_offsets.push_back(
        fb::CreateField(fbb, fbb.CreateString(name), dt, /*nullable=*/true));
  }
  return fb::CreateSchema(fbb, fbb.CreateVector(field_offsets));
}

/// Finish builder as a GpuPlan and return the buffer bytes.
static std::vector<uint8_t> finish_plan(
    flatbuffers::FlatBufferBuilder& fbb,
    flatbuffers::Offset<fb::PlanNode> root) {
  auto plan = fb::CreateGpuPlan(fbb, root);
  fbb.Finish(plan);
  auto* ptr = fbb.GetBufferPointer();
  return {ptr, ptr + fbb.GetSize()};
}

template <typename T>
static T get_scalar_value(const cudf::column_view& col, cudf::size_type row);

template <>
int32_t get_scalar_value<int32_t>(const cudf::column_view& col,
                                   cudf::size_type row) {
  std::vector<int32_t> host(col.size());
  cudaMemcpy(host.data(), col.data<int32_t>(),
             col.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
  return host[row];
}

template <>
int64_t get_scalar_value<int64_t>(const cudf::column_view& col,
                                   cudf::size_type row) {
  std::vector<int64_t> host(col.size());
  cudaMemcpy(host.data(), col.data<int64_t>(),
             col.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
  return host[row];
}

static std::string get_string_value(const cudf::column_view& col,
                                    cudf::size_type row) {
  cudf::strings_column_view scv{col};
  auto offsets = scv.offsets();
  auto chars_size = scv.chars_size(cudf::get_default_stream());

  // Large string columns carry int64 offsets, small ones int32.
  if (offsets.type().id() == cudf::type_id::INT64) {
    std::vector<int64_t> host_offsets(offsets.size());
    cudaMemcpy(host_offsets.data(), offsets.data<int64_t>(),
               offsets.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
    std::vector<char> host_chars(chars_size);
    cudaMemcpy(host_chars.data(), scv.chars_begin(cudf::get_default_stream()),
               chars_size, cudaMemcpyDeviceToHost);
    auto start = host_offsets[row];
    auto end = host_offsets[row + 1];
    return {host_chars.data() + start, host_chars.data() + end};
  } else {
    std::vector<int32_t> host_offsets(offsets.size());
    cudaMemcpy(host_offsets.data(), offsets.data<int32_t>(),
               offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    std::vector<char> host_chars(chars_size);
    cudaMemcpy(host_chars.data(), scv.chars_begin(cudf::get_default_stream()),
               chars_size, cudaMemcpyDeviceToHost);
    auto start = host_offsets[row];
    auto end = host_offsets[row + 1];
    return {host_chars.data() + start, host_chars.data() + end};
  }
}

TEST(PlanExecutor, ScanNation) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("nation"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});

  // nation schema: n_nationkey(Int32), n_name(Utf8View), n_regionkey(Int32),
  //                n_comment(Utf8View)
  auto schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });

  auto scan = fb::CreateGpuScan(fbb, paths, schema);
  auto node = make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());
  auto buf = finish_plan(fbb, node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 4);
  EXPECT_EQ(result.table->num_rows(), 25);
  EXPECT_EQ(result.column_names[0], "n_nationkey");
  EXPECT_EQ(result.column_names[1], "n_name");
  EXPECT_EQ(result.column_names[2], "n_regionkey");
  EXPECT_EQ(result.column_names[3], "n_comment");
}

TEST(PlanExecutor, ScanNationProjected) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("nation"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});

  auto schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });

  // Project only columns 1 (n_name) and 2 (n_regionkey).
  std::vector<uint32_t> proj{1, 2};
  auto proj_vec = fbb.CreateVector(proj);

  auto scan = fb::CreateGpuScan(fbb, paths, schema, proj_vec);
  auto node = make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());
  auto buf = finish_plan(fbb, node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 2);
  EXPECT_EQ(result.table->num_rows(), 25);
  EXPECT_EQ(result.column_names[0], "n_name");
  EXPECT_EQ(result.column_names[1], "n_regionkey");
}

TEST(PlanExecutor, FilterNation) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("nation"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  auto scan = fb::CreateGpuScan(fbb, paths, schema);
  auto scan_node = make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());

  // Filter: n_regionkey (col 2) > 2. Cast the Int32 column to Int64 so it matches
  // the Int64 literal rather than relying on cuDF AST implicit promotion.
  auto col2 = make_col_ref(fbb, 2, "n_regionkey");
  auto cast_col2 = make_cast_expr(fbb, col2, fb::DataType_Int64);
  auto lit2 = make_int64_literal(fbb, 2);
  auto predicate = make_binary_expr(fbb, cast_col2, fb::BinaryOp_Gt, lit2);

  auto filter = fb::CreateGpuFilter(fbb, predicate, scan_node);
  auto filter_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuFilter, filter.Union());
  auto buf = finish_plan(fbb, filter_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 4);
  // Regions 3 and 4 (0-indexed) have nations. Exact count depends on data.
  EXPECT_GT(result.table->num_rows(), 0);
  EXPECT_LT(result.table->num_rows(), 25);
}

TEST(PlanExecutor, HashJoinNationRegion) {
  flatbuffers::FlatBufferBuilder fbb;

  // Left: nation (n_nationkey, n_name, n_regionkey, n_comment)
  auto nation_path = fbb.CreateString(parquet_path("nation"));
  auto nation_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{nation_path});
  auto nation_schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  auto nation_scan = fb::CreateGpuScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, nation_scan.Union());

  // Right: region (r_regionkey, r_name, r_comment)
  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateGpuScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, region_scan.Union());

  // Join keys: n_regionkey (col 2 in left) = r_regionkey (col 0 in right)
  auto lk = make_col_ref(fbb, 2);
  auto rk = make_col_ref(fbb, 0);
  auto join_key = fb::CreateJoinKey(fbb, lk, rk);
  auto keys_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::JoinKey>>{join_key});

  auto join = fb::CreateGpuHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuHashJoin, join.Union());
  auto buf = finish_plan(fbb, join_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  // Every nation has exactly one region → 25 rows, 7 columns (4 + 3).
  ASSERT_EQ(result.table->num_columns(), 7);
  EXPECT_EQ(result.table->num_rows(), 25);
}

TEST(PlanExecutor, HashJoinWithProjection) {
  flatbuffers::FlatBufferBuilder fbb;

  // nation
  auto nation_path = fbb.CreateString(parquet_path("nation"));
  auto nation_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{nation_path});
  auto nation_schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  auto nation_scan = fb::CreateGpuScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, nation_scan.Union());

  // region
  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateGpuScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, region_scan.Union());

  // Join keys: n_regionkey (col 2) = r_regionkey (col 0)
  auto lk = make_col_ref(fbb, 2);
  auto rk = make_col_ref(fbb, 0);
  auto join_key = fb::CreateJoinKey(fbb, lk, rk);
  auto keys_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::JoinKey>>{join_key});

  // Output projection: keep only n_name(1), n_regionkey(2), r_name(5)
  // Full join output: [n_nationkey(0), n_name(1), n_regionkey(2),
  //   n_comment(3), r_regionkey(4), r_name(5), r_comment(6)]
  std::vector<uint32_t> proj{1, 2, 5};
  auto proj_vec = fbb.CreateVector(proj);

  auto join = fb::CreateGpuHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node, proj_vec);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuHashJoin, join.Union());
  auto buf = finish_plan(fbb, join_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 3);
  EXPECT_EQ(result.table->num_rows(), 25);
  EXPECT_EQ(result.column_names[0], "n_name");
  EXPECT_EQ(result.column_names[1], "n_regionkey");
  EXPECT_EQ(result.column_names[2], "r_name");
}

TEST(PlanExecutor, SortNationByName) {
  flatbuffers::FlatBufferBuilder fbb;

  // Scan nation, project n_name only.
  auto path = fbb.CreateString(parquet_path("nation"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  std::vector<uint32_t> proj_cols{1};
  auto proj_vec = fbb.CreateVector(proj_cols);
  auto scan = fb::CreateGpuScan(fbb, paths, schema, proj_vec);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());

  // Sort by col 0 (n_name) ascending.
  auto sort_expr = make_col_ref(fbb, 0);
  auto sort_spec = fb::CreateSortExprNode(fbb, sort_expr, /*asc=*/true,
                                           /*nulls_first=*/false);
  auto sort_specs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::SortExprNode>>{sort_spec});

  auto sort = fb::CreateGpuSort(fbb, sort_specs, /*fetch=*/-1, scan_node);
  auto sort_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuSort, sort.Union());
  auto buf = finish_plan(fbb, sort_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 1);
  EXPECT_EQ(result.table->num_rows(), 25);
  EXPECT_EQ(result.column_names[0], "n_name");

  // Sort order: ALGERIA first, VIETNAM last.
  auto first = get_string_value(result.table->view().column(0), 0);
  auto last = get_string_value(result.table->view().column(0), 24);
  EXPECT_EQ(first, "ALGERIA");
  EXPECT_EQ(last, "VIETNAM");
}

TEST(PlanExecutor, SortWithFetch) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("nation"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  std::vector<uint32_t> proj_cols{1};
  auto proj_vec = fbb.CreateVector(proj_cols);
  auto scan = fb::CreateGpuScan(fbb, paths, schema, proj_vec);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());

  auto sort_expr = make_col_ref(fbb, 0);
  auto sort_spec = fb::CreateSortExprNode(fbb, sort_expr, /*asc=*/true,
                                           /*nulls_first=*/false);
  auto sort_specs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::SortExprNode>>{sort_spec});

  // LIMIT 5
  auto sort = fb::CreateGpuSort(fbb, sort_specs, /*fetch=*/5, scan_node);
  auto sort_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuSort, sort.Union());
  auto buf = finish_plan(fbb, sort_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  EXPECT_EQ(result.table->num_rows(), 5);
}

TEST(PlanExecutor, AggregateCount) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("region"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto scan = fb::CreateGpuScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());

  // Aggregate: count(*) with no group-by.
  auto func_name = fbb.CreateString("count");
  auto func_alias = fbb.CreateString("count(*)");
  auto agg_func = fb::CreateAggregateFuncNode(fbb, func_name, /*args=*/0,
                                               /*distinct=*/false, func_alias);
  auto agg_funcs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::AggregateFuncNode>>{agg_func});

  auto agg = fb::CreateGpuAggregate(
      fbb, fb::AggregateMode_Single,
      /*group_exprs=*/0, /*group_names=*/0, agg_funcs, scan_node);
  auto agg_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuAggregate, agg.Union());
  auto buf = finish_plan(fbb, agg_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 1);
  ASSERT_EQ(result.table->num_rows(), 1);
  EXPECT_EQ(result.column_names[0], "count(*)");

  auto count = get_scalar_value<int64_t>(result.table->view().column(0), 0);
  EXPECT_EQ(count, 5);
}

TEST(PlanExecutor, AggregateGroupBy) {
  flatbuffers::FlatBufferBuilder fbb;

  // First: join nation and region, project r_name only.
  auto nation_path = fbb.CreateString(parquet_path("nation"));
  auto nation_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{nation_path});
  auto nation_schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  auto nation_scan = fb::CreateGpuScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, nation_scan.Union());

  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateGpuScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, region_scan.Union());

  // Join: n_regionkey (col 2) = r_regionkey (col 0)
  // Output projection: r_name only → col 5 in full output
  auto lk = make_col_ref(fbb, 2);
  auto rk = make_col_ref(fbb, 0);
  auto join_key = fb::CreateJoinKey(fbb, lk, rk);
  auto keys_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::JoinKey>>{join_key});
  std::vector<uint32_t> join_proj{5};
  auto join_proj_vec = fbb.CreateVector(join_proj);

  auto join = fb::CreateGpuHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node, join_proj_vec);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuHashJoin, join.Union());

  // Aggregate: GROUP BY r_name (col 0), count(*)
  auto group_expr = make_col_ref(fbb, 0);
  auto group_exprs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{group_expr});
  auto group_name = fbb.CreateString("r_name");
  auto group_names_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{group_name});

  auto func_name = fbb.CreateString("count");
  auto func_alias = fbb.CreateString("nation_count");
  auto agg_func = fb::CreateAggregateFuncNode(fbb, func_name, /*args=*/0,
                                               /*distinct=*/false, func_alias);
  auto agg_funcs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::AggregateFuncNode>>{agg_func});

  auto agg = fb::CreateGpuAggregate(
      fbb, fb::AggregateMode_Single, group_exprs, group_names_vec,
      agg_funcs, join_node);
  auto agg_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuAggregate, agg.Union());
  auto buf = finish_plan(fbb, agg_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  // 5 regions, each with 5 nations.
  ASSERT_EQ(result.table->num_columns(), 2);
  EXPECT_EQ(result.table->num_rows(), 5);
  EXPECT_EQ(result.column_names[0], "r_name");
  EXPECT_EQ(result.column_names[1], "nation_count");

  for (cudf::size_type i = 0; i < 5; ++i) {
    auto count =
        get_scalar_value<int64_t>(result.table->view().column(1), i);
    EXPECT_EQ(count, 5);
  }
}

TEST(PlanExecutor, ProjectRename) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("region"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto scan = fb::CreateGpuScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());

  // Project: select col 1 as "region_name", col 0 as "key"
  auto e1 = make_col_ref(fbb, 1);
  auto e2 = make_col_ref(fbb, 0);
  auto exprs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{e1, e2});
  auto a1 = fbb.CreateString("region_name");
  auto a2 = fbb.CreateString("key");
  auto aliases = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{a1, a2});

  auto proj = fb::CreateGpuProject(fbb, exprs, aliases, scan_node);
  auto proj_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuProject, proj.Union());
  auto buf = finish_plan(fbb, proj_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 2);
  EXPECT_EQ(result.table->num_rows(), 5);
  EXPECT_EQ(result.column_names[0], "region_name");
  EXPECT_EQ(result.column_names[1], "key");
}

TEST(PlanExecutor, PassthroughNodes) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("region"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto scan = fb::CreateGpuScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuScan, scan.Union());

  auto cb = fb::CreateGpuCoalesceBatches(fbb, /*target_batch_size=*/8192,
                                          scan_node);
  auto cb_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuCoalesceBatches, cb.Union());

  auto cp = fb::CreateGpuCoalescePartitions(fbb, cb_node);
  auto cp_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuCoalescePartitions, cp.Union());
  auto buf = finish_plan(fbb, cp_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 3);
  EXPECT_EQ(result.table->num_rows(), 5);
}

TEST(PlanExecutor, JoinProjectSort) {
  flatbuffers::FlatBufferBuilder fbb;

  // nation scan
  auto nation_path = fbb.CreateString(parquet_path("nation"));
  auto nation_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{nation_path});
  auto nation_schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  auto nation_scan = fb::CreateGpuScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, nation_scan.Union());

  // region scan
  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateGpuScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuScan, region_scan.Union());

  // Join: n_regionkey (col 2) = r_regionkey (col 0)
  // Project: n_name(1), r_name(5) from full join output
  auto lk = make_col_ref(fbb, 2);
  auto rk = make_col_ref(fbb, 0);
  auto join_key = fb::CreateJoinKey(fbb, lk, rk);
  auto keys_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::JoinKey>>{join_key});
  std::vector<uint32_t> join_proj{1, 5};
  auto join_proj_vec = fbb.CreateVector(join_proj);

  auto join = fb::CreateGpuHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node, join_proj_vec);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuHashJoin, join.Union());

  auto cb = fb::CreateGpuCoalesceBatches(fbb, /*target_batch_size=*/65536,
                                          join_node);
  auto cb_node = make_plan_node(
      fbb, fb::PlanNodeKind_GpuCoalesceBatches, cb.Union());

  auto pe1 = make_col_ref(fbb, 0);
  auto pe2 = make_col_ref(fbb, 1);
  auto proj_exprs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{pe1, pe2});
  auto pa1 = fbb.CreateString("n_name");
  auto pa2 = fbb.CreateString("r_name");
  auto proj_aliases = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{pa1, pa2});

  auto project = fb::CreateGpuProject(fbb, proj_exprs, proj_aliases, cb_node);
  auto proj_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuProject, project.Union());

  // Sort by n_name (col 0) ascending.
  auto sort_expr = make_col_ref(fbb, 0);
  auto sort_spec = fb::CreateSortExprNode(fbb, sort_expr, /*asc=*/true,
                                           /*nulls_first=*/false);
  auto sort_specs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::SortExprNode>>{sort_spec});
  auto sort = fb::CreateGpuSort(fbb, sort_specs, /*fetch=*/-1, proj_node);
  auto sort_node =
      make_plan_node(fbb, fb::PlanNodeKind_GpuSort, sort.Union());
  auto buf = finish_plan(fbb, sort_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 2);
  EXPECT_EQ(result.table->num_rows(), 25);
  EXPECT_EQ(result.column_names[0], "n_name");
  EXPECT_EQ(result.column_names[1], "r_name");

  // First nation alphabetically: ALGERIA.
  auto first_name = get_string_value(result.table->view().column(0), 0);
  EXPECT_EQ(first_name, "ALGERIA");
  // ALGERIA is in AFRICA.
  auto first_region = get_string_value(result.table->view().column(1), 0);
  EXPECT_EQ(first_region, "AFRICA");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  peacock_test::install_rmm_pool();
  return RUN_ALL_TESTS();
}
