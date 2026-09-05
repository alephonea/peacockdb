/// GPU tests for plan_executor: builds FlatBuffer plans programmatically and
/// executes them against testdata/tpch.minimal/ Parquet files.

#include "peacock_gpu.h"
#include "plan_executor.h"
#include "generated/gpu_plan_generated.h"

#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <filesystem>
#include <string>
#include <vector>

#include "peacock/rmm_pool.hpp"

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

template <>
double get_scalar_value<double>(const cudf::column_view& col,
                                cudf::size_type row) {
  std::vector<double> host(col.size());
  cudaMemcpy(host.data(), col.data<double>(),
             col.size() * sizeof(double), cudaMemcpyDeviceToHost);
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

  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto node = make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());
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

  auto scan = fb::CreateCudfScan(fbb, paths, schema, proj_vec);
  auto node = make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());
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
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto scan_node = make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  // Filter: n_regionkey (col 2) > 2. Cast the Int32 column to Int64 so it matches
  // the Int64 literal rather than relying on cuDF AST implicit promotion.
  auto col2 = make_col_ref(fbb, 2, "n_regionkey");
  auto cast_col2 = make_cast_expr(fbb, col2, fb::DataType_Int64);
  auto lit2 = make_int64_literal(fbb, 2);
  auto predicate = make_binary_expr(fbb, cast_col2, fb::BinaryOp_Gt, lit2);

  auto filter = fb::CreateCudfFilter(fbb, predicate, scan_node);
  auto filter_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfFilter, filter.Union());
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
  auto nation_scan = fb::CreateCudfScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, nation_scan.Union());

  // Right: region (r_regionkey, r_name, r_comment)
  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateCudfScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, region_scan.Union());

  // Join keys: n_regionkey (col 2 in left) = r_regionkey (col 0 in right)
  auto lk = make_col_ref(fbb, 2);
  auto rk = make_col_ref(fbb, 0);
  auto join_key = fb::CreateJoinKey(fbb, lk, rk);
  auto keys_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::JoinKey>>{join_key});

  auto join = fb::CreateCudfHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfHashJoin, join.Union());
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
  auto nation_scan = fb::CreateCudfScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, nation_scan.Union());

  // region
  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateCudfScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, region_scan.Union());

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

  auto join = fb::CreateCudfHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node, proj_vec);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfHashJoin, join.Union());
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
  auto scan = fb::CreateCudfScan(fbb, paths, schema, proj_vec);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  // Sort by col 0 (n_name) ascending.
  auto sort_expr = make_col_ref(fbb, 0);
  auto sort_spec = fb::CreateSortExprNode(fbb, sort_expr, /*asc=*/true,
                                           /*nulls_first=*/false);
  auto sort_specs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::SortExprNode>>{sort_spec});

  auto sort = fb::CreateCudfSort(fbb, sort_specs, /*fetch=*/-1, scan_node);
  auto sort_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfSort, sort.Union());
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
  auto scan = fb::CreateCudfScan(fbb, paths, schema, proj_vec);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  auto sort_expr = make_col_ref(fbb, 0);
  auto sort_spec = fb::CreateSortExprNode(fbb, sort_expr, /*asc=*/true,
                                           /*nulls_first=*/false);
  auto sort_specs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::SortExprNode>>{sort_spec});

  // LIMIT 5
  auto sort = fb::CreateCudfSort(fbb, sort_specs, /*fetch=*/5, scan_node);
  auto sort_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfSort, sort.Union());
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
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  // Aggregate: count(*) with no group-by.
  auto func_name = fbb.CreateString("count");
  auto func_alias = fbb.CreateString("count(*)");
  auto agg_func = fb::CreateAggregateFuncNode(fbb, func_name, /*args=*/0,
                                               /*distinct=*/false, func_alias);
  auto agg_funcs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::AggregateFuncNode>>{agg_func});

  auto agg = fb::CreateCudfAggregate(
      fbb, fb::AggregateMode_Single,
      /*group_exprs=*/0, /*group_names=*/0, agg_funcs, scan_node);
  auto agg_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfAggregate, agg.Union());
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
  auto nation_scan = fb::CreateCudfScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, nation_scan.Union());

  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateCudfScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, region_scan.Union());

  // Join: n_regionkey (col 2) = r_regionkey (col 0)
  // Output projection: r_name only → col 5 in full output
  auto lk = make_col_ref(fbb, 2);
  auto rk = make_col_ref(fbb, 0);
  auto join_key = fb::CreateJoinKey(fbb, lk, rk);
  auto keys_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::JoinKey>>{join_key});
  std::vector<uint32_t> join_proj{5};
  auto join_proj_vec = fbb.CreateVector(join_proj);

  auto join = fb::CreateCudfHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node, join_proj_vec);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfHashJoin, join.Union());

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

  auto agg = fb::CreateCudfAggregate(
      fbb, fb::AggregateMode_Single, group_exprs, group_names_vec,
      agg_funcs, join_node);
  auto agg_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfAggregate, agg.Union());
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
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  // Project: select col 1 as "region_name", col 0 as "key"
  auto e1 = make_col_ref(fbb, 1);
  auto e2 = make_col_ref(fbb, 0);
  auto exprs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{e1, e2});
  auto a1 = fbb.CreateString("region_name");
  auto a2 = fbb.CreateString("key");
  auto aliases = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{a1, a2});

  auto proj = fb::CreateCudfProject(fbb, exprs, aliases, scan_node);
  auto proj_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfProject, proj.Union());
  auto buf = finish_plan(fbb, proj_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 2);
  EXPECT_EQ(result.table->num_rows(), 5);
  EXPECT_EQ(result.column_names[0], "region_name");
  EXPECT_EQ(result.column_names[1], "key");
}

/// The fifth UnaryOp, appended so a finalizing aggregate can carry its own finalize
/// rather than leaving the arithmetic to this side. Two tests because the expression's
/// shape decides which evaluator runs it, and the enum arm was added to both: an AST-able
/// expression goes through cudf::ast, and anything the AST cannot express — a CASE, which
/// is what a stddev's finalize wraps its root in — goes through the column path.
TEST(PlanExecutor, ProjectSqrtThroughTheAst) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("region"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  // sqrt(CAST(r_regionkey AS FLOAT64)) — a cast to FLOAT64 is one of the two the AST
  // has, so the whole expression stays AST-able.
  auto casted = make_cast_expr(fbb, make_col_ref(fbb, 0), fb::DataType_Float64);
  auto un = fb::CreateUnaryExprNode(fbb, fb::UnaryOp_Sqrt, casted);
  auto root_sqrt = fb::CreateExpr(fbb, fb::ExprNode_UnaryExprNode, un.Union());
  auto exprs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{root_sqrt});
  auto alias = fbb.CreateString("root");
  auto aliases = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{alias});
  auto proj = fb::CreateCudfProject(fbb, exprs, aliases, scan_node);
  auto proj_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfProject, proj.Union());
  auto buf = finish_plan(fbb, proj_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 1);
  ASSERT_EQ(result.table->num_rows(), 5);
  auto col = result.table->view().column(0);
  ASSERT_EQ(col.type().id(), cudf::type_id::FLOAT64);
  // r_regionkey is 0..4 in tpch.minimal, so the roots are known exactly.
  for (cudf::size_type row = 0; row < 5; ++row) {
    EXPECT_NEAR(get_scalar_value<double>(col, row), std::sqrt(double(row)), 1e-12)
        << "row " << row;
  }
}

TEST(PlanExecutor, ProjectSqrtThroughTheColumnPath) {
  flatbuffers::FlatBufferBuilder fbb;

  auto path = fbb.CreateString(parquet_path("region"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  // sqrt(CASE WHEN r_regionkey >= 0 THEN CAST(r_regionkey AS FLOAT64) ELSE 0.0 END).
  // The CASE is what routes it: the AST has no such node, so is_ast_able says no for the
  // whole expression and the column evaluator takes it.
  auto when = make_binary_expr(fbb, make_col_ref(fbb, 0), fb::BinaryOp_GtEq,
                               make_int64_literal(fbb, 0));
  auto then = make_cast_expr(fbb, make_col_ref(fbb, 0), fb::DataType_Float64);
  auto arm = fb::CreateCaseWhenThen(fbb, when, then);
  auto arms = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::CaseWhenThen>>{arm});
  auto otherwise = make_float64_literal(fbb, 0.0);
  auto case_node = fb::CreateCaseExprNode(
      fbb, /*expr=*/flatbuffers::Offset<fb::Expr>{}, arms, otherwise);
  auto case_expr =
      fb::CreateExpr(fbb, fb::ExprNode_CaseExprNode, case_node.Union());
  auto un = fb::CreateUnaryExprNode(fbb, fb::UnaryOp_Sqrt, case_expr);
  auto root_sqrt = fb::CreateExpr(fbb, fb::ExprNode_UnaryExprNode, un.Union());
  auto exprs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{root_sqrt});
  auto alias = fbb.CreateString("root");
  auto aliases = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{alias});
  auto proj = fb::CreateCudfProject(fbb, exprs, aliases, scan_node);
  auto proj_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfProject, proj.Union());
  auto buf = finish_plan(fbb, proj_node);

  auto result = peacock::execute_plan(buf.data(), buf.size());

  ASSERT_EQ(result.table->num_columns(), 1);
  ASSERT_EQ(result.table->num_rows(), 5);
  auto col = result.table->view().column(0);
  ASSERT_EQ(col.type().id(), cudf::type_id::FLOAT64);
  for (cudf::size_type row = 0; row < 5; ++row) {
    EXPECT_NEAR(get_scalar_value<double>(col, row), std::sqrt(double(row)), 1e-12)
        << "row " << row;
  }
}

// --- AggregateMode_Merge: state in, state out -------------------------------
//
// The mode exists because this engine merges twice — once per lane, once across
// lanes — and finalizes in a project of its own, so a merge must not finalize. Each
// path through the arm gets its own test: the Welford triple, an avg's (sum, count)
// pair, and the one-column aggregates.
//
// The shape is the same in all three: one partial, then the SAME partial unioned with
// itself and merged. Merging a state with an identical copy of itself has an answer
// that needs no knowledge of the data — the counts and the sums double, the mean does
// not move — so the assertions are about the merge rather than about nation.parquet.

/// The nation scan every case below aggregates.
static flatbuffers::Offset<fb::PlanNode> nation_scan_node(
    flatbuffers::FlatBufferBuilder& fbb) {
  auto path = fbb.CreateString(parquet_path("nation"));
  auto paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
      {"n_nationkey", fb::DataType_Int32},
      {"n_name", fb::DataType_Utf8View},
      {"n_regionkey", fb::DataType_Int32},
      {"n_comment", fb::DataType_Utf8View},
  });
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  return make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());
}

/// GROUP BY n_regionkey with one aggregate over n_nationkey, in the given mode.
///
/// The ordinals differ by mode and that is the whole point of the pair: a Partial reads
/// the scan, where the key is column 2 and the value column 0, while a Merge reads the
/// Partial's own output, where the key is column 0 and the state begins at column 1.
static flatbuffers::Offset<fb::PlanNode> nation_aggregate(
    flatbuffers::FlatBufferBuilder& fbb, flatbuffers::Offset<fb::PlanNode> input,
    fb::AggregateMode mode, const char* func, bool mergeable) {
  bool merging = mode == fb::AggregateMode_Merge;
  auto group = make_col_ref(fbb, merging ? 0 : 2, "n_regionkey");
  auto groups = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{group});
  auto group_name = fbb.CreateString("n_regionkey");
  auto group_names = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{group_name});
  // A merge reads its state columns positionally from just past the keys, so the
  // argument matters only to the partial; it is written either way because the field is
  // not optional.
  auto arg = make_col_ref(fbb, merging ? 1 : 0, "n_nationkey");
  auto args = fbb.CreateVector(std::vector<flatbuffers::Offset<fb::Expr>>{arg});
  auto name = fbb.CreateString(func);
  auto alias = fbb.CreateString("state");
  auto agg_func = fb::CreateAggregateFuncNode(fbb, name, args, /*distinct=*/false,
                                              alias);
  auto funcs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::AggregateFuncNode>>{agg_func});
  fb::CudfAggregateBuilder builder(fbb);
  builder.add_mode(mode);
  builder.add_group_exprs(groups);
  builder.add_group_names(group_names);
  builder.add_aggr_funcs(funcs);
  builder.add_input(input);
  builder.add_mergeable_agg_state(mergeable);
  auto agg = builder.Finish();
  return make_plan_node(fbb, fb::PlanNodeKind_CudfAggregate, agg.Union());
}

/// The aggregate columns of one execution, keyed by the group they belong to, so two
/// runs are compared by key rather than by the order cuDF happened to group in.
static std::map<int32_t, std::vector<double>> by_group(
    const peacock::TableResult& result) {
  std::map<int32_t, std::vector<double>> rows;
  auto view = result.table->view();
  auto keys = view.column(0);
  for (cudf::size_type row = 0; row < view.num_rows(); ++row) {
    std::vector<double> values;
    for (cudf::size_type c = 1; c < view.num_columns(); ++c) {
      auto col = view.column(c);
      switch (col.type().id()) {
        case cudf::type_id::INT32:
          values.push_back(double(get_scalar_value<int32_t>(col, row)));
          break;
        case cudf::type_id::INT64:
          values.push_back(double(get_scalar_value<int64_t>(col, row)));
          break;
        default:
          values.push_back(get_scalar_value<double>(col, row));
          break;
      }
    }
    rows[get_scalar_value<int32_t>(keys, row)] = std::move(values);
  }
  return rows;
}

/// One partial, and the same partial unioned with itself and merged.
static std::pair<std::map<int32_t, std::vector<double>>,
                 std::map<int32_t, std::vector<double>>>
partial_and_merged(const char* func, bool mergeable) {
  std::map<int32_t, std::vector<double>> partial_rows;
  {
    flatbuffers::FlatBufferBuilder fbb;
    auto partial = nation_aggregate(fbb, nation_scan_node(fbb),
                                    fb::AggregateMode_Partial, func, mergeable);
    auto buf = finish_plan(fbb, partial);
    partial_rows = by_group(peacock::execute_plan(buf.data(), buf.size()));
  }
  std::map<int32_t, std::vector<double>> merged_rows;
  {
    flatbuffers::FlatBufferBuilder fbb;
    auto left = nation_aggregate(fbb, nation_scan_node(fbb),
                                 fb::AggregateMode_Partial, func, mergeable);
    auto right = nation_aggregate(fbb, nation_scan_node(fbb),
                                  fb::AggregateMode_Partial, func, mergeable);
    auto inputs = fbb.CreateVector(
        std::vector<flatbuffers::Offset<fb::PlanNode>>{left, right});
    auto both = fb::CreateCudfUnion(fbb, inputs);
    auto both_node = make_plan_node(fbb, fb::PlanNodeKind_CudfUnion, both.Union());
    auto merged = nation_aggregate(fbb, both_node, fb::AggregateMode_Merge, func,
                                   mergeable);
    auto buf = finish_plan(fbb, merged);
    merged_rows = by_group(peacock::execute_plan(buf.data(), buf.size()));
  }
  return {partial_rows, merged_rows};
}

TEST(AggregateMerge, WelfordStateComesBackAsStateAndNotAsAValue) {
  auto [partial, merged] = partial_and_merged("stddev", /*mergeable=*/true);
  ASSERT_EQ(partial.size(), 5u) << "five regions";
  ASSERT_EQ(merged.size(), partial.size());
  for (auto& [key, state] : partial) {
    ASSERT_EQ(state.size(), 3u) << "the partial emits [count, mean, m2]";
    auto& after = merged.at(key);
    ASSERT_EQ(after.size(), 3u) << "and so does the merge — a value would be one column";
    EXPECT_DOUBLE_EQ(after[0], state[0] * 2) << "group " << key;
    EXPECT_NEAR(after[1], state[1], 1e-9) << "two identical halves keep the mean";
    EXPECT_NEAR(after[2], state[2] * 2, 1e-9) << "and add their M2s, the means being equal";
  }
}

TEST(AggregateMerge, TheMergedCountIsWidenedSoASecondMergeReadsTheSameLayout) {
  // cuDF's group_merge_m2 takes an INT32 count and hands one back; a chain of merges
  // only works if the width the arm emits is the width the next arm reads.
  flatbuffers::FlatBufferBuilder fbb;
  auto left = nation_aggregate(fbb, nation_scan_node(fbb),
                               fb::AggregateMode_Partial, "stddev", true);
  auto right = nation_aggregate(fbb, nation_scan_node(fbb),
                                fb::AggregateMode_Partial, "stddev", true);
  auto inputs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::PlanNode>>{left, right});
  auto both = fb::CreateCudfUnion(fbb, inputs);
  auto both_node = make_plan_node(fbb, fb::PlanNodeKind_CudfUnion, both.Union());
  auto once = nation_aggregate(fbb, both_node, fb::AggregateMode_Merge, "stddev", true);
  auto twice = nation_aggregate(fbb, once, fb::AggregateMode_Merge, "stddev", true);
  auto buf = finish_plan(fbb, twice);

  auto result = peacock::execute_plan(buf.data(), buf.size());
  auto view = result.table->view();
  ASSERT_EQ(view.num_columns(), 4) << "the key and the three state columns";
  EXPECT_EQ(view.column(1).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(view.column(2).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(view.column(3).type().id(), cudf::type_id::FLOAT64);
}

TEST(AggregateMerge, AnAvgsSumAndCountBothSurviveTheMerge) {
  auto [partial, merged] = partial_and_merged("avg", /*mergeable=*/false);
  ASSERT_EQ(partial.size(), 5u);
  for (auto& [key, state] : partial) {
    ASSERT_EQ(state.size(), 2u) << "the partial emits [sum, count]";
    auto& after = merged.at(key);
    ASSERT_EQ(after.size(), 2u) << "the merge emits both, since the divide is the finalize";
    EXPECT_DOUBLE_EQ(after[0], state[0] * 2) << "group " << key;
    EXPECT_DOUBLE_EQ(after[1], state[1] * 2) << "group " << key;
  }
}

TEST(AggregateMerge, AOneColumnAggregateMergesByItsOwnRule) {
  // sum merges by sum, and count by sum — the one place a merge is not the same
  // aggregation as the partial it merges.
  for (const char* func : {"sum", "count"}) {
    auto [partial, merged] = partial_and_merged(func, /*mergeable=*/false);
    ASSERT_EQ(partial.size(), 5u) << func;
    for (auto& [key, state] : partial) {
      ASSERT_EQ(state.size(), 1u) << func;
      EXPECT_DOUBLE_EQ(merged.at(key)[0], state[0] * 2) << func << " group " << key;
    }
  }
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
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto scan_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());

  auto cb = fb::CreateCudfCoalesceBatches(fbb, /*target_batch_size=*/8192,
                                          scan_node);
  auto cb_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfCoalesceBatches, cb.Union());

  auto cp = fb::CreateCudfCoalescePartitions(fbb, cb_node);
  auto cp_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfCoalescePartitions, cp.Union());
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
  auto nation_scan = fb::CreateCudfScan(fbb, nation_paths, nation_schema);
  auto nation_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, nation_scan.Union());

  // region scan
  auto region_path = fbb.CreateString(parquet_path("region"));
  auto region_paths = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{region_path});
  auto region_schema = make_schema(fbb, {
      {"r_regionkey", fb::DataType_Int32},
      {"r_name", fb::DataType_Utf8View},
      {"r_comment", fb::DataType_Utf8View},
  });
  auto region_scan = fb::CreateCudfScan(fbb, region_paths, region_schema);
  auto region_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfScan, region_scan.Union());

  // Join: n_regionkey (col 2) = r_regionkey (col 0)
  // Project: n_name(1), r_name(5) from full join output
  auto lk = make_col_ref(fbb, 2);
  auto rk = make_col_ref(fbb, 0);
  auto join_key = fb::CreateJoinKey(fbb, lk, rk);
  auto keys_vec = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::JoinKey>>{join_key});
  std::vector<uint32_t> join_proj{1, 5};
  auto join_proj_vec = fbb.CreateVector(join_proj);

  auto join = fb::CreateCudfHashJoin(
      fbb, fb::JoinType_Inner, keys_vec,
      /*filter=*/0, /*filter_columns=*/0, nation_node, region_node, join_proj_vec);
  auto join_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfHashJoin, join.Union());

  auto cb = fb::CreateCudfCoalesceBatches(fbb, /*target_batch_size=*/65536,
                                          join_node);
  auto cb_node = make_plan_node(
      fbb, fb::PlanNodeKind_CudfCoalesceBatches, cb.Union());

  auto pe1 = make_col_ref(fbb, 0);
  auto pe2 = make_col_ref(fbb, 1);
  auto proj_exprs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::Expr>>{pe1, pe2});
  auto pa1 = fbb.CreateString("n_name");
  auto pa2 = fbb.CreateString("r_name");
  auto proj_aliases = fbb.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{pa1, pa2});

  auto project = fb::CreateCudfProject(fbb, proj_exprs, proj_aliases, cb_node);
  auto proj_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfProject, project.Union());

  // Sort by n_name (col 0) ascending.
  auto sort_expr = make_col_ref(fbb, 0);
  auto sort_spec = fb::CreateSortExprNode(fbb, sort_expr, /*asc=*/true,
                                           /*nulls_first=*/false);
  auto sort_specs = fbb.CreateVector(
      std::vector<flatbuffers::Offset<fb::SortExprNode>>{sort_spec});
  auto sort = fb::CreateCudfSort(fbb, sort_specs, /*fetch=*/-1, proj_node);
  auto sort_node =
      make_plan_node(fbb, fb::PlanNodeKind_CudfSort, sort.Union());
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

// ---------------------------------------------------------------------------
// Reading a scan one row-group subset at a time, exporting a row range of a handle,
// and slicing one. Value-level checks go through NodeSession, which is what the C
// entry points are thin wrappers over; the entry points themselves are checked for
// what they own (a refused empty list, the empty-export convention), and end to end
// from Rust on the GPU host.
// ---------------------------------------------------------------------------

/// customer.parquet projected to `projection` — two row groups (122880 + 27120), the
/// only committed fixture with more than one. Column 0 is c_custkey, the narrow one to
/// read back; column 1 is c_name, the one with content bytes to charge for.
static std::vector<uint8_t> customer_scan_plan(flatbuffers::FlatBufferBuilder& fbb,
                                               const std::vector<uint32_t>& projection) {
  auto path = fbb.CreateString(parquet_path("customer"));
  auto paths = fbb.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
                                     {"c_custkey", fb::DataType_Int64},
                                     {"c_name", fb::DataType_Utf8View},
                                     {"c_address", fb::DataType_Utf8View},
                                     {"c_nationkey", fb::DataType_Int32},
                                     {"c_phone", fb::DataType_Utf8View},
                                     {"c_acctbal", fb::DataType_Decimal128},
                                     {"c_mktsegment", fb::DataType_Utf8View},
                                     {"c_comment", fb::DataType_Utf8View},
                                 });
  auto scan = fb::CreateCudfScan(fbb, paths, schema, fbb.CreateVector(projection));
  return finish_plan(fbb, make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union()));
}

static std::vector<int64_t> host_int64_column(const cudf::column_view& col) {
  std::vector<int64_t> host(col.size());
  cudaMemcpy(host.data(), col.data<int64_t>(), col.size() * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  return host;
}

static std::vector<int64_t> keys_of(const peacock::TableResult& result) {
  return host_int64_column(result.table->view().column(0));
}

/// The C ABI over one loaded plan, released in the order the header requires.
class CApiPlan {
 public:
  explicit CApiPlan(const std::vector<uint8_t>& plan) {
    EXPECT_EQ(peacock_executor_create(/*gpu_memory_limit=*/0, &ex_), 0);
    uint64_t nodes = 0;
    EXPECT_EQ(peacock_executor_begin_plan(ex_, plan.data(), plan.size(), &nodes), 0);
  }
  ~CApiPlan() {
    peacock_executor_end_plan(ex_);
    peacock_executor_destroy(ex_);
  }
  peacock_executor_t* get() { return ex_; }
  std::string last_error() { return peacock_last_error(ex_); }

 private:
  peacock_executor_t* ex_ = nullptr;
};

TEST(ScanRowGroups, SubsetsUnionToTheWholeScan) {
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  auto whole = keys_of(peacock::execute_plan(buf.data(), buf.size()));

  peacock::NodeSession session(buf.data(), buf.size());
  peacock::NodeStats first_stats{}, second_stats{};
  std::vector<uint32_t> first{0}, second{1};
  // The same seq, twice: the scan arm is stateless per call, which is what lets one
  // node be a batch loader at all.
  uint64_t h0 = session.execute_scan_rowgroups(0, first, &first_stats);
  uint64_t h1 = session.execute_scan_rowgroups(0, second, &second_stats);

  auto keys = keys_of(session.table_for(h0));
  auto rest = keys_of(session.table_for(h1));
  EXPECT_EQ(first_stats.rows, keys.size());
  EXPECT_EQ(second_stats.rows, rest.size());
  keys.insert(keys.end(), rest.begin(), rest.end());
  EXPECT_EQ(keys, whole);
}

// ---------------------------------------------------------------------------
// The measurement side: what a region records, and the coordinate that tells two
// calls of one seq apart. Nothing on the execution path reads any of it, so these
// are the only checks it has.
// ---------------------------------------------------------------------------

/// A timing loan, so a failing expectation cannot leave the process-global switch on for
/// every later test in this binary.
struct TimingOn {
  TimingOn() { peacock::set_node_timing(peacock::NodeTiming::Events); }
  ~TimingOn() { peacock::set_node_timing(peacock::NodeTiming::Off); }
};

TEST(NodeRegions, CallIndexCountsCallsOfOneSeq) {
  TimingOn timing;
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  peacock::NodeSession session(buf.data(), buf.size());

  std::vector<uint32_t> first{0}, second{1};
  session.execute_scan_rowgroups(0, first, nullptr);
  session.execute_scan_rowgroups(0, second, nullptr);

  auto regions = session.collect_node_regions();
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_EQ(regions[0].seq, 0u);
  EXPECT_EQ(regions[1].seq, 0u);
  // The whole point of the field: without it the two rows are indistinguishable, and a
  // record row cannot be matched to the call it describes.
  EXPECT_EQ(regions[0].call_index, 0u);
  EXPECT_EQ(regions[1].call_index, 1u);
}

TEST(NodeRegions, ANewSessionStartsTheCountAgain) {
  TimingOn timing;
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  std::vector<uint32_t> groups{0};

  // Two sessions over the same plan, as a benchmark's repeated runs are: the index is a
  // coordinate WITHIN a run, so the second must not continue the first.
  for (int run = 0; run < 2; ++run) {
    peacock::NodeSession session(buf.data(), buf.size());
    session.execute_scan_rowgroups(0, groups, nullptr);
    auto regions = session.collect_node_regions();
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_EQ(regions[0].call_index, 0u) << "run " << run;
  }
}

TEST(NodeRegions, ARegionCarriesWhatOnlyAMeasurementReads) {
  TimingOn timing;
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0, 1});
  peacock::NodeSession session(buf.data(), buf.size());

  std::vector<uint32_t> groups{0};
  peacock::NodeStats stats{};
  session.execute_scan_rowgroups(0, groups, &stats);
  auto regions = session.collect_node_regions();
  ASSERT_EQ(regions.size(), 1u);

  // `NodeStats` keeps what the driver reads and nothing else; everything below travels
  // by collection instead, which is what keeps a shipping query from paying per call.
  EXPECT_GT(stats.rows, 0u);
  EXPECT_GT(regions[0].host_setup_us + regions[0].host_submit_us, 0u);
  EXPECT_GT(regions[0].device_us, 0u);
  EXPECT_GT(regions[0].logical_bytes, 0u);
}

TEST(NodeRegions, CollectingTwiceReportsNothingTheSecondTime) {
  TimingOn timing;
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  peacock::NodeSession session(buf.data(), buf.size());

  std::vector<uint32_t> groups{0};
  session.execute_scan_rowgroups(0, groups, nullptr);
  EXPECT_EQ(session.collect_node_regions().size(), 1u);
  // A second call must not report the same regions again, and a long session must not
  // accumulate events without bound.
  EXPECT_TRUE(session.collect_node_regions().empty());
}

TEST(NodeRegions, TimingOffRecordsNothing) {
  // No loan: the switch is already off, and this test is about it staying that way.
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  peacock::NodeSession session(buf.data(), buf.size());

  std::vector<uint32_t> groups{0};
  peacock::NodeStats stats{};
  session.execute_scan_rowgroups(0, groups, &stats);
  // The driver's two numbers are always there; the measurement is not allocated at all.
  EXPECT_GT(stats.rows, 0u);
  EXPECT_TRUE(session.collect_node_regions().empty());
}

TEST(ScanRowGroups, ACallOnAnotherKindOfNodeSaysWhichKind) {
  flatbuffers::FlatBufferBuilder fbb;
  auto path = fbb.CreateString(parquet_path("region"));
  auto paths = fbb.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{path});
  auto schema = make_schema(fbb, {
                                     {"r_regionkey", fb::DataType_Int32},
                                     {"r_name", fb::DataType_Utf8View},
                                     {"r_comment", fb::DataType_Utf8View},
                                 });
  auto scan = fb::CreateCudfScan(fbb, paths, schema);
  auto scan_node = make_plan_node(fbb, fb::PlanNodeKind_CudfScan, scan.Union());
  auto cb = fb::CreateCudfCoalesceBatches(fbb, /*target_batch_size=*/8192, scan_node);
  auto buf =
      finish_plan(fbb, make_plan_node(fbb, fb::PlanNodeKind_CudfCoalesceBatches, cb.Union()));

  peacock::NodeSession session(buf.data(), buf.size());
  std::vector<uint32_t> groups{0};
  try {
    session.execute_scan_rowgroups(/*seq=*/1, groups, nullptr);
    FAIL() << "a coalesce-batches seq was read as a scan";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("CudfCoalesceBatches"), std::string::npos) << e.what();
  }
}

TEST(ScanRowGroups, AnEmptyListIsRefused) {
  flatbuffers::FlatBufferBuilder fbb, fbb2;
  auto buf = customer_scan_plan(fbb, {0});
  CApiPlan plan(buf);

  std::vector<uint32_t> groups{0};  // a real pointer, and no groups to read
  uint64_t handle = 0;
  EXPECT_NE(peacock_executor_execute_scan_rowgroups(plan.get(), 0, groups.data(),
                                                    /*n=*/0, &handle, nullptr),
            0);
  EXPECT_NE(plan.last_error().find("empty row-group list"), std::string::npos) << plan.last_error();

  // And at the session, which is the layer a C++ caller meets first: one level down an
  // empty override reads as "no override", so refusing here is what stops a caller who
  // named a set from getting a whole-table read.
  auto buf2 = customer_scan_plan(fbb2, {0});
  peacock::NodeSession session(buf2.data(), buf2.size());
  EXPECT_ANY_THROW(session.execute_scan_rowgroups(0, {}, nullptr));
}

TEST(ScanRowGroups, TheStatsCarryTheVarlenBytes) {
  // The other two stats fields, which the one-column fixture cannot see: a string column
  // is what varlen_content_bytes is for, and the accountant prices a batch from it — a
  // zero there under-prices every string batch and the budget stops binding.
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0, 1});
  peacock::NodeSession session(buf.data(), buf.size());

  peacock::NodeStats stats{};
  std::vector<uint32_t> groups{1};
  uint64_t handle = session.execute_scan_rowgroups(0, groups, &stats);
  const auto& table = session.table_for(handle).table->view();
  EXPECT_EQ(stats.rows, static_cast<uint64_t>(table.num_rows()));
  EXPECT_EQ(table.num_columns(), 2);
  // c_name averages well over a byte per row, so any plausible reading clears the rows.
  EXPECT_GT(stats.varlen_content_bytes, stats.rows);
}

TEST(ScanRowGroups, AnOutOfRangeIndexSaysWhichCall) {
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  peacock::NodeSession session(buf.data(), buf.size());

  // customer.parquet has two row groups. cuDF refuses the tenth in its own words, which
  // name neither the node nor the list — the rethrow is what makes a planner defect
  // readable at the layer that produced the index.
  std::vector<uint32_t> beyond{9};
  try {
    session.execute_scan_rowgroups(0, beyond, nullptr);
    FAIL() << "an out-of-range row group was read as something";
  } catch (const std::exception& e) {
    const std::string said = e.what();
    EXPECT_NE(said.find("seq 0"), std::string::npos) << said;
    EXPECT_NE(said.find("[9]"), std::string::npos) << said;
  }
}

TEST(RangedExport, TheRangesAreTheRowsNamed) {
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  CApiPlan plan(buf);

  std::vector<uint32_t> groups{1};  // the smaller row group: 27120 rows
  uint64_t handle = 0;
  PeacockNodeStats stats{};
  ASSERT_EQ(peacock_executor_execute_scan_rowgroups(plan.get(), 0, groups.data(), groups.size(),
                                                    &handle, &stats),
            0);
  const uint64_t rows = stats.rows;
  ASSERT_GT(rows, 100u);

  auto export_range = [&](uint64_t offset, uint64_t length) {
    uint8_t* ipc = nullptr;
    uint64_t len = 0;
    EXPECT_EQ(peacock_result_from_handle(plan.get(), handle, offset, length, &ipc, &len), 0);
    std::vector<uint8_t> bytes;
    if (len > 0) bytes.assign(ipc, ipc + len);
    peacock_result_free(ipc);
    return bytes;
  };

  // To-the-end is the whole table, which is what every legacy caller asks for.
  EXPECT_EQ(export_range(0, UINT64_MAX), export_range(0, rows));
  // A fetch running past the end clamps rather than failing — the straddling batch.
  EXPECT_EQ(export_range(rows - 10, 1000), export_range(rows - 10, 10));
  // An offset at or past the end ships nothing at all.
  EXPECT_TRUE(export_range(rows, 10).empty());
  EXPECT_TRUE(export_range(rows + 5, UINT64_MAX).empty());
  EXPECT_TRUE(export_range(0, 0).empty());
  // A range inside the table is neither of those, and differs from the whole.
  auto head = export_range(0, 10);
  EXPECT_FALSE(head.empty());
  EXPECT_NE(head, export_range(0, rows));

  // An EMPTY table is the other side of that rule and still exports its schema, which is
  // what keeps whole-table callers' bytes what they always were. Consumes `handle`, so
  // it goes last.
  uint64_t empty = 0;
  ASSERT_EQ(peacock_executor_slice_handle(plan.get(), handle, rows, 0, &empty), 0);
  uint8_t* ipc = nullptr;
  uint64_t len = 0;
  EXPECT_EQ(peacock_result_from_handle(plan.get(), empty, 0, UINT64_MAX, &ipc, &len), 0);
  EXPECT_GT(len, 0u);
  peacock_result_free(ipc);

  // A failed export leaves the session standing, unlike the node-shaped calls: it read a
  // handle and touched nothing.
  EXPECT_NE(peacock_result_from_handle(plan.get(), /*handle=*/999, 0, UINT64_MAX, &ipc, &len), 0);
  EXPECT_EQ(peacock_result_from_handle(plan.get(), empty, 0, UINT64_MAX, &ipc, &len), 0);
  peacock_result_free(ipc);
}

TEST(SliceHandle, KeepsTheRowsNamedAndConsumesItsInput) {
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  peacock::NodeSession session(buf.data(), buf.size());
  std::vector<uint32_t> groups{1};

  uint64_t whole = session.execute_scan_rowgroups(0, groups, nullptr);
  auto keys = keys_of(session.table_for(whole));
  const uint64_t rows = keys.size();

  uint64_t head_handle = session.execute_scan_rowgroups(0, groups, nullptr);
  uint64_t tail_handle = session.execute_scan_rowgroups(0, groups, nullptr);
  uint64_t head = session.slice_handle(head_handle, 0, 100);
  uint64_t tail = session.slice_handle(tail_handle, 100, UINT64_MAX);

  auto sliced = keys_of(session.table_for(head));
  auto rest = keys_of(session.table_for(tail));
  EXPECT_EQ(sliced.size(), 100u);
  EXPECT_EQ(rest.size(), rows - 100);
  sliced.insert(sliced.end(), rest.begin(), rest.end());
  EXPECT_EQ(sliced, keys);

  // The input is gone: reading it and re-slicing it both fail, rather than the
  // second consumer getting a table the first one owns.
  EXPECT_ANY_THROW(session.table_for(head_handle));
  EXPECT_ANY_THROW(session.slice_handle(head_handle, 0, 10));
}

TEST(SliceHandle, ClampsAndEmpties) {
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  peacock::NodeSession session(buf.data(), buf.size());
  std::vector<uint32_t> groups{1};

  uint64_t sized = session.execute_scan_rowgroups(0, groups, nullptr);
  const uint64_t rows = session.table_for(sized).table->view().num_rows();

  uint64_t past = session.slice_handle(sized, rows, 10);
  // Empty but still a table: an empty slice keeps the columns, unlike the export,
  // which has a caller-visible "no bytes" convention instead.
  EXPECT_EQ(session.table_for(past).table->view().num_rows(), 0);
  EXPECT_EQ(session.table_for(past).table->view().num_columns(), 1);
  EXPECT_EQ(session.table_for(past).column_names[0], "c_custkey");

  uint64_t whole = session.execute_scan_rowgroups(0, groups, nullptr);
  uint64_t clamped = session.slice_handle(whole, rows - 10, 1000);
  EXPECT_EQ(session.table_for(clamped).table->view().num_rows(), 10);
}

TEST(SliceHandle, AnUnknownHandleFails) {
  flatbuffers::FlatBufferBuilder fbb;
  auto buf = customer_scan_plan(fbb, {0});
  CApiPlan plan(buf);

  uint64_t out = 0;
  EXPECT_NE(peacock_executor_slice_handle(plan.get(), /*handle=*/999, 0, 10, &out), 0);
  EXPECT_NE(plan.last_error().find("unknown input handle"), std::string::npos) << plan.last_error();

  // And the aftermath the driver leans on: the plan is gone with every handle it held,
  // so a release on this path is a no-op rather than a use of a dead handle.
  std::vector<uint32_t> groups{1};
  uint64_t handle = 0;
  EXPECT_NE(peacock_executor_execute_scan_rowgroups(plan.get(), 0, groups.data(), groups.size(),
                                                    &handle, nullptr),
            0);
  EXPECT_NE(plan.last_error().find("no plan loaded"), std::string::npos) << plan.last_error();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  peacock::install_rmm_pool();
  return RUN_ALL_TESTS();
}
