#include <peacock_gpu.h>
#include "plan_executor.h"
#include "plan_executor_internal.h"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

TEST(PeacockGpu, Version) {
  EXPECT_STREQ(peacock_gpu_version(), "0.1.0");
}

// Host-only unit tests for the decimal-scale / AST-routing helpers: they run on the
// non-GPU CI tiers (every push), where the GPU result tests do not, so a regression
// in these pure rules cannot slip through.

namespace {
namespace fb = peacock::plan;

cudf::data_type dec(int32_t scale) {
  return cudf::data_type{cudf::type_id::DECIMAL128, scale};
}
}  // namespace

// binop_output_type: predicates → BOOL8; fixed_point arithmetic result scale
// must follow DataFusion (cuDF scale = base-10 exponent, negative for
// fractional). A MUL/DIV swap here is exactly the regression this guards.
TEST(DecimalScale, BinopOutputType) {
  // Comparisons / logical ops → boolean, regardless of operand types.
  EXPECT_EQ(peacock::binop_output_type(fb::BinaryOp_Gt, dec(-2), dec(-2)).id(),
            cudf::type_id::BOOL8);
  EXPECT_EQ(peacock::binop_output_type(fb::BinaryOp_Eq, dec(-4), dec(-2)).id(),
            cudf::type_id::BOOL8);

  // ADD / SUB: result scale = min(exponent) = max fractional digits.
  EXPECT_EQ(peacock::binop_output_type(fb::BinaryOp_Plus, dec(-2), dec(-4)).scale(), -4);
  EXPECT_EQ(peacock::binop_output_type(fb::BinaryOp_Minus, dec(-4), dec(-2)).scale(), -4);

  // MUL: exponents add.
  EXPECT_EQ(peacock::binop_output_type(fb::BinaryOp_Multiply, dec(-2), dec(-3)).scale(), -5);

  // DIV: exponents subtract.
  EXPECT_EQ(peacock::binop_output_type(fb::BinaryOp_Divide, dec(-6), dec(-2)).scale(), -4);

  // Decimal arithmetic stays DECIMAL128.
  EXPECT_EQ(peacock::binop_output_type(fb::BinaryOp_Multiply, dec(-2), dec(-3)).id(),
            cudf::type_id::DECIMAL128);
}

namespace {

// Build a standalone fb::Expr (column ref) and return the finished buffer.
flatbuffers::DetachedBuffer make_column_ref(flatbuffers::FlatBufferBuilder& b,
                                            uint32_t index) {
  auto cr = fb::CreateColumnRef(b, index, b.CreateString("c"));
  auto e = fb::CreateExpr(b, fb::ExprNode_ColumnRef, cr.Union());
  b.Finish(e);
  return b.Release();
}

// Build `left <op> right` over two column refs at indices 0 and 1.
flatbuffers::DetachedBuffer make_binary(flatbuffers::FlatBufferBuilder& b,
                                        fb::BinaryOp op) {
  auto l_cr = fb::CreateColumnRef(b, 0, b.CreateString("l"));
  auto l = fb::CreateExpr(b, fb::ExprNode_ColumnRef, l_cr.Union());
  auto r_cr = fb::CreateColumnRef(b, 1, b.CreateString("r"));
  auto r = fb::CreateExpr(b, fb::ExprNode_ColumnRef, r_cr.Union());
  auto bin = fb::CreateBinaryExprNode(b, l, op, r);
  auto e = fb::CreateExpr(b, fb::ExprNode_BinaryExprNode, bin.Union());
  b.Finish(e);
  return b.Release();
}

// A 0-row column_view carrying only a type (enough for type inference).
cudf::column_view typed_col(cudf::data_type dt) {
  return cudf::column_view{dt, 0, nullptr, nullptr, 0};
}

}  // namespace

// is_ast_able: same-type non-decimal operands fuse via AST; any decimal operand
// or a type mismatch routes to the column path.
TEST(AstRouting, IsAstAble) {
  // int32 < int32  →  AST-able.
  {
    flatbuffers::FlatBufferBuilder b;
    auto buf = make_binary(b, fb::BinaryOp_Lt);
    auto* expr = flatbuffers::GetRoot<fb::Expr>(buf.data());
    std::vector<cudf::column_view> cols{typed_col(cudf::data_type{cudf::type_id::INT32}),
                                        typed_col(cudf::data_type{cudf::type_id::INT32})};
    EXPECT_TRUE(peacock::is_ast_able(expr, cudf::table_view{cols}));
  }
  // int32 < int64  →  type mismatch, column path.
  {
    flatbuffers::FlatBufferBuilder b;
    auto buf = make_binary(b, fb::BinaryOp_Lt);
    auto* expr = flatbuffers::GetRoot<fb::Expr>(buf.data());
    std::vector<cudf::column_view> cols{typed_col(cudf::data_type{cudf::type_id::INT32}),
                                        typed_col(cudf::data_type{cudf::type_id::INT64})};
    EXPECT_FALSE(peacock::is_ast_able(expr, cudf::table_view{cols}));
  }
  // decimal * decimal  →  decimal operand, column path.
  {
    flatbuffers::FlatBufferBuilder b;
    auto buf = make_binary(b, fb::BinaryOp_Multiply);
    auto* expr = flatbuffers::GetRoot<fb::Expr>(buf.data());
    std::vector<cudf::column_view> cols{typed_col(dec(-2)), typed_col(dec(-2))};
    EXPECT_FALSE(peacock::is_ast_able(expr, cudf::table_view{cols}));
  }
  // Bare column ref is trivially AST-able.
  {
    flatbuffers::FlatBufferBuilder b;
    auto buf = make_column_ref(b, 0);
    auto* expr = flatbuffers::GetRoot<fb::Expr>(buf.data());
    std::vector<cudf::column_view> cols{typed_col(cudf::data_type{cudf::type_id::INT32})};
    EXPECT_TRUE(peacock::is_ast_able(expr, cudf::table_view{cols}));
  }
}

TEST(PeacockGpu, ExecutorCreateDestroy) {
  peacock_executor_t* ex = nullptr;
  ASSERT_EQ(peacock_executor_create(/*gpu_memory_limit=*/0, &ex), 0);
  ASSERT_NE(ex, nullptr);
  peacock_executor_destroy(ex);
}

TEST(PeacockGpu, ExecutorNullOut) {
  EXPECT_NE(peacock_executor_create(0, nullptr), 0);
}

// The timing switch, on the tier that has no GPU because it needs none: it is a
// process-global mode, and the whole safety argument for the benchmark work is that the
// correctness suite never leaves it on. Stuck on, it would leak a CUDA event pair per
// region until the session ends, and nothing would fail.
TEST(NodeTiming, DefaultsOff) {
  EXPECT_EQ(peacock::node_timing(), peacock::NodeTiming::Off);
  EXPECT_FALSE(peacock::node_timing_enabled());
}

TEST(NodeTiming, SwitchRoundTrips) {
  peacock::set_node_timing(peacock::NodeTiming::Events);
  EXPECT_EQ(peacock::node_timing(), peacock::NodeTiming::Events);
  EXPECT_TRUE(peacock::node_timing_enabled());
  peacock::set_node_timing(peacock::NodeTiming::Off);
  EXPECT_EQ(peacock::node_timing(), peacock::NodeTiming::Off);
  EXPECT_FALSE(peacock::node_timing_enabled());
}

// The row range every per-batch caller's bounds go through, on the tier that needs no
// device: it is arithmetic, and reaching it through a scan on the GPU host is the long
// way round to a case that cannot fail there for a reason worth knowing.
TEST(ClampRowRange, ToTheEndSentinel) {
  EXPECT_EQ(peacock::clamp_row_range(0, UINT64_MAX, 100), std::make_pair(0, 100));
  EXPECT_EQ(peacock::clamp_row_range(40, UINT64_MAX, 100), std::make_pair(40, 100));
}

TEST(ClampRowRange, PastTheEndClamps) {
  EXPECT_EQ(peacock::clamp_row_range(90, 1000, 100), std::make_pair(90, 100));
  EXPECT_EQ(peacock::clamp_row_range(0, 100, 100), std::make_pair(0, 100));
}

TEST(ClampRowRange, AtOrPastTheEndIsEmpty) {
  EXPECT_EQ(peacock::clamp_row_range(100, 10, 100), std::make_pair(100, 100));
  EXPECT_EQ(peacock::clamp_row_range(500, UINT64_MAX, 100), std::make_pair(100, 100));
  EXPECT_EQ(peacock::clamp_row_range(0, 0, 100), std::make_pair(0, 0));
  EXPECT_EQ(peacock::clamp_row_range(0, UINT64_MAX, 0), std::make_pair(0, 0));
}

TEST(ClampRowRange, TheSentinelDoesNotOverflow) {
  // The reason the length is taken against the rows REMAINING rather than added to the
  // offset: offset + UINT64_MAX wraps, and the wrapped end lands inside the table.
  EXPECT_EQ(peacock::clamp_row_range(1, UINT64_MAX, 100), std::make_pair(1, 100));
  EXPECT_EQ(peacock::clamp_row_range(UINT64_MAX, UINT64_MAX, 100), std::make_pair(100, 100));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
