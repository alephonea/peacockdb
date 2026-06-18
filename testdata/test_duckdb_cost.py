#!/usr/bin/env python3
"""Unit tests for duckdb_cost.py (stdlib unittest, no external deps).

Run: python3 testdata/test_duckdb_cost.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb_cost as dc  # noqa: E402

# Small profiling-tree fixture (wrapper -> TOP_N -> HASH_GROUP_BY -> HASH_JOIN
# -> [TABLE_SCAN(left/probe), FILTER -> TABLE_SCAN(right/build)]).
DOC = {
    "result_set_size": 320,  # query wrapper: no operator_type, must be dropped
    "children": [{
        "operator_type": "TOP_N", "result_set_size": 320, "operator_cardinality": 10,
        "children": [{
            "operator_type": "HASH_GROUP_BY", "result_set_size": 290500, "operator_cardinality": 11620,
            "children": [{
                "operator_type": "HASH_JOIN", "result_set_size": 976608, "operator_cardinality": 30519,
                "children": [
                    {"operator_type": "TABLE_SCAN", "result_set_size": 240, "operator_cardinality": 10,
                     "operator_rows_scanned": 100,
                     "extra_info": {"Table": "lineitem", "Projections": "l_orderkey", "Filters": "x>1"},
                     "children": []},
                    {"operator_type": "FILTER", "result_set_size": 241136, "operator_cardinality": 30142,
                     "children": [
                         {"operator_type": "TABLE_SCAN", "result_set_size": 80, "operator_cardinality": 10,
                          "operator_rows_scanned": 50, "extra_info": {"Projections": ["a", "b"]},
                          "children": []}]},
                ]}]}]}]}


class TestNormalizeProjections(unittest.TestCase):
    def test_all_forms(self):
        # DuckDB's inconsistent typing: str (1 col), list (2+), '' / missing.
        self.assertEqual(dc.normalize_projections("d_date_sk"), ["d_date_sk"])  # was dropped before
        self.assertEqual(dc.normalize_projections(["a", "b"]), ["a", "b"])
        self.assertIsNone(dc.normalize_projections(""))
        self.assertIsNone(dc.normalize_projections(None))
        self.assertIsNone(dc.normalize_projections([]))


class TestClassifierAndCost(unittest.TestCase):
    def setUp(self):
        self.root = dc.build_tree(DOC)
        self.nowarn = lambda *a: None

    def test_wrapper_dropped_root_is_operator(self):
        self.assertEqual(self.root.op, "TOP_N")  # query wrapper not the root

    def test_build_input_counts_child_input(self):
        # HASH_GROUP_BY materializes its input = child HASH_JOIN output_bytes.
        gb = self.root.children[0]
        self.assertEqual(dc.node_materialized(gb, self.nowarn), 976608)

    def test_join_counts_left_right_own_no_double_count(self):
        # HASH_JOIN = own output (976608) + non-self-counting children's output.
        # Left child is a TABLE_SCAN (self-counts its output via the two-part cost)
        # -> skipped; right child FILTER (streaming, doesn't self-count) -> +241136.
        hj = self.root.children[0].children[0]
        self.assertEqual(dc.node_materialized(hj, self.nowarn), 976608 + 241136)

    def test_scan_two_part_cost(self):
        # left scan: bytes_read = 100 * (240/10) = 2400; materialized = 2400 + 240.
        scan = self.root.children[0].children[0].children[0]
        bytes_read, rows_read, materialized = dc.scan_cost(scan)
        self.assertEqual((bytes_read, rows_read, materialized), (2400, 100, 2640))

    def test_streaming_zero(self):
        flt = self.root.children[0].children[0].children[1]
        self.assertEqual(dc.node_materialized(flt, self.nowarn), 0)

    def test_total_and_tree(self):
        lines, total = dc.build_cost_tree(self.root, self.nowarn)
        # 290500 (TOP_N->GB) + 976608 (GB->HJ output) + 1217744 (HJ own+FILTER,
        # scan child skipped) + 2640 (scan1) + 0 (FILTER) + 480 (scan2)
        self.assertEqual(total, 2_487_972)
        self.assertTrue(lines[0].startswith("TOP_N:"))  # root at indent 0
        self.assertTrue(lines[1].startswith("  HASH_GROUP_BY:"))  # child indented
        # scan line carries the transparency fields incl. normalized 1-col projection
        scan_line = next(l for l in lines if "table=lineitem" in l)
        self.assertIn("materialized=2640", scan_line)
        self.assertIn("bytes_read_est=2400", scan_line)
        self.assertIn("rows_read=100", scan_line)
        self.assertIn("projections=[l_orderkey]", scan_line)
        self.assertIn('filters="x>1"', scan_line)

    def test_binary_join_asserts_two_children(self):
        n = dc.Node("HASH_JOIN", 0, 0, 0, {})
        n.children = [dc.Node("X", 5, 0, 0, {})]  # only 1 child -> loud failure
        with self.assertRaises(AssertionError):
            dc.node_materialized(n, lambda *a: None)

    def test_delim_join_excludes_rereads_and_self_counters(self):
        # DELIM joins (>2 children, no binary assert): own output + each child that
        # doesn't self-count. A DELIM_SCAN re-reads an already-materialized buffer
        # (excluded), a child HASH_JOIN self-counts (excluded), a TABLE_SCAN
        # self-counts via two-part (excluded). Only the PROJECTION is added.
        n = dc.Node("RIGHT_DELIM_JOIN", output_bytes=5, output_rows=0, rows_scanned=0, extra={})
        n.children = [
            dc.Node("PROJECTION", 100, 0, 0, {}),                 # added
            dc.Node("DELIM_SCAN", 200, 0, 0, {}),                 # excluded (re-read)
            dc.Node("HASH_JOIN", 300, 0, 0, {}),                  # excluded (self-counts)
            dc.Node("TABLE_SCAN", 400, 10, 50, {}),              # excluded (self-counts)
        ]
        self.assertEqual(dc.node_materialized(n, lambda *a: None), 5 + 100)

    def test_build_input_excludes_reread(self):
        # A group-by over a re-read (CTE_SCAN) must not add the re-read's output.
        n = dc.Node("HASH_GROUP_BY", 0, 0, 0, {})
        n.children = [dc.Node("CTE_SCAN", 999, 0, 0, {}), dc.Node("PROJECTION", 50, 0, 0, {})]
        self.assertEqual(dc.node_materialized(n, lambda *a: None), 50)

    def test_unrecognized_op_warns(self):
        warned = []
        node = dc.Node("FUTURE_OP", output_bytes=99999, output_rows=1, rows_scanned=0, extra={})
        self.assertEqual(dc.node_materialized(node, lambda op, ob: warned.append(op)), 0)
        self.assertEqual(warned, ["FUTURE_OP"])


class TestAnnotations(unittest.TestCase):
    def _n(self, op, extra):
        return dc.Node(op, 0, 0, 0, extra)

    def test_join(self):
        n = self._n("HASH_JOIN", {"Join Type": "INNER", "Conditions": "o_orderkey = l_orderkey"})
        self.assertEqual(dc.annotation(n), "join_type=INNER, conditions=[o_orderkey = l_orderkey]")

    def test_groupby(self):
        n = self._n("HASH_GROUP_BY", {"Groups": "#0", "Aggregates": ["count_star()", "sum(#1)"]})
        self.assertEqual(dc.annotation(n), "groups=[#0], aggregates=[count_star(), sum(#1)]")

    def test_top_n(self):
        n = self._n("TOP_N", {"Order By": "l_returnflag ASC", "Top": "10"})
        self.assertEqual(dc.annotation(n), "order_by=[l_returnflag ASC], top=10")

    def test_projection_keeps_duckdb_exprs(self):
        n = self._n("PROJECTION", {"Projections": ["#0", "__internal_compress_string(#1)"]})
        self.assertEqual(dc.annotation(n), "projections=[#0, __internal_compress_string(#1)]")

    def test_scan(self):
        n = self._n("TABLE_SCAN", {"Table": "lineitem", "Projections": "l_orderkey",
                                    "Filters": ["a>1", "b<2"]})
        self.assertEqual(dc.annotation(n), 'table=lineitem, projections=[l_orderkey], filters="a>1 AND b<2"')

    def test_streaming_node_no_annotation(self):
        self.assertEqual(dc.annotation(self._n("FILTER", {})), "")


class TestNormalize(unittest.TestCase):
    def test_keeps_only_needed_fields(self):
        raw = {"latency": 1.2, "children": [{
            "operator_type": "TABLE_SCAN", "result_set_size": 10, "operator_cardinality": 5,
            "operator_rows_scanned": 100, "cpu_time": 9,
            "extra_info": {"Table": "t", "Estimated Cardinality": 99, "Projections": "c"},
            "children": []}]}
        out = dc.normalize(raw)
        self.assertNotIn("latency", out)  # wrapper noise dropped
        scan = out["children"][0]
        self.assertEqual(scan["operator_type"], "TABLE_SCAN")
        self.assertEqual(scan["operator_rows_scanned"], 100)
        self.assertNotIn("cpu_time", scan)  # measured-but-unused dropped
        # KEEP_EXTRA only — "Estimated Cardinality" dropped.
        self.assertEqual(scan["extra_info"], {"Table": "t", "Projections": "c"})

    def test_roundtrip_through_extract(self):
        # normalize then build_tree must yield the same materialized as raw.
        raw = DOC
        norm = dc.normalize(raw)
        self.assertEqual(
            dc.build_cost_tree(dc.build_tree(norm), lambda *a: None)[1],
            dc.build_cost_tree(dc.build_tree(raw), lambda *a: None)[1],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
