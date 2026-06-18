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
        # left scan: bytes_read (storage) = 100*(240/10)=2400; output (materialization)
        # = min(10,100)*(240/10)=240. Read tracked separately from materialization now.
        scan = self.root.children[0].children[0].children[0]
        sc = dc.scan_cost(scan)
        self.assertEqual((sc["bytes_read"], sc["rows_read"], sc["out_bytes"]), (2400, 100, 240))
        self.assertEqual(dc.node_materialized(scan, self.nowarn), 240)   # scan materialization = output
        self.assertEqual(dc.scan_bytes_read(scan), 2400)                 # storage component, separate

    def test_streaming_zero(self):
        flt = self.root.children[0].children[0].children[1]
        self.assertEqual(dc.node_materialized(flt, self.nowarn), 0)

    def test_total_and_tree(self):
        lines, materialization, storage_read = dc.build_cost_tree(self.root, self.nowarn)
        # combined is unchanged from the old single total (2_487_972): the scan read
        # term just moved from `materialized` into the separate storage_read component.
        self.assertEqual(materialization + storage_read, 2_487_972)
        self.assertGreater(storage_read, 0)            # scans contribute storage read
        self.assertTrue(lines[0].startswith("TOP_N:"))  # root at indent 0
        self.assertTrue(lines[1].startswith("  HASH_GROUP_BY:"))  # child indented
        # scan line: materialization = output (240); storage read shown separately (2400).
        scan_line = next(l for l in lines if "table=lineitem" in l)
        self.assertIn("materialized=240", scan_line)
        self.assertIn("bytes_read=2400", scan_line)
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
            dc.build_cost_tree(dc.build_tree(norm), lambda *a: None)[1:],
            dc.build_cost_tree(dc.build_tree(raw), lambda *a: None)[1:],
        )


class TestPruning(unittest.TestCase):
    import datetime as _dt

    def test_table_base(self):
        self.assertEqual(dc.table_base("tpch.main.lineitem"), "lineitem")
        self.assertEqual(dc.table_base("lineitem"), "lineitem")

    def test_parse_literal(self):
        d = self._dt.date
        self.assertEqual(dc._parse_literal("'1994-01-01'::DATE"), d(1994, 1, 1))
        # 1.5.4 timestamp form -> date part
        self.assertEqual(dc._parse_literal("'1995-01-01 00:00:00'::TIMESTAMP"), d(1995, 1, 1))
        self.assertEqual(dc._parse_literal("2451545"), 2451545)
        self.assertIsNone(dc._parse_literal("0.05"))      # decimal -> not prunable
        self.assertIsNone(dc._parse_literal("'STORE'"))   # string -> not prunable

    def test_parse_range_filters_date_and_cast(self):
        d = self._dt.date
        f = ("l_shipdate>='1994-01-01'::DATE AND "
             "(CAST(l_shipdate AS TIMESTAMP) < '1995-01-01 00:00:00'::TIMESTAMP) AND "
             "l_discount>=0.05 AND l_quantity<24.00")
        r = dc.parse_range_filters(f)
        self.assertEqual(r, {"l_shipdate": {"lo": d(1994, 1, 1), "hi": d(1995, 1, 1)}})  # decimals skipped

    def test_parse_range_filters_int_eq(self):
        self.assertEqual(dc.parse_range_filters("ss_sold_date_sk=2451545"),
                         {"ss_sold_date_sk": {"lo": 2451545, "hi": 2451545}})

    def test_filter_columns_includes_cast(self):
        cols = dc.filter_columns("l_shipdate>='x' AND (CAST(l_shipdate AS TIMESTAMP) < 'y') AND l_discount>=0.05")
        self.assertEqual(cols, {"l_shipdate", "l_discount"})

    def test_rowgroup_survives(self):
        d = self._dt.date
        ranges = {"c": {"lo": d(1994, 1, 1), "hi": d(1995, 1, 1)}}
        self.assertFalse(dc.rowgroup_survives({"c": {"min": d(1992, 1, 1), "max": d(1992, 6, 1)}}, ranges))
        self.assertTrue(dc.rowgroup_survives({"c": {"min": d(1994, 3, 1), "max": d(1994, 6, 1)}}, ranges))
        self.assertFalse(dc.rowgroup_survives({"c": {"min": d(1999, 1, 1), "max": d(1999, 6, 1)}}, ranges))
        # boundary touch survives
        self.assertTrue(dc.rowgroup_survives({"c": {"min": d(1993, 1, 1), "max": d(1994, 1, 1)}}, ranges))
        # missing stats (e.g. decimal) -> cannot prune -> survive
        self.assertTrue(dc.rowgroup_survives({"c": {"min": None, "max": None}}, ranges))

    def test_compute_pruning_from_rowgroups(self):
        d = self._dt.date
        ranges = {"sd": {"lo": d(1994, 1, 1), "hi": d(1995, 1, 1)}}
        rg = lambda lo, hi: {"num_rows": 100, "cols": {
            "sd": {"min": lo, "max": hi, "compressed": 800},
            "v": {"min": None, "max": None, "compressed": 200}}}
        rows = [rg(d(1992, 1, 1), d(1992, 6, 1)), rg(d(1994, 3, 1), d(1994, 6, 1)), rg(d(1999, 1, 1), d(1999, 6, 1))]
        res = dc.compute_pruning_from_rowgroups(rows, {"sd", "v"}, ranges)
        self.assertEqual(res["row_groups_kept"], 1)
        self.assertEqual(res["row_groups_total"], 3)
        self.assertEqual((res["rows_fetched"], res["rows_total"]), (100, 300))
        self.assertEqual((res["bytes_fetched"], res["bytes_total"]), (1000, 3000))

    def test_compute_pruning_no_prunable_filter_keeps_all(self):
        d = self._dt.date
        rg = {"num_rows": 50, "cols": {"sd": {"min": d(1994, 1, 1), "max": d(1994, 2, 1), "compressed": 10}}}
        res = dc.compute_pruning_from_rowgroups([rg, rg], {"sd"}, {})  # no ranges
        self.assertEqual(res["row_groups_kept"], 2)


class TestDynamicFilters(unittest.TestCase):
    def test_dynamic_filter_str_cleans(self):
        # optional: markers dropped; bloom IN BF(...) / IN (...) dropped; ranges kept
        v = ("optional: sr_returned_date_sk>=2451545 AND optional: sr_returned_date_sk<=2451910 "
             "AND optional: c_customer_sk IN BF(ctr_customer_sk)")
        self.assertEqual(dc.dynamic_filter_str(v),
                         "sr_returned_date_sk>=2451545 AND sr_returned_date_sk<=2451910")

    def test_dynamic_filter_str_list(self):
        v = ["optional: a>=1 AND optional: a<=9", "optional: b IN (1,2,3)"]
        self.assertEqual(dc.dynamic_filter_str(v), "a>=1 AND a<=9")

    def test_dynamic_filter_str_empty(self):
        self.assertEqual(dc.dynamic_filter_str(None), "")
        self.assertEqual(dc.dynamic_filter_str(""), "")

    def test_extract_dynamic_filters_preorder(self):
        # query wrapper -> agg -> [scan(with dyn), scan(no dyn)] : pre-order list
        raw = {"children": [{"operator_type": "HASH_JOIN", "children": [
            {"operator_type": "TABLE_SCAN",
             "extra_info": {"Table": "f", "Dynamic Filters": "optional: k>=2 AND optional: k<=8"}},
            {"operator_type": "TABLE_SCAN", "extra_info": {"Table": "d"}}]}]}
        self.assertEqual(dc.extract_dynamic_filters(raw), ["k>=2 AND k<=8", ""])

    def test_merge_ranges_intersects(self):
        a = {"k": {"lo": 1, "hi": 10}, "x": {"lo": 5, "hi": None}}
        b = {"k": {"lo": 3, "hi": 8}, "y": {"lo": None, "hi": 4}}
        m = dc.merge_ranges(a, b)
        self.assertEqual(m["k"], {"lo": 3, "hi": 8})   # intersection
        self.assertEqual(m["x"], {"lo": 5, "hi": None})
        self.assertEqual(m["y"], {"lo": None, "hi": 4})


class TestGuards(unittest.TestCase):
    def test_scan_count_mismatch_fails_loud(self):
        # pass2 bounds count != pass1 scan count -> SystemExit (plan-drift guard).
        root = dc.Node("HASH_JOIN", 0, 0, 0, {})
        root.children = [dc.Node("TABLE_SCAN", 10, 1, 1, {"Table": "t"})]
        with self.assertRaises(SystemExit):
            dc.compute_scan_pruning(root, ["a>=1", "b<=2"], "/no/such/dir", lambda *a: None)

    def test_crosscheck_date_bound_match(self):
        warned = []
        # bound matches a reconstructed date_dim range -> no warning
        dc.crosscheck_date_dynfilters({"ss_sold_date_sk": {"lo": 100, "hi": 200}},
                                      [(100, 200)], "store_sales", lambda m: warned.append(m))
        self.assertEqual(warned, [])

    def test_crosscheck_date_bound_mismatch_warns(self):
        warned = []
        dc.crosscheck_date_dynfilters({"ss_sold_date_sk": {"lo": 100, "hi": 999}},
                                      [(100, 200)], "store_sales", lambda m: warned.append(m))
        self.assertEqual(len(warned), 1)

    def test_crosscheck_skips_non_date_and_no_dim(self):
        warned = []
        # non-date_sk col ignored; and no dim ranges -> no cross-check at all
        dc.crosscheck_date_dynfilters({"ss_item_sk": {"lo": 1, "hi": 9}}, [(100, 200)],
                                      "store_sales", lambda m: warned.append(m))
        dc.crosscheck_date_dynfilters({"ss_sold_date_sk": {"lo": 1, "hi": 9}}, [],
                                      "store_sales", lambda m: warned.append(m))
        self.assertEqual(warned, [])


class TestSections(unittest.TestCase):
    def test_pruning_section_empty_without_pruning(self):
        # No compute_scan_pruning run (no parquet) -> no pruning attached -> omitted.
        scan = dc.Node("TABLE_SCAN", 100, 10, 0, {"Table": "t", "Filters": "sd>=1"})
        self.assertEqual(dc.format_pruning_section(scan), [])

    def test_scan_cost_uses_rows_fetched(self):
        # With pruning folded in, the scan read term uses post-prune rows_fetched,
        # and the output term is capped at the read (can't output > read).
        scan = dc.Node("TABLE_SCAN", 320, 10, 1000, {"Table": "t"})  # post-static rows=1000
        self.assertEqual(dc.scan_cost(scan)["rows_read"], 1000)      # default: post-static
        scan.rows_fetched = 5                                        # post-prune (< output_rows=10)
        sc = dc.scan_cost(scan)
        self.assertEqual(sc["rows_read"], 5)                         # uses rows_fetched
        self.assertEqual(sc["bytes_read"], 5 * 32)                  # storage read = 5 * (320/10)
        self.assertEqual(sc["out_rows"], 5)                         # output CAPPED at read (was 10)
        self.assertEqual(sc["out_bytes"], 5 * 32)                  # capped output (materialization)

    def test_scan_cost_uses_decoded_bytes(self):
        # When pruning carries bytes_fetched_decoded (Arrow in-memory size), the read
        # term uses THAT (peacockdb units), not the compressed bytes_fetched.
        scan = dc.Node("TABLE_SCAN", 320, 10, 1000, {"Table": "t"})
        scan.rows_fetched = 5
        scan.pruning = {"rows_fetched": 5, "bytes_fetched": 999,        # compressed (ignored by cost)
                        "bytes_fetched_decoded": 4096}                  # decoded Arrow (used)
        sc = dc.scan_cost(scan)
        self.assertEqual(sc["bytes_read"], 4096)                       # decoded, not 999, not derived
        # absent decoded -> falls back to derived rows_read x per_row
        scan.pruning = {"rows_fetched": 5, "bytes_fetched": 999}
        self.assertEqual(dc.scan_cost(scan)["bytes_read"], 5 * 32)

    def test_scan_cost_byte_clamp_read_ge_output(self):
        # Two-basis seam: decoded Arrow read (300) < DuckDB-basis out_bytes (10*100=1000).
        # The byte-level max-clamp must restore read >= output (bytes_read == out_bytes).
        scan = dc.Node("TABLE_SCAN", 1000, 10, 10, {"Table": "dim"})  # per_row=100
        scan.rows_fetched = 10
        scan.pruning = {"rows_fetched": 10, "bytes_fetched": 50, "bytes_fetched_decoded": 300}
        sc = dc.scan_cost(scan)
        self.assertEqual(sc["out_bytes"], 1000)
        self.assertEqual(sc["bytes_read"], 1000)        # clamped up from 300 to out_bytes
        self.assertGreaterEqual(sc["bytes_read"], sc["out_bytes"])  # invariant holds

    def test_breakdown_section(self):
        # scan (materialized via two-part) + an aggregate breaker over it.
        scan = dc.Node("TABLE_SCAN", 1826560, 114160, 4403775, {"Table": "lineitem"})
        agg = dc.Node("UNGROUPED_AGGREGATE", 16, 1, 0, {})
        agg.children = [scan]
        out = "\n".join(dc.format_breakdown_section(agg))
        self.assertIn("--- cost breakdown ---", out)
        self.assertIn("TABLE_SCAN(lineitem)", out)
        self.assertTrue(out.strip().endswith("%") or "total=" in out)
        self.assertIn("total=", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
