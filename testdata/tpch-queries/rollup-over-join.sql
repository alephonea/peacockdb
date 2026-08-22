-- A grouping set over an input that declares a hash, which nothing else in either corpus
-- has: DataFusion puts a partial aggregate straight over its scan, so the rollup's rows are
-- never already partitioned when it runs. The join is what forces the shuffle here.
--
-- The obvious form -- a rollup over an inner GROUP BY -- does not work: projecting two
-- columns of customer reads under small_table_bytes, so that scan plans one lane and there
-- is no shuffle to carry a hash at all. A shape-hunting query has to clear the threshold or
-- get its distribution from a join.
SELECT c_nationkey, c_mktsegment, count(*)
FROM customer JOIN nation ON c_nationkey = n_nationkey
GROUP BY ROLLUP(c_nationkey, c_mktsegment);
