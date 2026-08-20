-- Both row-interval lowerings on one root-to-leaf path, and the only OFFSETs in either
-- corpus: the root-adjacent interval, which becomes GpuUnload's skip/fetch, and the
-- mid-plan GpuLimit over the scan. What connects them has to be a cross join. With an
-- aggregate there instead DataFusion drops the inner interval at tp4, and a limit inside
-- a UNION ALL branch is dropped at every tp -- both are #166, which is where that lives.
--
-- Returns 20 rows of one column: 40 part keys after the first 5, crossed with the 5
-- regions, then rows 4 to 23 of that 200-row product. The plan never builds those 200:
-- the outer interval narrows the inner fetch to 23, and the scan reads 28 rows in all.
SELECT k
FROM (SELECT p_partkey AS k FROM part LIMIT 40 OFFSET 5) x, region
LIMIT 20 OFFSET 3;
