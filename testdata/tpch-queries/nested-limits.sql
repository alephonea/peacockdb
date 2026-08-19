-- Several row intervals on one root-to-leaf path, and the only OFFSETs in either corpus:
-- the skip half of both lowerings -- the GpuUnload that absorbs a root-adjacent interval
-- and the mid-plan GpuLimit -- is canonized nowhere else.
--
-- An aggregate is what makes a limit survive: DataFusion merges adjacent limits and pushes
-- the rest into a scan or a sort, so only a limit something blocks from moving stays a node
-- of its own. Hence the stack -- group, limit, join, group, limit.
--
-- The union's own branch limits do NOT survive, and that is the shape rather than an
-- oversight: any limit above a union is pushed into its branches and merged with theirs, so
-- a plan holds either branch limits or a root one. This query takes the root one, since
-- nothing else canonizes a GpuUnload carrying a skip.
SELECT k, n
FROM (
  SELECT k, count(*) AS n
  FROM (
    SELECT k
    FROM (
      (SELECT p_brand AS k FROM part GROUP BY p_brand LIMIT 10 OFFSET 2)
      UNION ALL
      (SELECT n_name AS k FROM nation GROUP BY n_name LIMIT 8 OFFSET 3)
    ) u,
    region
    LIMIT 40 OFFSET 5
  ) mid
  GROUP BY k
  LIMIT 6 OFFSET 2
) o,
nation
LIMIT 20 OFFSET 3;
