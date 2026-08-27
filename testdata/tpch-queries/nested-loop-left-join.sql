-- The one join cell no corpus query reaches: a nested-loop Left. Its probe side is a
-- single batch — the finish trick accumulates probe keys and a predicate join has none —
-- so it is the only mode whose probe does not stream, and the only one whose unmatched
-- build rows come out of the same call as the matches.
--
-- Returns 51 rows: the four regions above 0 matched by 5, 10, 15 and 20 nations, plus
-- region 0, which is matched by none and so arrives padded with NULLs.
SELECT *
FROM region r LEFT JOIN nation n ON r.r_regionkey > n.n_regionkey;
