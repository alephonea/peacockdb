"""The cuDF calls the C++ join operators make, modelled with pandas.

`cpp/src/operators/join.cpp` is written against a small set of cuDF primitives, and this
module is that set — same names, same signatures, same return shapes. The joins here
return **index vectors**, as cuDF's do, and the caller gathers; the gather policy is a
parameter because choosing it wrong is what turns an unmatched row into a fault
(architecture.md, "cuDF options"). Nothing here knows what a plan is.

Keeping the shape means a claim made here can be read back against join.cpp line by line.
A model that answered `left_join` with a finished table instead of two gather maps would
prove nothing about the code that has to build one.

**Tables are columns and names**, never a pandas frame: a join's output is
`[left_cols..., right_cols...]` and both sides may carry a column called `k`, which a
frame cannot hold without renaming one. `TableResult` in C++ has exactly this shape and
exactly this reason.

**Null equality is a parameter on every join**, never a default: cuDF's default matches
NULL keys and SQL's does not (frame.py, subset rule 4). The two meanings are implemented
here rather than approximated — UNEQUAL holds null-keyed rows out of the hash table, which
is what makes them unmatched rather than absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


#: cuDF's `JoinNoneValue`: the index a join writes for a row with no counterpart.
#: Gathering it under DONT_CHECK is the `cudaErrorIllegalAddress` in architecture.md.
JOIN_NONE = np.iinfo(np.int32).min


class NullEquality(Enum):
    """`cudf::null_equality`. EQUAL: NULL == NULL. UNEQUAL: a NULL key matches nothing."""

    EQUAL = "EQUAL"
    UNEQUAL = "UNEQUAL"


class OutOfBoundsPolicy(Enum):
    """`cudf::out_of_bounds_policy`. NULLIFY turns a sentinel index into a null."""

    NULLIFY = "NULLIFY"
    DONT_CHECK = "DONT_CHECK"


@dataclass
class Table:
    """`cudf::table` plus the parallel name vector — peacock's `TableResult`.

    The two are separate on purpose: #135 is about there being no invariant tying them
    together in C++, so a model that fused them would hide the thing being modelled.
    """

    columns: list[pd.Series]
    names: list[str]

    def __post_init__(self):
        if len(self.columns) != len(self.names):
            raise ValueError(
                f"table has {len(self.columns)} columns against {len(self.names)} names"
            )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Table":
        return cls(
            [frame.iloc[:, i].reset_index(drop=True) for i in range(frame.shape[1])],
            list(frame.columns),
        )

    def to_frame(self) -> pd.DataFrame:
        """Back to a frame. Duplicate names are disambiguated, since a plan that keeps
        both sides' key column is legal here and unrepresentable there — the projection
        that normally drops one is the join node's own `projection` field."""
        frame = pd.DataFrame(dict(enumerate(self.columns)))
        frame.columns = _unique_names(self.names)
        return frame

    def num_rows(self) -> int:
        return len(self.columns[0]) if self.columns else 0

    def num_columns(self) -> int:
        return len(self.columns)

    def column(self, i: int) -> pd.Series:
        """`table_view::column(i)` — `_columns.at(i)`, so out of range throws."""
        if not 0 <= i < len(self.columns):
            raise IndexError(f"column {i} of a {len(self.columns)}-column table")
        return self.columns[i]

    def select(self, indices) -> "Table":
        """A column subset by ordinal — what a `projection` field means everywhere."""
        return Table([self.column(i) for i in indices], [self.names[i] for i in indices])

    def byte_size(self) -> int:
        return int(sum(c.memory_usage(index=False, deep=True) for c in self.columns))


def _unique_names(names: list[str]) -> list[str]:
    seen, out = {}, []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def concatenate(tables: list[Table]) -> Table:
    """`cudf::concatenate` — identical column count and names, as the real one requires
    identical types (frame.py, subset rule 5)."""
    if not tables:
        raise ValueError("concatenate of nothing")
    names = tables[0].names
    for other in tables[1:]:
        if other.names != names:
            raise ValueError(f"concatenate name mismatch: {names} vs {other.names}")
    return Table(
        [
            pd.concat([t.columns[i] for t in tables], ignore_index=True)
            for i in range(len(names))
        ],
        list(names),
    )


def gather(table: Table, indices: np.ndarray, policy=OutOfBoundsPolicy.DONT_CHECK) -> Table:
    """`cudf::gather`. Under NULLIFY an out-of-range index becomes a null row.

    DONT_CHECK raises here where cuDF would read out of bounds — the failure mode is
    undefined behaviour there, so the model makes the wrong policy loud rather than
    plausible.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n = table.num_rows()
    out_of_range = (indices < 0) | (indices >= n)
    if policy is OutOfBoundsPolicy.DONT_CHECK:
        if out_of_range.any():
            raise IndexError(
                "gather under DONT_CHECK saw an out-of-range index — cuDF would fault; "
                "the caller wanted NULLIFY"
            )
        safe = indices
    else:
        safe = np.where(out_of_range, 0, indices)
    columns = []
    for column in table.columns:
        taken = column.to_numpy()[safe] if n else np.empty(len(safe), dtype=object)
        series = pd.Series(taken)
        if policy is OutOfBoundsPolicy.NULLIFY and out_of_range.any():
            # A nulled row makes the column nullable, which pandas expresses by widening
            # an integer column to float — a known divergence (frame.py) and not one this
            # model can remove.
            series = series.astype("object").where(~pd.Series(out_of_range), other=None)
            series = _renarrow(series, column)
        columns.append(series.reset_index(drop=True))
    return Table(columns, list(table.names))


def _renarrow(series: pd.Series, like: pd.Series) -> pd.Series:
    """Keep a nulled numeric column numeric — object columns compare badly downstream."""
    if pd.api.types.is_numeric_dtype(like) and not pd.api.types.is_bool_dtype(like):
        return pd.to_numeric(series, errors="coerce")
    return series


def scatter(source: Table, indices: np.ndarray, target: Table) -> Table:
    """`cudf::scatter` — target rows at `indices` take the source's rows, in order.

    The mark join's only use: `true` scattered into an all-false column.
    """
    columns = []
    for src, dst in zip(source.columns, target.columns):
        values = dst.to_numpy().copy()
        values[np.asarray(indices, dtype=np.int64)] = src.to_numpy()
        columns.append(pd.Series(values))
    return Table(columns, list(target.names))


def apply_boolean_mask(table: Table, mask: pd.Series) -> Table:
    """`cudf::apply_boolean_mask` — a null in the mask drops the row, as in cuDF."""
    keep = np.asarray(mask.fillna(False).astype(bool))
    return Table([c[keep].reset_index(drop=True) for c in table.columns], list(table.names))


def make_column_from_scalar(value, n_rows: int) -> pd.Series:
    """`cudf::make_column_from_scalar`."""
    return pd.Series([value] * n_rows)


def cross_join(left: Table, right: Table) -> Table:
    """`cudf::cross_join` — every left row against every right row, left-major."""
    l_rows, r_rows = left.num_rows(), right.num_rows()
    l_idx = np.repeat(np.arange(l_rows), r_rows)
    r_idx = np.tile(np.arange(r_rows), l_rows)
    gathered_left = gather(left, l_idx)
    gathered_right = gather(right, r_idx)
    return Table(
        gathered_left.columns + gathered_right.columns,
        list(left.names) + list(right.names),
    )


# -- the equality core the hash joins share ----------------------------------------
#
# cuDF builds a hash table over one side's key rows and probes with the other's. Modelled
# directly rather than through `pandas.merge`, because merge's null handling is its own
# (it matches NaN to NaN) and that is exactly the choice this has to state explicitly.


class _Null:
    """A hashable stand-in for NULL, so a null key can be a dict key under EQUAL."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "NULL"


NULL = _Null()


def _key_rows(keys: Table) -> tuple[list, np.ndarray]:
    """Per row: its key tuple, and whether the row holds any null key."""
    if not keys.columns:
        return [() for _ in range(keys.num_rows())], np.zeros(keys.num_rows(), dtype=bool)
    arrays = [c.to_numpy() for c in keys.columns]
    nulls = [pd.isna(c).to_numpy() for c in keys.columns]
    has_null = np.logical_or.reduce(nulls)
    rows = [
        tuple(NULL if null[i] else array[i] for array, null in zip(arrays, nulls))
        for i in range(keys.num_rows())
    ]
    return rows, has_null


def _match_lists(
    left_keys: Table, right_keys: Table, compare_nulls: NullEquality
) -> list[list[int]]:
    """For each left row, the right rows its key matches, in right-row order."""
    right_rows, right_null = _key_rows(right_keys)
    table: dict[tuple, list[int]] = {}
    for i, key in enumerate(right_rows):
        if compare_nulls is NullEquality.UNEQUAL and right_null[i]:
            continue        # a null key is not in the hash table, so it matches nothing
        table.setdefault(key, []).append(i)
    left_rows, left_null = _key_rows(left_keys)
    matches = []
    for i, key in enumerate(left_rows):
        if compare_nulls is NullEquality.UNEQUAL and left_null[i]:
            matches.append([])
        else:
            matches.append(table.get(key, []))
    return matches


def inner_join(left_keys: Table, right_keys: Table, compare_nulls: NullEquality):
    """`cudf::inner_join` → (left_indices, right_indices), one entry per matched pair."""
    left_idx, right_idx = [], []
    for i, right_rows in enumerate(_match_lists(left_keys, right_keys, compare_nulls)):
        for j in right_rows:
            left_idx.append(i)
            right_idx.append(j)
    return _idx(left_idx), _idx(right_idx)


def left_join(left_keys: Table, right_keys: Table, compare_nulls: NullEquality):
    """`cudf::left_join` — every left row at least once; unmatched carry `JOIN_NONE`.

    Reached by `Right` (which is this call with the sides swapped) and by a single-batch
    probe — and by a streamed Left join **never**, which is one of this prototype's
    results: the batch-partitioned lowering decomposes Left into a per-batch `Inner` plus a
    `LeftAnti` at finish, so the frozen surface executes a Left outer join without ever
    calling `left_join`. The same holds of `full_join` below, via `Right`.
    """
    left_idx, right_idx = [], []
    for i, right_rows in enumerate(_match_lists(left_keys, right_keys, compare_nulls)):
        if right_rows:
            for j in right_rows:
                left_idx.append(i)
                right_idx.append(j)
        else:
            left_idx.append(i)
            right_idx.append(JOIN_NONE)
    return _idx(left_idx), _idx(right_idx)


def full_join(left_keys: Table, right_keys: Table, compare_nulls: NullEquality):
    """`cudf::full_join` — the left join plus the right rows nothing matched."""
    matches = _match_lists(left_keys, right_keys, compare_nulls)
    left_idx, right_idx = left_join(left_keys, right_keys, compare_nulls)
    matched_right = {j for rows in matches for j in rows}
    unmatched = [j for j in range(right_keys.num_rows()) if j not in matched_right]
    return (
        np.concatenate([left_idx, np.full(len(unmatched), JOIN_NONE, dtype=np.int64)]),
        np.concatenate([right_idx, _idx(unmatched)]),
    )


def left_semi_join(left_keys: Table, right_keys: Table, compare_nulls: NullEquality):
    """`cudf::left_semi_join` / `filtered_join::semi_join` → left rows with ≥1 match."""
    matches = _match_lists(left_keys, right_keys, compare_nulls)
    return _idx([i for i, rows in enumerate(matches) if rows])


def left_anti_join(left_keys: Table, right_keys: Table, compare_nulls: NullEquality):
    """`cudf::left_anti_join` / `filtered_join::anti_join` → left rows with no match."""
    matches = _match_lists(left_keys, right_keys, compare_nulls)
    return _idx([i for i, rows in enumerate(matches) if not rows])


def mixed_left_semi_join(
    left_keys: Table,
    right_keys: Table,
    left: Table,
    right: Table,
    predicate,
    compare_nulls: NullEquality,
):
    """`cudf::mixed_left_semi_join` — equality on the keys **and** the AST predicate.

    The pair the predicate sees is (left row, right row), which is why it takes the whole
    tables and not just the keys.
    """
    return _idx(sorted(_mixed_matched(left_keys, right_keys, left, right, predicate,
                                      compare_nulls)))


def mixed_left_anti_join(
    left_keys: Table,
    right_keys: Table,
    left: Table,
    right: Table,
    predicate,
    compare_nulls: NullEquality,
):
    """`cudf::mixed_left_anti_join` — left rows no (key, predicate) pair satisfied."""
    matched = _mixed_matched(left_keys, right_keys, left, right, predicate, compare_nulls)
    return _idx([i for i in range(left.num_rows()) if i not in matched])


def _mixed_matched(left_keys, right_keys, left, right, predicate, compare_nulls) -> set:
    left_idx, right_idx = inner_join(left_keys, right_keys, compare_nulls)
    if not len(left_idx):
        return set()
    keep = predicate.evaluate(left, right, left_idx, right_idx)
    return set(np.asarray(left_idx)[np.asarray(keep, dtype=bool)].tolist())


def conditional_inner_join(left: Table, right: Table, predicate):
    """`cudf::conditional_inner_join` — the predicate over every (left, right) pair."""
    l_idx, r_idx = _all_pairs(left, right)
    keep = np.asarray(predicate.evaluate(left, right, l_idx, r_idx), dtype=bool)
    return l_idx[keep], r_idx[keep]


def conditional_left_join(left: Table, right: Table, predicate):
    """`cudf::conditional_left_join` — as above, plus unmatched left rows at `JOIN_NONE`."""
    l_kept, r_kept = conditional_inner_join(left, right, predicate)
    matched = set(l_kept.tolist())
    extra = [i for i in range(left.num_rows()) if i not in matched]
    return (
        np.concatenate([l_kept, _idx(extra)]),
        np.concatenate([r_kept, np.full(len(extra), JOIN_NONE, dtype=np.int64)]),
    )


def _all_pairs(left: Table, right: Table):
    l_rows, r_rows = left.num_rows(), right.num_rows()
    return (
        np.repeat(np.arange(l_rows, dtype=np.int64), r_rows),
        np.tile(np.arange(r_rows, dtype=np.int64), l_rows),
    )


def _idx(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)
