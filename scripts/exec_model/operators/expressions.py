"""A small typed expression IR, evaluated with pandas' vectorized operators only.

Mirrors what `cudf::ast` accepts: column references, literals, and a fixed set of binary
and unary operations over whole columns. There is deliberately no escape hatch — no
`apply`, no lambda — because anything reachable only through one has no C++ counterpart
(subset rule 2 in `frame.py`).

Columns are named here rather than addressed by ordinal. The real IR uses ordinals
(`ColumnRef.index`, architecture.md's "Column indexing"), and nothing checks them; names
are chosen for the prototype because a wrong name fails loudly and a wrong ordinal is the
silent class of bug the prototype is not trying to model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class Expr:
    """Evaluates to one column, given a frame."""

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Col(Expr):
    column: str

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        if self.column not in frame.columns:
            raise KeyError(f"no column {self.column!r} in {list(frame.columns)}")
        return frame[self.column]

    def name(self) -> str:
        return self.column


@dataclass(frozen=True)
class Lit(Expr):
    value: object

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series([self.value] * len(frame), index=frame.index)

    def name(self) -> str:
        return repr(self.value)


#: The operator set cudf::ast exposes over two columns. Anything not here is unsupported
#: on purpose: adding a python-only operation would make the prototype unportable.
_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
}


@dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr
    alias: str | None = None

    def __post_init__(self):
        if self.op not in _BINARY:
            raise ValueError(f"unsupported operator {self.op!r}; cudf::ast has no counterpart")

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return _BINARY[self.op](self.left.evaluate(frame), self.right.evaluate(frame))

    def name(self) -> str:
        return self.alias or f"({self.left.name()} {self.op} {self.right.name()})"


@dataclass(frozen=True)
class Not(Expr):
    inner: Expr
    alias: str | None = None

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return ~self.inner.evaluate(frame)

    def name(self) -> str:
        return self.alias or f"(not {self.inner.name()})"


@dataclass(frozen=True)
class IsNotNull(Expr):
    """The predicate #137 wants the planner to insert under a shuffle."""

    inner: Expr
    alias: str | None = None

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return self.inner.evaluate(frame).notna()

    def name(self) -> str:
        return self.alias or f"({self.inner.name()} is not null)"


@dataclass(frozen=True)
class Sqrt(Expr):
    """`cudf::unary_operator::SQRT`, which the IR gains for the stddev finalize."""

    inner: Expr
    alias: str | None = None

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        import numpy as np

        return pd.Series(np.sqrt(self.inner.evaluate(frame).to_numpy(dtype="float64")))

    def name(self) -> str:
        return self.alias or f"sqrt({self.inner.name()})"


@dataclass(frozen=True)
class Case(Expr):
    """Search-form `CASE WHEN c THEN t … ELSE e END`.

    Folded from the last branch backwards, each step selecting `then` where the condition
    holds and the accumulated result otherwise — the same shape as `build_column_case`'s
    `copy_if_else` fold in `cpp/src/expr.cpp`. Value-form CASE is deliberately absent: it
    is what #57 withheld on the GPU.
    """

    whens: tuple           # ((condition, then), …)
    otherwise: Expr
    alias: str | None = None

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        result = self.otherwise.evaluate(frame)
        for condition, then in reversed(self.whens):
            mask = condition.evaluate(frame).fillna(False).astype(bool)
            result = then.evaluate(frame).where(mask, result)
        return result

    def name(self) -> str:
        if self.alias:
            return self.alias
        arms = " ".join(f"WHEN {c.name()} THEN {t.name()}" for c, t in self.whens)
        return f"CASE {arms} ELSE {self.otherwise.name()} END"


@dataclass(frozen=True)
class Alias(Expr):
    """A projection's `expr AS name`."""

    inner: Expr
    as_name: str

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return self.inner.evaluate(frame)

    def name(self) -> str:
        return self.as_name


def project(frame: pd.DataFrame, exprs: list[Expr]) -> pd.DataFrame:
    """Build a new frame from an expression list — column order is the list order."""
    return pd.DataFrame({expr.name(): expr.evaluate(frame).to_numpy() for expr in exprs})
