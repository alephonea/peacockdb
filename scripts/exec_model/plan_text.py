"""Rendering a plan as text, for the corpus plan goldens.

The spec's [Node display](../../llm-wiki/tasks/batch_partitioned_executor.md#node-display)
section defines what the real mode's `.plans.txt` will carry: the node, its declared
`PartitionLayout` — lane count, batch layout, key distribution, sort order — and the
parameters that decide what it computes. This is that, at the prototype's level of detail,
so a corpus query's lowering is reviewable as a tree rather than as forty lines of builder
calls, and so a change to one shows up as a diff.

Parameters come from the builder call each node recorded (`nodes.Recipe`), which is why
nothing here knows what a filter or an aggregate is. Frames, schemas and child nodes are
dropped: a frame is the data, a schema is derived, and children are the tree itself.
"""

from __future__ import annotations

import enum
import inspect

import pandas as pd

from .layout import BatchLayout


def render(root, title: str = "") -> str:
    """The plan under `root` as an indented tree, one line per node."""
    lines = [f"== {title}"] if title else []
    _render_node(root, 0, lines)
    return "\n".join(lines) + "\n"


def _render_node(node, depth: int, lines: list) -> None:
    lines.append("  " * depth + f"{node.name()}: {_describe(node)}")
    for child in node.children():
        _render_node(child, depth + 1, lines)


def _describe(node) -> str:
    parts = [node.category().value]
    parts.extend(_parameters(node))
    layout = node.output_partitions()
    if layout is not None:
        parts.append(_layout(layout))
    return ", ".join(parts)


def _layout(layout) -> str:
    text = f"lanes={layout.n}"
    if layout.batch_layout is BatchLayout.SINGLE_BATCH:
        text += ", single_batch"
    if layout.key_distribution.hash_keys:
        text += f", hashed_on={list(layout.key_distribution.hash_keys)}"
    if layout.sort_order.columns:
        text += f", sorted_on={[order.column for order in layout.sort_order.columns]}"
    return text


#: Builder arguments that are not the node's own description: the tree, the data, and what
#: is derived from them.
_SKIPPED = {"name", "child", "children", "build", "probe", "frame", "schema"}


def _parameters(node) -> list:
    """The builder arguments that are not defaults — what this node was asked to do.

    An argument left at its default says nothing about the query, and printing it makes a
    golden that moves when a default does rather than when a plan does.
    """
    recipe = getattr(node, "recipe", None)
    if recipe is None:
        return []
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(recipe.builder).parameters.items()
    }
    described = []
    for key, value in recipe.args.items():
        if key in _SKIPPED or value is None or isinstance(value, pd.DataFrame):
            continue
        if key in defaults and _is_default(value, defaults[key]):
            continue
        rendered = _value(value)
        if rendered is not None:
            described.append(f"{key}={rendered}")
    return described


def _is_default(value, default) -> bool:
    if default is inspect.Parameter.empty:
        return False
    try:
        return bool(value == default)
    except ValueError:                     # an array-like never compares to a scalar
        return False


def _value(value):
    if isinstance(value, enum.Enum):
        return value.name
    if hasattr(value, "func") and hasattr(value, "output"):
        # An aggregate call, as the query wrote it: sum(l_quantity) -> sum_qty
        argument = getattr(value, "column", None) or "*"
        return f"{value.func}({argument})->{value.output}"
    if hasattr(value, "name") and callable(value.name) and not isinstance(value, type):
        try:
            return value.name()                      # an expression renders itself
        except TypeError:
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        rendered = [_value(item) for item in value]
        return "[" + ", ".join(str(item) for item in rendered) + "]"
    if callable(value):
        return getattr(value, "__name__", "fn")      # a hash function, say
    if isinstance(value, (str, int, float, bool)):
        return repr(value)
    return str(value)
