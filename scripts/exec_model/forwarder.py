"""`BatchForwarder` — the routing-only category, and its three standard mappings.

`GpuMergePartitions`, `GpuUnion` and `GpuInterleave` never touch rows and never buffer:
they relabel lanes. One driver arm serves all three by cycling `sources_of(p)`, so the
merge's round-robin and the interleave's per-lane child rotation are the same rule
applied to different mappings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BatchForwarder(ABC):
    @abstractmethod
    def out_lanes(self) -> int: ...

    @abstractmethod
    def sources_of(self, out_lane: int) -> list[tuple[int, int]]:
        """The (child index, child lane) pairs feeding `out_lane`, in service order."""


class MergePartitionsForwarder(BatchForwarder):
    """N → 1: lane 0 ← [(0,0), (0,1), …, (0,N-1)], cycled round-robin."""

    def __init__(self, n_in: int):
        self.n_in = n_in

    def out_lanes(self) -> int:
        return 1

    def sources_of(self, out_lane: int) -> list[tuple[int, int]]:
        assert out_lane == 0
        return [(0, lane) for lane in range(self.n_in)]


class UnionForwarder(BatchForwarder):
    """Lane counts sum: output lane k is served by exactly one input lane."""

    def __init__(self, child_lane_counts):
        self.child_lane_counts = list(child_lane_counts)
        self._map: list[tuple[int, int]] = [
            (child, lane)
            for child, count in enumerate(self.child_lane_counts)
            for lane in range(count)
        ]

    def out_lanes(self) -> int:
        return len(self._map)

    def sources_of(self, out_lane: int) -> list[tuple[int, int]]:
        return [self._map[out_lane]]


class InterleaveForwarder(BatchForwarder):
    """Lane p ← [(0,p), (1,p), …, (k-1,p)] — child-major, lane count preserved."""

    def __init__(self, n_children: int, n_lanes: int):
        self.n_children = n_children
        self.n_lanes = n_lanes

    def out_lanes(self) -> int:
        return self.n_lanes

    def sources_of(self, out_lane: int) -> list[tuple[int, int]]:
        return [(child, out_lane) for child in range(self.n_children)]
