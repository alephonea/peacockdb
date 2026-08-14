"""Run-time state shared by the two drivers.

Output queues live on the *producing* node, not on the consumer's input edge. That is
what makes lane remapping free: an emitter reads its child's single out lane, a forwarder
reads whichever (child, lane) pair `sources_of` names, a join reads lane p of each side —
all without the producer knowing who consumes it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .accounting import ResidentAccountant
from .batch import Batch
from .plan import PlanNodeInfo


@dataclass
class NodeState:
    info: PlanNodeInfo
    out_queues: list[deque] = field(default_factory=list)
    out_done: list[bool] = field(default_factory=list)

    #: lane-scoped categories: one BatchSinglePartitionDriver per lane, made on demand
    lane_drivers: dict = field(default_factory=dict)
    #: cross-lane categories: one executor for the whole node
    cross_executor: object = None

    #: BatchForwarder: per out lane, the rotation cursor and the retired source set
    cursors: list[int] = field(default_factory=list)
    retired: list[set] = field(default_factory=list)

    #: PartitionAccumulator: whether each input lane's Done has been delivered
    lane_done_sent: list[bool] = field(default_factory=list)
    #: PartitionEmitter: whether the single input lane has been drained to the end
    emitter_finished: bool = False

    @classmethod
    def create(cls, info: PlanNodeInfo, n_input_lanes: int) -> "NodeState":
        return cls(
            info=info,
            out_queues=[deque() for _ in range(info.n_lanes)],
            out_done=[False] * info.n_lanes,
            cursors=[0] * info.n_lanes,
            retired=[set() for _ in range(info.n_lanes)],
            lane_done_sent=[False] * n_input_lanes,
        )

    def queued_batches(self) -> int:
        return sum(len(q) for q in self.out_queues)

    def all_done(self) -> bool:
        return all(self.out_done) and self.queued_batches() == 0


class LaneInputs:
    """One lane's view of its node's input edges — one entry per child slot.

    `done(slot)` means no further batch can ever arrive on that slot: the producer
    finished *and* its queue for this lane is empty.
    """

    def __init__(self, sources: list[tuple[NodeState, int]], accountant: ResidentAccountant):
        self._sources = sources
        self._accountant = accountant

    def has(self, slot: int) -> bool:
        state, lane = self._sources[slot]
        return bool(state.out_queues[lane])

    def done(self, slot: int) -> bool:
        state, lane = self._sources[slot]
        return state.out_done[lane] and not state.out_queues[lane]

    def take(self, slot: int) -> Batch:
        state, lane = self._sources[slot]
        batch = state.out_queues[lane].popleft()
        self._accountant.remove_in_flight(batch.byte_size())
        return batch
