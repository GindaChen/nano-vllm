from collections import deque

from nanovllm.engine.sequence import Sequence


class SeqSlotManager:

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.free_slots: deque[int] = deque(range(max_slots))

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_slots) > 0

    def allocate(self, seq: Sequence):
        seq.seq_slot = self.free_slots.popleft()
        seq.block_table = [seq.seq_slot]    # kept for warmup check

    def deallocate(self, seq: Sequence):
        self.free_slots.append(seq.seq_slot)
        seq.seq_slot = -1
        seq.block_table = []
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        return True    # no block boundary management needed

    def may_append(self, seq: Sequence):
        pass    # no new blocks needed; seq writes to contiguous slot
