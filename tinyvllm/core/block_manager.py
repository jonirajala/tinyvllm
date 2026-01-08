"""Block Manager - Centralized KV cache block allocation across GPUs."""

from typing import Dict, List, Optional, Tuple


class BlockManager:
    """Manages KV cache block allocation for paged attention."""

    def __init__(self, num_gpus: int, blocks_per_gpu: int, block_size: int):
        self.num_gpus = num_gpus
        self.blocks_per_gpu = blocks_per_gpu
        self.block_size = block_size
        self.free_blocks: Dict[int, List[int]] = {
            gpu_id: list(range(blocks_per_gpu)) for gpu_id in range(num_gpus)
        }
        self.ref_counts: Dict[int, Dict[int, int]] = {
            gpu_id: {block_id: 0 for block_id in range(blocks_per_gpu)}
            for gpu_id in range(num_gpus)
        }
        self.seq_to_gpu: Dict[int, int] = {}
        self.block_tables: Dict[int, List[int]] = {}
        self.seq_positions: Dict[int, int] = {}

    def register_sequence(self, seq_id: int, gpu_id: int = 0) -> None:
        """Register sequence without allocating blocks. Blocks allocated on-demand."""
        self.seq_to_gpu[seq_id] = gpu_id
        self.block_tables[seq_id] = []
        self.seq_positions[seq_id] = 0

    def ensure_block_for_position(self, seq_id: int, pos: int) -> int:
        """Ensure a block exists for the given position. Returns block_id."""
        block_idx = pos // self.block_size
        while len(self.block_tables[seq_id]) <= block_idx:
            gpu_id = self.seq_to_gpu[seq_id]
            if len(self.free_blocks[gpu_id]) == 0:
                raise RuntimeError("Out of KV cache memory!")
            new_block = self.free_blocks[gpu_id].pop()
            self.ref_counts[gpu_id][new_block] = 1
            self.block_tables[seq_id].append(new_block)
        return self.block_tables[seq_id][block_idx]

    def get_slot(self, seq_id: int) -> Tuple[int, int, int]:
        """Get (gpu_id, block_id, offset) for next token write. Allocates new block if needed."""
        pos = self.seq_positions[seq_id]
        block_id = self.ensure_block_for_position(seq_id, pos)
        return (self.seq_to_gpu[seq_id], block_id, pos % self.block_size)

    def advance_position(self, seq_id: int) -> None:
        """Increment position after writing K/V for one token across all layers."""
        self.seq_positions[seq_id] += 1

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get block table for attention computation."""
        return self.block_tables[seq_id]

    def get_context_length(self, seq_id: int) -> int:
        """Get number of tokens written to KV cache."""
        return self.seq_positions[seq_id]

    def get_gpu_for_seq(self, seq_id: int) -> int:
        """Get GPU id for sequence."""
        return self.seq_to_gpu[seq_id]

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a completed sequence."""
        gpu_id = self.seq_to_gpu[seq_id]

        for block in self.block_tables[seq_id]:
            self.ref_counts[gpu_id][block] -= 1
            if self.ref_counts[gpu_id][block] == 0:
                self.free_blocks[gpu_id].append(block)

        del self.seq_to_gpu[seq_id]
        del self.block_tables[seq_id]
        del self.seq_positions[seq_id]

    def get_num_free_blocks(self, gpu_id: Optional[int] = None) -> int:
        """Get number of free blocks. If gpu_id is None, returns total across all GPUs."""
        if gpu_id is None:
            return sum(len(blocks) for blocks in self.free_blocks.values())
        return len(self.free_blocks[gpu_id])

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int) -> None:
        """Fork sequence for beam search. Shares blocks via ref counting."""
        gpu_id = self.seq_to_gpu[src_seq_id]

        self.block_tables[dst_seq_id] = self.block_tables[src_seq_id].copy()
        self.seq_to_gpu[dst_seq_id] = gpu_id
        self.seq_positions[dst_seq_id] = self.seq_positions[src_seq_id]

        for block_id in self.block_tables[src_seq_id]:
            self.ref_counts[gpu_id][block_id] += 1
