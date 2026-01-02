"""Block Manager - Centralized allocation across GPUs."""

from typing import Dict, List, Tuple
from math import ceil

class BlockManager:
    def __init__(self, num_gpus: int, blocks_per_gpu: int, block_size: int):
        self.num_gpus = num_gpus  # Number of GPUs
        self.blocks_per_gpu = blocks_per_gpu  # Total blocks per GPU
        self.block_size = block_size  # Tokens per block (e.g., 16)
        self.free_blocks = {gpu_id: list(range(blocks_per_gpu)) for gpu_id in range(num_gpus)}  # gpu_id → [available block IDs]
        self.ref_counts = {gpu_id: {block_id: 0 for block_id in range(blocks_per_gpu)} for gpu_id in range(num_gpus)}  # gpu_id → {block_id: count} for beam search fork
        self.seq_to_gpu = {}  # seq_id → gpu_id (which GPU owns this sequence)
        self.block_tables = {}  # seq_id → [block_ids] (logical → physical mapping)
        self.seq_positions = {}  # seq_id → current token position (how many K/V written)

    def allocate_sequence(self, seq_id: int, num_tokens: int):
        """
        Seq_id is the id of the sequence (prompt)
        Num_tokens is the number of tokens in the sequence.

        Allocate a sequence of blocks for a given sequence id.
        This is called when a new sequence is started.
        It allocates the blocks for the sequence and returns the gpu id.
        """
        blocks_needed = ceil(num_tokens / self.block_size)

        # find the gpu with the most free blocks
        free_blocks_per_gpu = {gpu_id: len(self.free_blocks[gpu_id]) for gpu_id in self.free_blocks}
        max_free_blocks_gpu = max(free_blocks_per_gpu, key=free_blocks_per_gpu.get)

        # Raise RuntimeError if not enough blocks
        if free_blocks_per_gpu[max_free_blocks_gpu] < blocks_needed:
            raise RuntimeError(f"Not enough free blocks on GPU {max_free_blocks_gpu}")
        
        # Pop blocks from free list, set ref_count=1
        blocks = [self.free_blocks[max_free_blocks_gpu].pop() for _ in range(blocks_needed)]
        for block in blocks:
            self.ref_counts[max_free_blocks_gpu][block] = 1

        # Store in block_tables, seq_to_gpu, seq_positions=0
        self.block_tables[seq_id] = blocks
        self.seq_to_gpu[seq_id] = max_free_blocks_gpu
        self.seq_positions[seq_id] = 0

        return max_free_blocks_gpu

    def get_slot(self, seq_id: int) -> Tuple[int, int, int]:
        """
        Checks if the current allocated block has space for one token
        If the current allocated block has no space for one token, it allocates a new block.

        Get the slot (gpu_id, block_id, offset) for the next write.
        This is called when a new token is generated.
        It returns the slot for the next write.

        This assumes that gpu has enough free blocks to allocate a new block. This will be done in the scheduler.
          Option 2: Preemption (vLLM does this)
            - Pause a lower-priority sequence
            - Free its blocks
            - Resume it later when space available

          Option 3: Swap to CPU (vLLM does this)
            - Move some blocks from GPU → CPU memory
            - Swap back when needed
            - Slower but avoids killing sequences

        """
        # Get (gpu_id, block_id, offset) for next write
        pos = self.seq_positions[seq_id]
        logical_block = pos // self.block_size
        offset = pos % self.block_size

        # If logical_block >= len(block_table), allocate new block
        if logical_block >= len(self.block_tables[seq_id]):
            if len(self.free_blocks[self.seq_to_gpu[seq_id]]) == 0:
              raise RuntimeError("Out of KV cache memory!")

            # Current blocks full, grab one more
            new_block = self.free_blocks[self.seq_to_gpu[seq_id]].pop()
            self.ref_counts[self.seq_to_gpu[seq_id]][new_block] = 1
            self.block_tables[seq_id].append(new_block)

        return (self.seq_to_gpu[seq_id], self.block_tables[seq_id][logical_block], offset)


    def advance_position(self, seq_id: int):
        """
        advance_position() is called after writing K/V for all layers for one token.
        It increments the position by 1.

        - Each token has K/V written at same position across all layers
        - Layer 0: position 5, Layer 1: position 5, ... Layer 31: position 5
        - Only after all layers wrote to position 5, move to position 6

        """
        self.seq_positions[seq_id] += 1

    def get_block_table(self, seq_id: int) -> List[int]:
        """
        get_block_table() is used during attention computation - to know which blocks to read K/V from.
        It returns the block table for the sequence.
        """
        return self.block_tables[seq_id]

    def get_context_length(self, seq_id: int) -> int:
        """
        get_context_length() is used during attention computation - to know how many tokens to read K/V from.
        It returns the context length for the sequence.
        """
        return self.seq_positions[seq_id]

    def get_gpu_for_seq(self, seq_id: int) -> int:
        """
        get_gpu_for_seq() is used to know which GPU the sequence is allocated to.
        It returns the GPU id for the sequence.
        """
        return self.seq_to_gpu[seq_id]

    def free_sequence(self, seq_id: int):
        """
        free_sequence() is called when a sequence is done generating - either hit EOS token or max tokens.
            - Decrement ref_counts for all blocks
            - Return blocks with ref_count=0 to free_blocks
            - Delete from seq_to_gpu, block_tables, seq_positions
        """
        gpu_id = self.seq_to_gpu[seq_id]

        for block in self.block_tables[seq_id]:
            self.ref_counts[gpu_id][block] -= 1

            # if no one else is using this block, return it to the free_blocks pool
            if self.ref_counts[gpu_id][block] == 0:
                self.free_blocks[gpu_id].append(block)
        
        # delete the sequence from the block manager
        del self.seq_to_gpu[seq_id]
        del self.block_tables[seq_id]
        del self.seq_positions[seq_id]

    def get_num_free_blocks(self, gpu_id: int = None) -> int:
        """
        get_num_free_blocks() is used to know how many free blocks are available on the GPU.
        It returns the number of free blocks for the GPU.
        """
        # free blocks on all gpus
        if gpu_id is None:
            return sum(len(blocks) for blocks in self.free_blocks.values())
        # free blocks on specific gpu
        else:
            return len(self.free_blocks[gpu_id])

    def can_allocate(self, num_tokens: int) -> bool:
        """
        can_allocate() is used to know if any GPU has enough free blocks to allocate a new sequence.
        It returns True if any GPU has enough free blocks, False otherwise.
        """
        blocks_needed = ceil(num_tokens / self.block_size)
        # Can ANY GPU fit this sequence?
        return any(
            len(self.free_blocks[gpu]) >= blocks_needed
            for gpu in range(self.num_gpus)
        )

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int):
        """
        fork_sequence() is used in beam search - when you need to split one sequence into multiple candidates.
        """
        gpu_id = self.seq_to_gpu[src_seq_id]

        # Copy block table (same blocks, not copied data!)
        self.block_tables[dst_seq_id] = self.block_tables[src_seq_id].copy()
        self.seq_to_gpu[dst_seq_id] = gpu_id
        self.seq_positions[dst_seq_id] = self.seq_positions[src_seq_id]

        # Increment ref_counts (blocks now shared)
        for block_id in self.block_tables[src_seq_id]:
            self.ref_counts[gpu_id][block_id] += 1    
        