"""Tests for BlockManager."""

import pytest
from tinyvllm.core.block_manager import BlockManager


class TestBlockManagerInit:
    def test_init_single_gpu(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=100, block_size=16)
        assert bm.num_gpus == 1
        assert bm.blocks_per_gpu == 100
        assert bm.block_size == 16
        assert len(bm.free_blocks[0]) == 100

    def test_init_multi_gpu(self):
        bm = BlockManager(num_gpus=4, blocks_per_gpu=50, block_size=16)
        assert bm.num_gpus == 4
        for gpu_id in range(4):
            assert len(bm.free_blocks[gpu_id]) == 50

    def test_init_ref_counts_zero(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        for block_id in range(10):
            assert bm.ref_counts[0][block_id] == 0


class TestAllocateSequence:
    def test_allocate_single_block(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        gpu_id = bm.allocate_sequence(seq_id=0, num_tokens=10)

        assert gpu_id == 0
        assert len(bm.block_tables[0]) == 1  # 10 tokens, block_size=16 → 1 block
        assert bm.seq_positions[0] == 0
        assert len(bm.free_blocks[0]) == 9  # 1 block used

    def test_allocate_multiple_blocks(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=50)

        assert len(bm.block_tables[0]) == 4  # ceil(50/16) = 4 blocks
        assert len(bm.free_blocks[0]) == 6

    def test_allocate_sets_ref_count(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=20)

        for block_id in bm.block_tables[0]:
            assert bm.ref_counts[0][block_id] == 1

    def test_allocate_load_balancing(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)

        # Use up some blocks on GPU 0
        bm.free_blocks[0] = bm.free_blocks[0][:3]  # Only 3 free on GPU 0

        # Should allocate on GPU 1 (has more free blocks)
        gpu_id = bm.allocate_sequence(seq_id=0, num_tokens=10)
        assert gpu_id == 1

    def test_allocate_out_of_memory(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=2, block_size=16)

        with pytest.raises(RuntimeError, match="Not enough free blocks"):
            bm.allocate_sequence(seq_id=0, num_tokens=100)  # Needs 7 blocks, only 2 available


class TestGetSlot:
    def test_get_slot_basic(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)

        gpu_id, block_id, offset = bm.get_slot(seq_id=0)

        assert gpu_id == 0
        assert block_id == bm.block_tables[0][0]
        assert offset == 0  # Position 0 → offset 0

    def test_get_slot_with_offset(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 5  # Simulate 5 tokens written

        gpu_id, block_id, offset = bm.get_slot(seq_id=0)

        assert offset == 5

    def test_get_slot_crosses_block_boundary(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)  # 1 block allocated
        bm.seq_positions[0] = 16  # Position 16 → need second block

        initial_blocks = len(bm.block_tables[0])
        gpu_id, block_id, offset = bm.get_slot(seq_id=0)

        assert len(bm.block_tables[0]) == initial_blocks + 1  # New block allocated
        assert offset == 0  # First position in new block

    def test_get_slot_new_block_ref_count(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 16  # Force new block allocation

        bm.get_slot(seq_id=0)
        new_block = bm.block_tables[0][-1]

        assert bm.ref_counts[0][new_block] == 1

    def test_get_slot_out_of_memory(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=1, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)  # Uses the only block
        bm.seq_positions[0] = 16  # Need new block but none available

        with pytest.raises(RuntimeError, match="Out of KV cache memory"):
            bm.get_slot(seq_id=0)


class TestAdvancePosition:
    def test_advance_position(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)

        assert bm.seq_positions[0] == 0
        bm.advance_position(seq_id=0)
        assert bm.seq_positions[0] == 1
        bm.advance_position(seq_id=0)
        assert bm.seq_positions[0] == 2


class TestGetBlockTable:
    def test_get_block_table(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=50)

        block_table = bm.get_block_table(seq_id=0)

        assert block_table == bm.block_tables[0]
        assert len(block_table) == 4


class TestGetContextLength:
    def test_get_context_length(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)

        assert bm.get_context_length(seq_id=0) == 0

        bm.advance_position(seq_id=0)
        bm.advance_position(seq_id=0)
        bm.advance_position(seq_id=0)

        assert bm.get_context_length(seq_id=0) == 3


class TestGetGpuForSeq:
    def test_get_gpu_for_seq(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)

        # Force allocation on GPU 1
        bm.free_blocks[0] = []
        bm.allocate_sequence(seq_id=0, num_tokens=10)

        assert bm.get_gpu_for_seq(seq_id=0) == 1


class TestFreeSequence:
    def test_free_sequence_returns_blocks(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=32)  # 2 blocks

        assert len(bm.free_blocks[0]) == 8

        bm.free_sequence(seq_id=0)

        assert len(bm.free_blocks[0]) == 10  # All blocks returned

    def test_free_sequence_cleans_up(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)

        bm.free_sequence(seq_id=0)

        assert 0 not in bm.seq_to_gpu
        assert 0 not in bm.block_tables
        assert 0 not in bm.seq_positions

    def test_free_sequence_with_shared_blocks(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)
        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        # Both sequences share same blocks, ref_count=2
        shared_blocks = bm.block_tables[0].copy()

        bm.free_sequence(seq_id=0)

        # Blocks should NOT be returned to free list (still used by seq 1)
        for block_id in shared_blocks:
            assert block_id not in bm.free_blocks[0]
            assert bm.ref_counts[0][block_id] == 1


class TestGetNumFreeBlocks:
    def test_get_num_free_blocks_specific_gpu(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        bm.free_blocks[0] = bm.free_blocks[0][:7]  # 7 free on GPU 0

        assert bm.get_num_free_blocks(gpu_id=0) == 7
        assert bm.get_num_free_blocks(gpu_id=1) == 10

    def test_get_num_free_blocks_all_gpus(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        bm.free_blocks[0] = bm.free_blocks[0][:7]  # 7 free on GPU 0

        assert bm.get_num_free_blocks() == 17  # 7 + 10


class TestCanAllocate:
    def test_can_allocate_enough_space(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)

        assert bm.can_allocate(num_tokens=100) is True  # Needs 7 blocks, have 10

    def test_can_allocate_not_enough_space(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=2, block_size=16)

        assert bm.can_allocate(num_tokens=100) is False  # Needs 7 blocks, have 2

    def test_can_allocate_multi_gpu(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=5, block_size=16)
        bm.free_blocks[0] = []  # GPU 0 full

        # GPU 1 has 5 blocks, enough for 50 tokens
        assert bm.can_allocate(num_tokens=50) is True


class TestForkSequence:
    def test_fork_copies_block_table(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=32)

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        assert bm.block_tables[1] == bm.block_tables[0]
        assert bm.block_tables[1] is not bm.block_tables[0]  # Different list object

    def test_fork_copies_position(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 5

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        assert bm.seq_positions[1] == 5

    def test_fork_increments_ref_counts(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.allocate_sequence(seq_id=0, num_tokens=32)

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        for block_id in bm.block_tables[0]:
            assert bm.ref_counts[0][block_id] == 2

    def test_fork_same_gpu(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        bm.free_blocks[0] = []  # Force allocation on GPU 1
        bm.allocate_sequence(seq_id=0, num_tokens=10)

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        assert bm.seq_to_gpu[1] == bm.seq_to_gpu[0] == 1


class TestIntegration:
    def test_full_sequence_lifecycle(self):
        """Test: allocate → write tokens → free"""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)

        # Allocate
        bm.allocate_sequence(seq_id=0, num_tokens=10)
        assert bm.get_num_free_blocks() == 9

        # Simulate writing 20 tokens (crosses block boundary)
        for i in range(20):
            gpu_id, block_id, offset = bm.get_slot(seq_id=0)
            # Would write K/V here
            bm.advance_position(seq_id=0)

        assert bm.get_context_length(seq_id=0) == 20
        assert len(bm.get_block_table(seq_id=0)) == 2  # 20 tokens → 2 blocks

        # Free
        bm.free_sequence(seq_id=0)
        assert bm.get_num_free_blocks() == 10  # All blocks returned

    def test_multiple_sequences(self):
        """Test multiple concurrent sequences"""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=20, block_size=16)

        bm.allocate_sequence(seq_id=0, num_tokens=20)
        bm.allocate_sequence(seq_id=1, num_tokens=30)
        bm.allocate_sequence(seq_id=2, num_tokens=10)

        # 2 + 2 + 1 = 5 blocks used
        assert bm.get_num_free_blocks() == 15

        bm.free_sequence(seq_id=1)
        assert bm.get_num_free_blocks() == 17

    def test_beam_search_scenario(self):
        """Test fork and diverge scenario"""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=20, block_size=16)

        # Initial sequence
        bm.allocate_sequence(seq_id=0, num_tokens=10)
        for _ in range(10):
            bm.get_slot(seq_id=0)
            bm.advance_position(seq_id=0)

        # Fork for beam search
        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)
        bm.fork_sequence(src_seq_id=0, dst_seq_id=2)

        # All share same blocks
        assert bm.block_tables[0] == bm.block_tables[1] == bm.block_tables[2]

        # Diverge: each generates different tokens
        for seq_id in [0, 1, 2]:
            bm.get_slot(seq_id=seq_id)
            bm.advance_position(seq_id=seq_id)

        # Free one beam
        bm.free_sequence(seq_id=2)

        # Original blocks still in use by seq 0 and 1
        assert 0 in bm.block_tables
        assert 1 in bm.block_tables
