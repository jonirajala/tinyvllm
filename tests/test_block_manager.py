"""Tests for BlockManager."""

import pytest
from tinyvllm.core.block_manager import BlockManager


def setup_sequence(bm: BlockManager, seq_id: int, num_tokens: int, gpu_id: int = 0) -> None:
    """Helper to register sequence and pre-allocate blocks for num_tokens."""
    bm.register_sequence(seq_id, gpu_id)
    if num_tokens > 0:
        bm.ensure_block_for_position(seq_id, num_tokens - 1)


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


class TestRegisterSequence:
    def test_register_creates_empty_entry(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.register_sequence(seq_id=0)

        assert 0 in bm.seq_to_gpu
        assert bm.block_tables[0] == []
        assert bm.seq_positions[0] == 0
        assert len(bm.free_blocks[0]) == 10

    def test_register_with_gpu_id(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        bm.register_sequence(seq_id=0, gpu_id=1)

        assert bm.seq_to_gpu[0] == 1


class TestEnsureBlockForPosition:
    def test_allocates_single_block(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.register_sequence(seq_id=0)

        block_id = bm.ensure_block_for_position(seq_id=0, pos=10)

        assert len(bm.block_tables[0]) == 1
        assert len(bm.free_blocks[0]) == 9

    def test_allocates_multiple_blocks(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.register_sequence(seq_id=0)

        bm.ensure_block_for_position(seq_id=0, pos=49)  # ceil(50/16) = 4 blocks

        assert len(bm.block_tables[0]) == 4
        assert len(bm.free_blocks[0]) == 6

    def test_sets_ref_count(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        bm.register_sequence(seq_id=0)
        bm.ensure_block_for_position(seq_id=0, pos=19)

        for block_id in bm.block_tables[0]:
            assert bm.ref_counts[0][block_id] == 1

    def test_out_of_memory(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=2, block_size=16)
        bm.register_sequence(seq_id=0)

        with pytest.raises(RuntimeError, match="Out of KV cache memory"):
            bm.ensure_block_for_position(seq_id=0, pos=99)  # Needs 7 blocks


class TestGetSlot:
    def test_get_slot_basic(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)

        gpu_id, block_id, offset = bm.get_slot(seq_id=0)

        assert gpu_id == 0
        assert block_id == bm.block_tables[0][0]
        assert offset == 0

    def test_get_slot_with_offset(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 5

        gpu_id, block_id, offset = bm.get_slot(seq_id=0)

        assert offset == 5

    def test_get_slot_crosses_block_boundary(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 16

        initial_blocks = len(bm.block_tables[0])
        gpu_id, block_id, offset = bm.get_slot(seq_id=0)

        assert len(bm.block_tables[0]) == initial_blocks + 1
        assert offset == 0

    def test_get_slot_new_block_ref_count(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 16

        bm.get_slot(seq_id=0)
        new_block = bm.block_tables[0][-1]

        assert bm.ref_counts[0][new_block] == 1

    def test_get_slot_out_of_memory(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=1, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 16

        with pytest.raises(RuntimeError, match="Out of KV cache memory"):
            bm.get_slot(seq_id=0)


class TestAdvancePosition:
    def test_advance_position(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)

        assert bm.seq_positions[0] == 0
        bm.advance_position(seq_id=0)
        assert bm.seq_positions[0] == 1
        bm.advance_position(seq_id=0)
        assert bm.seq_positions[0] == 2


class TestGetBlockTable:
    def test_get_block_table(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=50)

        block_table = bm.get_block_table(seq_id=0)

        assert block_table == bm.block_tables[0]
        assert len(block_table) == 4


class TestGetContextLength:
    def test_get_context_length(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)

        assert bm.get_context_length(seq_id=0) == 0

        bm.advance_position(seq_id=0)
        bm.advance_position(seq_id=0)
        bm.advance_position(seq_id=0)

        assert bm.get_context_length(seq_id=0) == 3


class TestGetGpuForSeq:
    def test_get_gpu_for_seq(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10, gpu_id=1)

        assert bm.get_gpu_for_seq(seq_id=0) == 1


class TestFreeSequence:
    def test_free_sequence_returns_blocks(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=32)

        assert len(bm.free_blocks[0]) == 8

        bm.free_sequence(seq_id=0)

        assert len(bm.free_blocks[0]) == 10

    def test_free_sequence_cleans_up(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)

        bm.free_sequence(seq_id=0)

        assert 0 not in bm.seq_to_gpu
        assert 0 not in bm.block_tables
        assert 0 not in bm.seq_positions

    def test_free_sequence_with_shared_blocks(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)
        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        shared_blocks = bm.block_tables[0].copy()

        bm.free_sequence(seq_id=0)

        for block_id in shared_blocks:
            assert block_id not in bm.free_blocks[0]
            assert bm.ref_counts[0][block_id] == 1


class TestGetNumFreeBlocks:
    def test_get_num_free_blocks_specific_gpu(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        bm.free_blocks[0] = bm.free_blocks[0][:7]

        assert bm.get_num_free_blocks(gpu_id=0) == 7
        assert bm.get_num_free_blocks(gpu_id=1) == 10

    def test_get_num_free_blocks_all_gpus(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        bm.free_blocks[0] = bm.free_blocks[0][:7]

        assert bm.get_num_free_blocks() == 17


class TestForkSequence:
    def test_fork_copies_block_table(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=32)

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        assert bm.block_tables[1] == bm.block_tables[0]
        assert bm.block_tables[1] is not bm.block_tables[0]

    def test_fork_copies_position(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10)
        bm.seq_positions[0] = 5

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        assert bm.seq_positions[1] == 5

    def test_fork_increments_ref_counts(self):
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=32)

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        for block_id in bm.block_tables[0]:
            assert bm.ref_counts[0][block_id] == 2

    def test_fork_same_gpu(self):
        bm = BlockManager(num_gpus=2, blocks_per_gpu=10, block_size=16)
        setup_sequence(bm, seq_id=0, num_tokens=10, gpu_id=1)

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)

        assert bm.seq_to_gpu[1] == bm.seq_to_gpu[0] == 1


class TestIntegration:
    def test_full_sequence_lifecycle(self):
        """Test: register → write tokens → free"""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)

        bm.register_sequence(seq_id=0)

        # Simulate writing 20 tokens (crosses block boundary)
        for i in range(20):
            gpu_id, block_id, offset = bm.get_slot(seq_id=0)
            bm.advance_position(seq_id=0)

        assert bm.get_context_length(seq_id=0) == 20
        assert len(bm.get_block_table(seq_id=0)) == 2

        bm.free_sequence(seq_id=0)
        assert bm.get_num_free_blocks() == 10

    def test_multiple_sequences(self):
        """Test multiple concurrent sequences"""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=20, block_size=16)

        setup_sequence(bm, seq_id=0, num_tokens=20)
        setup_sequence(bm, seq_id=1, num_tokens=30)
        setup_sequence(bm, seq_id=2, num_tokens=10)

        assert bm.get_num_free_blocks() == 15

        bm.free_sequence(seq_id=1)
        assert bm.get_num_free_blocks() == 17

    def test_beam_search_scenario(self):
        """Test fork and diverge scenario"""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=20, block_size=16)

        bm.register_sequence(seq_id=0)
        for _ in range(10):
            bm.get_slot(seq_id=0)
            bm.advance_position(seq_id=0)

        bm.fork_sequence(src_seq_id=0, dst_seq_id=1)
        bm.fork_sequence(src_seq_id=0, dst_seq_id=2)

        assert bm.block_tables[0] == bm.block_tables[1] == bm.block_tables[2]

        for seq_id in [0, 1, 2]:
            bm.get_slot(seq_id=seq_id)
            bm.advance_position(seq_id=seq_id)

        bm.free_sequence(seq_id=2)

        assert 0 in bm.block_tables
        assert 1 in bm.block_tables
