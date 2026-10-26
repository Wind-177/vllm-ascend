# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Ascend QSA metadata builder.

Adapted from vllm.models.qwen4_exp.common.qsa_cache at the profiled
revision. CUDA PDL support is removed; producers and consumers must retain
normal NPU stream/event ordering. This candidate requires NPU parity and
performance validation before enabling the environment gate.
"""

import math

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "num_mapped_tokens",
        "num_tokens",
        "max_num_work",
        "num_search_steps",
        "work_search_steps",
    ]
)
def _build_qsa_metadata_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    common_slot_mapping_ptr,
    block_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    visible_blocks_ptr,
    slot_mapping_ptr,
    k_work_metadata_ptr,
    block_table_stride_0: tl.constexpr,
    block_table_stride_1: tl.constexpr,
    num_reqs,
    num_mapped_tokens,
    num_tokens,
    max_num_work,
    num_search_steps,
    work_search_steps,
    storage_block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    circular_buffer_size: tl.constexpr,
    num_block_table_columns: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    REQUEST_SCAN_SIZE: tl.constexpr,
    WORK_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    store_mask = token_idx < num_tokens
    mapped = token_idx < num_mapped_tokens
    search_token_idx = tl.minimum(token_idx, num_mapped_tokens - 1)
    request_idx = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int32)
    # Find the last query start at or before each token. The dynamic loop avoids
    # compiling a kernel variant for every ceil(log2(num_reqs)).
    for step in tl.range(0, num_search_steps):
        candidate = request_idx + (1 << (num_search_steps - step - 1))
        valid_candidate = candidate < num_reqs
        candidate_start = tl.load(
            query_start_loc_ptr + candidate,
            mask=valid_candidate,
            other=num_mapped_tokens + 1,
        )
        advance = valid_candidate & (candidate_start <= search_token_idx)
        request_idx = tl.where(advance, candidate, request_idx)
    query_start = tl.load(query_start_loc_ptr + request_idx, mask=mapped, other=0)
    query_end = tl.load(query_start_loc_ptr + request_idx + 1, mask=mapped, other=0)
    seq_len = tl.load(seq_lens_ptr + request_idx, mask=mapped, other=0)
    logical_position = seq_len - (query_end - query_start) + token_idx - query_start
    logical_position = tl.where(mapped, logical_position, -1)
    tl.store(
        token_to_req_ptr + token_idx,
        tl.where(mapped, request_idx, 0),
        mask=store_mask,
    )
    tl.store(
        logical_positions_ptr + token_idx,
        logical_position,
        mask=store_mask,
    )
    visible_blocks = tl.maximum(
        0,
        tl.minimum(
            (logical_position + 1) // compress_ratio,
            seq_len // compress_ratio,
        ),
    ).to(tl.int32)
    tl.store(visible_blocks_ptr + token_idx, visible_blocks, mask=store_mask)

    # A zero-column tensor may carry a null device pointer. Masked tl.load
    # is insufficient on this NPU compiler: eliminate all block-table loads
    # at compile time for the empty-table case.
    if num_block_table_columns == 0:
        if (circular_buffer_size > 0) or (compress_ratio != 1):
            tl.store(slot_mapping_ptr + token_idx, -1, mask=store_mask)
    else:
        # circular_buffer_size is constexpr, so each builder instance compiles out
        # the other QSA cache owner's slot-mapping rule.
        if circular_buffer_size > 0:
            valid = (
                mapped
                & (logical_position >= 0)
                & (token_idx + circular_buffer_size >= query_end)
                & (num_block_table_columns > 0)
            )
            physical_block = tl.load(
                block_table_ptr + request_idx * block_table_stride_0,
                mask=valid,
                other=-1,
            )
            # Dummy and padding requests own no ring and sit on the null block.
            valid &= physical_block > 0
            slot = physical_block * circular_buffer_size + (
                logical_position % circular_buffer_size
            )
        elif compress_ratio != 1:
            compressed_position = tl.maximum(logical_position, 0) // compress_ratio
            logical_block = compressed_position // storage_block_size
            valid = (
                mapped
                & (logical_position >= 0)
                & ((logical_position + 1) % compress_ratio == 0)
                & (logical_block < num_block_table_columns)
            )
            physical_block = tl.load(
                block_table_ptr
                + request_idx * block_table_stride_0
                + logical_block * block_table_stride_1,
                mask=valid,
                other=-1,
            )
            valid &= physical_block >= 0
            valid &= (
                tl.load(common_slot_mapping_ptr + token_idx, mask=mapped, other=-1) >= 0
            )
            slot = physical_block * storage_block_size + (
                compressed_position % storage_block_size
            )
        if (circular_buffer_size > 0) or (compress_ratio != 1):
            tl.store(
                slot_mapping_ptr + token_idx,
                tl.where(valid, slot, -1),
                mask=store_mask,
            )
    work_tile_start = pid * WORK_BLOCK_SIZE
    has_work_tile = work_tile_start < max_num_work
    if k_work_metadata_ptr is not None and has_work_tile:
        # Every work CTA builds the request prefix in registers. Recomputing this
        # small vector lets CTAs write disjoint work tiles without a grid barrier.
        requests = tl.arange(0, REQUEST_SCAN_SIZE)
        valid_request = requests < num_reqs
        request_query_start = tl.load(
            query_start_loc_ptr + requests, mask=valid_request, other=0
        )
        request_query_end = tl.load(
            query_start_loc_ptr + requests + 1, mask=valid_request, other=0
        )
        request_seq_len = tl.load(seq_lens_ptr + requests, mask=valid_request, other=0)
        request_query_len = request_query_end - request_query_start
        chunk_start = request_seq_len - request_query_len
        num_groups = request_seq_len // compress_ratio - chunk_start // compress_ratio
        # Nonempty requests need one item even without a completed compression
        # group because work item zero also commits the current raw-K suffix.
        work_counts = tl.where(request_query_len > 0, tl.maximum(num_groups, 1), 0)
        work_ends = tl.cumsum(tl.where(valid_request, work_counts, 0), axis=0)
        total_work = tl.sum(tl.where(valid_request, work_counts, 0), axis=0)
        work_offsets = tl.arange(0, WORK_BLOCK_SIZE)
        work = work_tile_start + work_offsets
        in_bounds = work < max_num_work
        active = in_bounds & (work < total_work)
        request = tl.zeros((WORK_BLOCK_SIZE,), dtype=tl.int32)
        # Request zero starts at zero for every active work item. Descending
        # steps find the last request starting at or before this item.
        for step_idx in tl.range(0, work_search_steps):
            step = 1 << (work_search_steps - step_idx - 1)
            candidate = request + step
            valid_candidate = candidate < num_reqs
            # Match the existing Ascend integer-gather adaptation: transport
            # int32 bits through fp32, without any numeric float conversion.
            candidate_start = tl.gather(
                work_ends.to(tl.float32, bitcast=True), candidate - 1, 0
            ).to(tl.int32, bitcast=True)
            advance = active & valid_candidate & (candidate_start <= work)
            request = tl.where(advance, candidate, request)

        owner_work_start = tl.gather(
            work_ends.to(tl.float32, bitcast=True), tl.maximum(request - 1, 0), 0
        ).to(tl.int32, bitcast=True)
        owner_work_start = tl.where(request == 0, 0, owner_work_start)
        work_in_request = work - owner_work_start
        tl.store(
            k_work_metadata_ptr + work * 2,
            tl.where(active, request, -1),
            mask=in_bounds,
        )
        tl.store(
            k_work_metadata_ptr + work * 2 + 1,
            tl.where(active, work_in_request, -1),
            mask=in_bounds,
        )



def build_qsa_metadata_triton(
    common_attn_metadata: CommonAttentionMetadata,
    token_to_req_buffer: torch.Tensor,
    logical_positions_buffer: torch.Tensor,
    visible_blocks_buffer: torch.Tensor,
    slot_mapping_buffer: torch.Tensor,
    *,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int = 0,
    k_work_metadata_buffer: torch.Tensor | None = None,
    request_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build QSA side-cache and optional pre-indexer work metadata."""
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    visible_blocks = visible_blocks_buffer[:num_tokens]
    slot_mapping = slot_mapping_buffer[:num_tokens]
    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1
    assert num_reqs > 0

    if k_work_metadata_buffer is not None:
        if request_capacity is None:
            request_capacity = num_reqs
        assert request_capacity >= num_reqs
        # Pad for tl.arange while keeping the scan width stable across live batches.
        request_scan_size = 1 << int(math.ceil(math.log2(request_capacity)))
        max_num_work = k_work_metadata_buffer.shape[0]
    else:
        request_scan_size = 1
        max_num_work = 0

    if num_tokens == 0 and k_work_metadata_buffer is None:
        return token_to_req, logical_positions, visible_blocks, slot_mapping

    block_table = common_attn_metadata.block_table_tensor
    num_search_steps = int(math.ceil(math.log2(num_reqs)))
    work_search_steps = int(math.ceil(math.log2(num_reqs)))
    # The same grid covers token tiles and, for the compressed cache, work tiles.
    num_token_blocks = cdiv(num_tokens, 128)
    num_work_blocks = (
        cdiv(max_num_work, 256) if k_work_metadata_buffer is not None else 0
    )
    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        block_table,
        token_to_req,
        logical_positions,
        visible_blocks,
        slot_mapping,
        k_work_metadata_buffer,
        block_table.stride(0),
        block_table.stride(1),
        num_reqs,
        num_mapped_tokens,
        num_tokens,
        max_num_work,
        num_search_steps,
        work_search_steps,
        storage_block_size,
        compress_ratio,
        circular_buffer_size,
        block_table.shape[1],
        TOKEN_BLOCK_SIZE=128,
        REQUEST_SCAN_SIZE=request_scan_size,
        WORK_BLOCK_SIZE=256,
        num_warps=4,
    )
    if circular_buffer_size == 0 and compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    return token_to_req, logical_positions, visible_blocks, slot_mapping
