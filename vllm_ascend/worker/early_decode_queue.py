# SPDX-License-Identifier: Apache-2.0
"""Owned CPU layouts for the experimental uniform Qwen MTP queue path.

Only scheduled row shapes are cached. Sampled IDs, accepted counts, computed
positions, KV tables, InputBatch storage and device buffers are never cached.
"""

from dataclasses import dataclass

import numpy as np


UNIFORM_MTP_DRAFT_COUNT = 3


@dataclass(frozen=True)
class EarlyDecodeLayout:
    signature: tuple[tuple[int, int], ...]
    req_indices: np.ndarray
    query_positions: np.ndarray
    cu_num_tokens: np.ndarray
    num_valid_tokens: np.ndarray
    spec_metadata: tuple[np.ndarray, ...]

    @classmethod
    def build(cls, signature: tuple[tuple[int, int], ...]) -> "EarlyDecodeLayout":
        if not signature or any(
            scheduled != UNIFORM_MTP_DRAFT_COUNT + 1 or drafts != UNIFORM_MTP_DRAFT_COUNT
            for scheduled, drafts in signature
        ):
            raise ValueError("Early decode queue requires uniform fixed K3 rows")
        scheduled = np.array([row[0] for row in signature], dtype=np.int32)
        drafts = np.array([row[1] for row in signature], dtype=np.int32)
        cu_num_tokens = np.cumsum(scheduled, dtype=np.int64)
        req_indices = np.repeat(np.arange(len(signature), dtype=np.int64), scheduled)
        query_positions = np.arange(int(cu_num_tokens[-1]), dtype=np.int64)
        query_positions -= np.repeat(cu_num_tokens - scheduled, scheduled)
        sampled = drafts + 1
        cu_sampled = np.cumsum(sampled, dtype=np.int32)
        sample_positions = np.arange(int(cu_sampled[-1]), dtype=np.int64)
        sample_positions -= np.repeat(cu_sampled - sampled, sampled)
        logits_indices = np.repeat(cu_num_tokens - sampled, sampled) + sample_positions
        cu_drafts = np.cumsum(drafts, dtype=np.int32)
        draft_positions = np.arange(int(cu_drafts[-1]), dtype=np.int64)
        draft_positions -= np.repeat(cu_drafts - drafts, drafts)
        target_logits_indices = np.repeat(cu_sampled - sampled, drafts)
        # Match the runner's in-place int32 accumulation, rather than NumPy's
        # out-of-place promotion of the int64 relative-position scratch.
        target_logits_indices += draft_positions
        num_valid_tokens = scheduled - drafts
        spec_metadata = (
            cu_drafts,
            cu_sampled,
            logits_indices,
            target_logits_indices,
            cu_sampled - 1,
        )
        for array in (req_indices, query_positions, cu_num_tokens, num_valid_tokens, *spec_metadata):
            array.setflags(write=False)
        return cls(
            signature=signature,
            req_indices=req_indices,
            query_positions=query_positions,
            cu_num_tokens=cu_num_tokens,
            num_valid_tokens=num_valid_tokens,
            spec_metadata=spec_metadata,
        )
