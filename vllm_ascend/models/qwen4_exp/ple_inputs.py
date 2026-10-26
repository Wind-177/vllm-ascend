# SPDX-License-Identifier: Apache-2.0
"""Fixed-address Model Runner V1 inputs for Qwen4Exp PLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_ascend import envs as ascend_envs
from vllm_ascend.models.input_adapter import ModelInputContext


def pad_ple_graph_inputs(
    context_buffer: torch.Tensor,
    query_start_loc_buffer: torch.Tensor,
    runtime_context: torch.Tensor,
    runtime_query_start_loc: torch.Tensor,
    num_reqs: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refresh fixed-address PLE inputs for ACL graph replay."""
    max_num_reqs = context_buffer.shape[0]
    if not 0 < num_reqs <= max_num_reqs:
        raise ValueError(f"PLE num_reqs must be in [1, {max_num_reqs}], got {num_reqs}")
    if runtime_context.shape != context_buffer[:num_reqs].shape:
        raise ValueError(
            "PLE runtime context shape does not match the active graph rows: "
            f"expected={tuple(context_buffer[:num_reqs].shape)}, "
            f"actual={tuple(runtime_context.shape)}"
        )
    if runtime_query_start_loc.shape != (num_reqs + 1,):
        raise ValueError(
            "PLE runtime query_start_loc must have one boundary per active "
            f"request: expected={(num_reqs + 1,)}, "
            f"actual={tuple(runtime_query_start_loc.shape)}"
        )

    context_buffer.fill_(eos_token_id)
    context_buffer[:num_reqs].copy_(runtime_context, non_blocking=False)
    query_start_loc_buffer[: num_reqs + 1].copy_(runtime_query_start_loc, non_blocking=False)
    query_start_loc_buffer[num_reqs + 1 :].fill_(runtime_query_start_loc[-1])
    return context_buffer, query_start_loc_buffer


class Qwen4ExpPLEInputAdapter:
    """Own PLE history inputs and their graph-stable backing allocations."""

    def __init__(
        self,
        text_config: Any,
        *,
        max_num_reqs: int,
        device: torch.device,
    ) -> None:
        self.context_len = int(text_config.ngram_size) - 1
        if self.context_len <= 0:
            raise ValueError("Qwen4Exp PLE requires ngram_size >= 2")
        self.eos_token_id = int(text_config.eos_token_id)
        self.max_num_reqs = max_num_reqs
        self.device = device
        self._context_buffer: torch.Tensor | None = None
        self._query_start_loc_buffer: torch.Tensor | None = None
        self._context_cpu_buffer: torch.Tensor | None = None
        self._query_start_loc_cpu_buffer: torch.Tensor | None = None
        self._staging_copy_done: Any | None = None

    def _get_context_buffer(self) -> torch.Tensor:
        expected_shape = (self.max_num_reqs, self.context_len)
        if self._context_buffer is None:
            self._context_buffer = torch.empty(
                expected_shape,
                dtype=torch.int32,
                device=self.device,
            )
        elif (
            self._context_buffer.shape != expected_shape
            or self._context_buffer.dtype != torch.int32
            or self._context_buffer.device != self.device
        ):
            raise RuntimeError(
                "Qwen4Exp PLE persistent ngram_context buffer changed "
                f"unexpectedly: shape={tuple(self._context_buffer.shape)}, "
                f"dtype={self._context_buffer.dtype}, "
                f"device={self._context_buffer.device}"
            )
        return self._context_buffer

    def _get_query_start_loc_buffer(self) -> torch.Tensor:
        expected_shape = (self.max_num_reqs + 1,)
        if self._query_start_loc_buffer is None:
            self._query_start_loc_buffer = torch.empty(
                expected_shape,
                dtype=torch.int32,
                device=self.device,
            )
        elif (
            self._query_start_loc_buffer.shape != expected_shape
            or self._query_start_loc_buffer.dtype != torch.int32
            or self._query_start_loc_buffer.device != self.device
        ):
            raise RuntimeError(
                "Qwen4Exp PLE persistent query_start_loc buffer changed "
                f"unexpectedly: shape={tuple(self._query_start_loc_buffer.shape)}, "
                f"dtype={self._query_start_loc_buffer.dtype}, "
                f"device={self._query_start_loc_buffer.device}"
            )
        return self._query_start_loc_buffer

    def _prepare_async_inputs(
        self,
        runtime_context: torch.Tensor,
        runtime_query_start_loc: torch.Tensor,
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Snapshot CPU inputs into owned pinned storage before async H2D."""
        if not 0 < num_reqs <= self.max_num_reqs:
            raise ValueError(f"PLE num_reqs must be in [1, {self.max_num_reqs}], got {num_reqs}")
        context_buffer = self._get_context_buffer()
        query_start_loc_buffer = self._get_query_start_loc_buffer()
        if runtime_context.shape != context_buffer[:num_reqs].shape:
            raise ValueError(
                "PLE runtime context shape does not match the active graph rows: "
                f"expected={tuple(context_buffer[:num_reqs].shape)}, "
                f"actual={tuple(runtime_context.shape)}"
            )
        if runtime_query_start_loc.shape != (num_reqs + 1,):
            raise ValueError(
                "PLE runtime query_start_loc must have one boundary per active "
                f"request: expected={(num_reqs + 1,)}, "
                f"actual={tuple(runtime_query_start_loc.shape)}"
            )

        # PLE preparation is outside the runner's synchronize_input_prep scope.
        # Own a completion event so real, dummy and explicit override calls all
        # protect these CPU DMA sources before any subsequent host overwrite.
        with torch.profiler.record_function("qsa_async/ple_staging_reuse_wait"):
            if self._staging_copy_done is not None and not self._staging_copy_done.query():
                self._staging_copy_done.synchronize()

        used_cpu_staging = False
        with torch.profiler.record_function("qsa_async/ple_context_h2d"):
            if runtime_context.device.type == "cpu":
                if self._context_cpu_buffer is None:
                    self._context_cpu_buffer = torch.empty(
                        (self.max_num_reqs, self.context_len),
                        dtype=torch.int32,
                        device="cpu",
                        pin_memory=True,
                    )
                self._context_cpu_buffer.fill_(self.eos_token_id)
                self._context_cpu_buffer[:num_reqs].copy_(runtime_context)
                context_buffer.copy_(self._context_cpu_buffer, non_blocking=True)
                used_cpu_staging = True
            else:
                context_buffer.fill_(self.eos_token_id)
                context_buffer[:num_reqs].copy_(
                    runtime_context.to(device=self.device, dtype=torch.int32, non_blocking=True),
                    non_blocking=True,
                )

        with torch.profiler.record_function("qsa_async/ple_query_h2d"):
            if runtime_query_start_loc.device.type == "cpu":
                if self._query_start_loc_cpu_buffer is None:
                    self._query_start_loc_cpu_buffer = torch.empty(
                        (self.max_num_reqs + 1,),
                        dtype=torch.int32,
                        device="cpu",
                        pin_memory=True,
                    )
                self._query_start_loc_cpu_buffer[: num_reqs + 1].copy_(runtime_query_start_loc)
                self._query_start_loc_cpu_buffer[num_reqs + 1 :].fill_(
                    int(self._query_start_loc_cpu_buffer[num_reqs])
                )
                query_start_loc_buffer.copy_(self._query_start_loc_cpu_buffer, non_blocking=True)
                used_cpu_staging = True
            else:
                query_start_loc = runtime_query_start_loc.to(
                    device=self.device, dtype=torch.int32, non_blocking=True
                )
                query_start_loc_buffer[: num_reqs + 1].copy_(query_start_loc, non_blocking=True)
                # Stay on device; do not extract a device scalar for padding.
                if num_reqs < self.max_num_reqs:
                    query_start_loc_buffer[num_reqs + 1 :].copy_(
                        query_start_loc[-1:].expand(self.max_num_reqs - num_reqs),
                        non_blocking=True,
                    )

        if used_cpu_staging:
            self._staging_copy_done = torch.npu.Event()
            self._staging_copy_done.record(torch.npu.current_stream())
        return context_buffer, query_start_loc_buffer

    def _build_runtime_context(
        self,
        context: ModelInputContext,
    ) -> torch.Tensor:
        runtime_context = np.full(
            (context.num_reqs_padded, self.context_len),
            self.eos_token_id,
            dtype=np.int32,
        )
        if context.is_dummy:
            return torch.from_numpy(runtime_context)

        for req_idx in range(context.num_reqs):
            context_end = int(context.num_computed_tokens[req_idx])
            context_start = max(0, context_end - self.context_len)
            tokens = context.token_ids[req_idx, context_start:context_end]
            if len(tokens) > 0:
                runtime_context[req_idx, -len(tokens) :] = tokens
        return torch.from_numpy(runtime_context)

    def prepare(
        self,
        model_kwargs: dict[str, Any],
        context: ModelInputContext,
    ) -> None:
        """Populate graph-stable PLE inputs for one target or dummy forward."""
        if not 0 < context.num_reqs_padded <= self.max_num_reqs:
            raise ValueError(
                f"Qwen4Exp PLE num_reqs_padded must be in [1, {self.max_num_reqs}], got {context.num_reqs_padded}"
            )

        runtime_query_start_loc = model_kwargs.get("query_start_loc", context.query_start_loc)[
            : context.num_reqs_padded + 1
        ]
        runtime_context = model_kwargs.get("ngram_context")
        if runtime_context is None:
            runtime_context = self._build_runtime_context(context)
        elif (
            runtime_context.shape != (context.num_reqs_padded, self.context_len) or runtime_context.dtype != torch.int32
        ):
            raise ValueError(
                "Qwen4Exp PLE ngram_context must have shape "
                f"({context.num_reqs_padded}, {self.context_len}) and dtype "
                f"torch.int32, got shape={tuple(runtime_context.shape)}, "
                f"dtype={runtime_context.dtype}"
            )

        if (
            ascend_envs.VLLM_ASCEND_ASYNC_PLE_INPUTS
            and self.device.type == "npu"
            and not torch.npu.is_current_stream_capturing()
        ):
            static_context, static_query_start_loc = self._prepare_async_inputs(
                runtime_context,
                runtime_query_start_loc,
                context.num_reqs_padded,
            )
        else:
            with torch.profiler.record_function("qsa_async/ple_context_h2d"):
                device_context = runtime_context.to(device=self.device, dtype=torch.int32)
            with torch.profiler.record_function("qsa_async/ple_query_h2d"):
                device_query_start_loc = runtime_query_start_loc.to(device=self.device, dtype=torch.int32)
            static_context, static_query_start_loc = pad_ple_graph_inputs(
                self._get_context_buffer(),
                self._get_query_start_loc_buffer(),
                device_context,
                device_query_start_loc,
                context.num_reqs_padded,
                self.eos_token_id,
            )
        model_kwargs["ngram_context"] = static_context
        model_kwargs["query_start_loc"] = static_query_start_loc


__all__ = ["Qwen4ExpPLEInputAdapter", "pad_ple_graph_inputs"]
