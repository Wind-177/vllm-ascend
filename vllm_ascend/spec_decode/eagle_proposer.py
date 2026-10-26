# SPDX-License-Identifier: Apache-2.0

from copy import copy

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.qwen4_exp import Qwen4ExpMTPProposer


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)


class AscendQwen4ExpMTPProposer(
    Qwen4ExpMTPProposer,
    AscendSpecDecodeBaseProposer,
):
    """Ascend proposer for Qwen4Exp's multi-stream MTP draft."""

    hidden_size: int

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(
            self,
            vllm_config,
            device,
            True,
            runner=runner,
        )
        qwen_hidden_size = self._get_hidden_size()
        if self.hidden_size != qwen_hidden_size:
            self.hidden_size = qwen_hidden_size
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=device,
            )
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # FULL graph replay records tensor addresses. Keep a persistent slot
        # buffer for every later draft step and cache group.
        self._per_group_next_step_slot_mappings: dict[tuple[int, int], torch.Tensor] = {}
        self._qwen_graph_metadata_active = False
        self._qwen_graph_metadata_buffers: dict[tuple, torch.Tensor] = {}
        self._qwen_graph_configured_k = self.num_speculative_tokens

    def _propose(self, *args, **kwargs):
        descriptor = kwargs.get("target_model_batch_desc")
        if descriptor is None and len(args) > 7:
            descriptor = args[7]
        sizes = self.compilation_config.cudagraph_capture_sizes or []
        self._qwen_graph_metadata_active = bool(
            self.use_cuda_graph
            and descriptor is not None
            and descriptor.uniform
            and descriptor.num_tokens in sizes
        )
        try:
            if self._qwen_graph_metadata_active:
                scheduled_k = kwargs.get("num_speculative_tokens", args[0] if args else self.num_speculative_tokens)
                if scheduled_k == 0:
                    self._qwen_graph_metadata_active = False
                    return AscendSpecDecodeBaseProposer._propose(self, *args, **kwargs)
                if scheduled_k != self._qwen_graph_configured_k:
                    raise RuntimeError("Qwen MTP graph requires the configured fixed draft count")
                common = kwargs.get("common_attn_metadata")
                if common is None and len(args) > 6:
                    common = args[6]
                expected_tokens = common.num_reqs * (self._qwen_graph_configured_k + 1)
                if expected_tokens != descriptor.num_tokens:
                    raise RuntimeError("Qwen MTP graph requires an exact uniform decode bucket")
                mode, draft_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                    num_tokens=descriptor.num_tokens,
                    uniform_decode=True,
                    has_lora=bool(self.runner.input_batch.lora_id_to_lora_request),
                )
                self._qwen_graph_metadata_active = mode == CUDAGraphMode.FULL
                if self._qwen_graph_metadata_active and draft_descriptor.num_tokens != descriptor.num_tokens:
                    raise RuntimeError("Qwen MTP graph dispatcher changed the uniform decode bucket")
            return AscendSpecDecodeBaseProposer._propose(self, *args, **kwargs)
        finally:
            self._qwen_graph_metadata_active = False

    def _qwen_graph_tensor(self, key, value):
        """Retain one allocation per graph bucket, draft step and cache group."""
        buffers = self._qwen_graph_metadata_buffers
        buffer = buffers.get(key)
        if buffer is None:
            buffer = value.clone()
            buffers[key] = buffer
        else:
            if buffer.shape != value.shape or buffer.dtype != value.dtype or buffer.device != value.device:
                raise RuntimeError(f"Qwen MTP graph metadata allocation changed: {key}")
            buffer.copy_(value)
        return buffer

    def _qwen_graph_metadata_snapshot(self, metadata, key):
        # QSA builders reuse scratch tensors across draft steps. A graph must
        # read the step's own snapshot, not the last builder invocation's data.
        def snapshot(value, path):
            if isinstance(value, torch.Tensor):
                return self._qwen_graph_tensor((*key, *path), value)
            if isinstance(value, list):
                return [snapshot(item, (*path, index)) for index, item in enumerate(value)]
            if isinstance(value, tuple):
                return tuple(snapshot(item, (*path, index)) for index, item in enumerate(value))
            if isinstance(value, dict):
                return {name: snapshot(item, (*path, name)) for name, item in value.items()}
            return value

        result = copy(metadata)
        for name, value in vars(metadata).items():
            setattr(result, name, snapshot(value, (name,)))
        return result

    def _qwen_graph_query_layout(self, key, num_reqs, batch_size, draft_index, device):
        """Initialize static uniform query boundaries before capture, once."""
        cpu_key = (*key, "query_start_loc_cpu")
        device_key = (*key, "query_start_loc")
        buffers = self._qwen_graph_metadata_buffers
        cpu = buffers.get(cpu_key)
        gpu = buffers.get(device_key)
        if cpu is None and gpu is None:
            if self._qwen_graph_metadata_active:
                raise RuntimeError("Qwen MTP query layout was not initialized before replay")
            cpu = torch.arange(num_reqs + 1, dtype=torch.int32)
            if draft_index == 0:
                cpu.mul_(self._qwen_graph_configured_k + 1)
            else:
                cpu.clamp_max_(batch_size)
            cpu = self._qwen_graph_tensor(cpu_key, cpu)
            # The one blocking H2D initialization occurs during dummy/capture
            # setup. Real replay reuses these immutable source allocations.
            gpu = self._qwen_graph_tensor(device_key, cpu.to(device))
        elif cpu is None or gpu is None:
            raise RuntimeError("Qwen MTP query layout cache is incomplete")
        if (
            cpu.shape != (num_reqs + 1,)
            or gpu.shape != cpu.shape
            or cpu.dtype != torch.int32
            or gpu.dtype != torch.int32
            or cpu.device.type != "cpu"
            or gpu.device != device
        ):
            raise RuntimeError("Qwen MTP query layout allocation changed")
        return cpu, gpu

    def _qwen_graph_common_metadata(self, common, group, num_input_tokens, draft_index):
        """Use the same fixed-address layout during capture and real replay."""
        result = self.shallow_copy_metadata(common)
        gid = group.kv_cache_group_id
        key = (num_input_tokens, draft_index, gid, "common")
        batch_size = num_input_tokens // (self.num_speculative_tokens + 1)
        if batch_size < 1:
            raise ValueError("Qwen MTP graph bucket is smaller than one verification query")
        num_reqs = batch_size if draft_index == 0 else num_input_tokens
        num_actual_tokens = num_input_tokens if draft_index == 0 else batch_size
        result.num_reqs = num_reqs
        result.num_actual_tokens = num_actual_tokens
        result.num_input_tokens = num_input_tokens
        result.max_query_len = self.num_speculative_tokens + 1 if draft_index == 0 else 1
        result.decode_token_per_req = result.max_query_len
        result.graph_pad_size = -1

        result.query_start_loc_cpu, result.query_start_loc = self._qwen_graph_query_layout(
            key, num_reqs, batch_size, draft_index, common.query_start_loc.device
        )

        for name in ("seq_lens", "seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound", "num_computed_tokens_cpu"):
            values = getattr(common, name, None)
            if values is not None:
                values = self._adjust_tensor(values, num_reqs).clone()
                values[batch_size:].zero_()
                setattr(result, name, self._qwen_graph_tensor((*key, name), values))

        if gid == self.kv_cache_gid:
            block_table = common.block_table_tensor
            slots = common.slot_mapping
        else:
            block_table = self._per_group_block_tables.get(gid)
            if block_table is None:
                raise RuntimeError(f"Missing Qwen MTP graph block table for cache group {gid}")
            if draft_index == 0:
                slots = self._per_group_slot_mappings.get(gid)
                if slots is None:
                    raise RuntimeError(f"Missing Qwen MTP graph slot mapping for cache group {gid}")
            else:
                positions = common.positions[:batch_size]
                block_size = self._per_group_kernel_block_sizes[gid]
                block_numbers = positions // block_size
                block_ids = block_table[:batch_size].gather(1, block_numbers.to(torch.int64).view(-1, 1)).view(-1)
                slots = block_ids * block_size + positions % block_size
                slots = slots.to(torch.int32)
                slots.masked_fill_(common.slot_mapping[:batch_size] < 0, -1)

        table = self._adjust_tensor(block_table, num_reqs).clone()
        table[batch_size:].zero_()
        result.block_table_tensor = self._qwen_graph_tensor((*key, "block_table"), table)
        slot_buffer = torch.full_like(self.slot_mapping_group[draft_index][:num_input_tokens], -1)
        slot_buffer[:num_actual_tokens].copy_(slots[:num_actual_tokens])
        result.slot_mapping = self._qwen_graph_tensor((*key, "slot_mapping"), slot_buffer)
        return result

    def _build_qwen_graph_group_metadata(self, common, group, num_input_tokens, draft_index):
        group_common = self._qwen_graph_common_metadata(common, group, num_input_tokens, draft_index)
        metadata = group.get_metadata_builder().build_for_drafting(group_common, draft_index)
        return self._qwen_graph_metadata_snapshot(
            metadata, (num_input_tokens, draft_index, group.kv_cache_group_id, tuple(group.layer_names), "output")
        )

    def _build_multi_group_graph_capture_metadata(self, common_attn_metadata, draft_index):
        num_input_tokens = common_attn_metadata.num_input_tokens
        common = self.shallow_copy_metadata(common_attn_metadata)
        batch_size = num_input_tokens // (self.num_speculative_tokens + 1)
        if draft_index > 0:
            # Dummy setup starts from target metadata on every substep. Only
            # values are synthetic; layouts and backing addresses match replay.
            positions = self._get_positions(num_input_tokens)
            common.positions = positions[0] if positions.ndim > 1 else positions
            common.slot_mapping = self.slot_mapping_group[draft_index]
            common.slot_mapping[batch_size:].fill_(-1)
        per_layer = {}
        for group in self.draft_attn_groups:
            metadata = self._build_qwen_graph_group_metadata(common, group, num_input_tokens, draft_index)
            per_layer.update({name: metadata for name in group.layer_names})
        return per_layer

    def _common_attn_metadata_for_draft_group(self, common, group, num_input_tokens, draft_index=0):
        if getattr(self, "_qwen_graph_metadata_active", False):
            return self._qwen_graph_common_metadata(common, group, num_input_tokens, draft_index)
        return common

    def _update_full_graph_params(self, forward_context, num_tokens, draft_attn_metadatas=None):
        primary_group = self._get_primary_draft_attn_group()
        if primary_group.backend.get_name() != "QWEN4_EXP_QSA_ASCEND":
            return AscendSpecDecodeBaseProposer._update_full_graph_params(
                self, forward_context, num_tokens, draft_attn_metadatas
            )

        # Load model-specific backends only in their worker path. QSA uses
        # native GBSA nodes with the fixed metadata tensors above, not the
        # FlashAttention task-group parameter update protocol.
        from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend

        from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend, AscendQSAImpl

        if (
            primary_group.backend is not AscendQSABackend
            or primary_group.backend.get_impl_cls() is not AscendQSAImpl
            or any(
                group.backend is not AscendQSABackend and group.backend is not QSAStateBackend
                for group in self.draft_attn_groups
            )
        ):
            raise NotImplementedError("Qwen MTP QSA FULL graphs require only Ascend QSA and cache-only QSA state groups")
        # Cache-only QSAStateBackend has no attention implementation. Do not
        # call get_impl_cls() on it or suppress callbacks for mixed FA graphs.
        return None

    def attn_update_stack_num_spec_norm(self, *args, **kwargs):
        common, metadata = AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm(self, *args, **kwargs)
        if getattr(self, "_qwen_graph_metadata_active", False):
            draft_index = kwargs.get("draft_index", args[0] if args else None)
            group = kwargs["attn_group"]
            metadata = self._qwen_graph_metadata_snapshot(
                metadata, (common.num_input_tokens, draft_index, group.kv_cache_group_id, tuple(group.layer_names), "output")
            )
        return common, metadata

    def _get_hidden_size(self) -> int:
        text_config = self.draft_model_config.hf_text_config
        return int(text_config.hidden_size * text_config.hc_count)

    def build_draft_attn_metadata(
        self,
        common_attn_metadata,
        num_input_tokens,
        num_actual_tokens,
    ):
        """Build step-0 metadata with each Qwen cache group's block table.

        Ascend's merged MTP path bypasses upstream ``_propose`` and therefore
        does not call Qwen4ExpMTPProposer.build_per_group_and_layer_attn_metadata
        unless it is wired in here.  Without this override the raw and
        compressed QSA builders receive the main-attention block table.
        """
        del num_input_tokens, num_actual_tokens
        if getattr(self, "_qwen_graph_metadata_active", False):
            per_layer = {}
            for group in self.draft_attn_groups:
                metadata = self._build_qwen_graph_group_metadata(
                    common_attn_metadata, group, common_attn_metadata.num_input_tokens, 0
                )
                per_layer.update({name: metadata for name in group.layer_names})
            primary_group = self._get_primary_draft_attn_group()
            return [per_layer], per_layer[primary_group.layer_names[0]]
        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata,
            draft_index=0,
        )
        primary_group = self._get_primary_draft_attn_group()
        primary_metadata = per_layer_attn_metadata[primary_group.layer_names[0]]
        return [per_layer_attn_metadata], primary_metadata

    def _build_cache_only_group_next_step_attn_metadata(
        self,
        common_attn_metadata,
        draft_index,
        num_input_tokens,
        primary_group,
        primary_metadata,
        cache_only_groups,
    ):
        """Rebind later MTP steps to every QSA cache group's address pair."""
        per_layer_attn_metadata = {layer_name: primary_metadata for layer_name in primary_group.layer_names}
        for attn_group in cache_only_groups:
            if getattr(self, "_qwen_graph_metadata_active", False):
                attn_metadata = self._build_qwen_graph_group_metadata(
                    common_attn_metadata, attn_group, num_input_tokens, draft_index
                )
                per_layer_attn_metadata.update({name: attn_metadata for name in attn_group.layer_names})
                continue
            gid = attn_group.kv_cache_group_id
            group_common = common_attn_metadata
            if gid != self.kv_cache_gid:
                block_table = self._per_group_block_tables.get(gid)
                assert block_table is not None, f"Missing Qwen draft block table for KV cache group {gid}"
                group_common = self.shallow_copy_metadata(common_attn_metadata)
                group_common.block_table_tensor = block_table[: group_common.num_reqs]
                block_size = self._per_group_kernel_block_sizes[gid]
                positions = group_common.positions[: group_common.num_actual_tokens]
                block_numbers = positions // block_size
                block_ids = block_table[: positions.shape[0]].gather(
                    dim=1,
                    index=block_numbers.view(-1, 1),
                ).view(-1)
                slot_mapping = block_ids * block_size + positions % block_size

                key = (draft_index, gid)
                slot_buffer = self._per_group_next_step_slot_mappings.get(key)
                if slot_buffer is None:
                    slot_buffer = torch.empty_like(self.slot_mapping_group[draft_index])
                    self._per_group_next_step_slot_mappings[key] = slot_buffer
                slot_buffer[: slot_mapping.shape[0]].copy_(slot_mapping.to(torch.int32))
                slot_buffer[slot_mapping.shape[0] : num_input_tokens].fill_(-1)
                group_common.slot_mapping = slot_buffer[:num_input_tokens]
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                group_common,
                draft_index,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata

    def _get_draft_slot_block_size(self) -> int:
        """Return the kernel block size of Qwen's main draft cache group.

        Qwen3.8 Next is a hybrid GDN/QSA model.  ``has_gdn`` describes the
        target as a whole, while the MTP draft attention writes the QSA main
        cache, whose per-group kernel block size is stored in ``block_size``.
        """
        return self.block_size

    def maybe_pad_and_reduce(self, hidden_states, positions):
        return hidden_states, positions

    def maybe_all_gather_and_unpad(
        self,
        last_hidden_states,
        positions,
        hidden_states=None,
    ):
        return last_hidden_states, positions, hidden_states
