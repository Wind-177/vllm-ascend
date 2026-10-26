# SPDX-License-Identifier: Apache-2.0
"""Unit regressions for Qwen4Exp MTP hybrid KV-cache groups."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.spec_decode.eagle_proposer import AscendQwen4ExpMTPProposer
from vllm_ascend.spec_decode.qwen4_exp import Qwen4ExpMTPProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def _qsa_graph_update_proposer():
    class QSAImpl:
        pass

    class QSABackend:
        @staticmethod
        def get_name():
            return "QWEN4_EXP_QSA_ASCEND"

        @staticmethod
        def get_impl_cls():
            return QSAImpl

    class StateBackend:
        get_impl_cls = MagicMock(side_effect=AssertionError("Cache-only groups have no attention implementation"))

    class FABackend:
        @staticmethod
        def get_name():
            return "FLASH_ATTN"

    main = SimpleNamespace(backend=QSABackend)
    raw = SimpleNamespace(backend=StateBackend)
    compressed = SimpleNamespace(backend=StateBackend)
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    proposer.draft_attn_groups = [main, raw, compressed]
    proposer._get_primary_draft_attn_group = MagicMock(return_value=main)
    proposer.update_stream = object()
    proposer.vllm_config = SimpleNamespace(speculative_config=object())
    modules = {
        "vllm.models.qwen4_exp.common.qsa_cache": SimpleNamespace(QSAStateBackend=StateBackend),
        "vllm_ascend.models.qwen4_exp.qsa": SimpleNamespace(AscendQSABackend=QSABackend, AscendQSAImpl=QSAImpl),
    }
    return proposer, modules, QSABackend, StateBackend, FABackend


@pytest.mark.parametrize("entry", ["runtime_FULL", "dummy_postcapture"])
def test_qsa_full_graph_update_uses_fixed_tensors_without_fia_callback(entry):
    proposer, modules, _, state_backend, _ = _qsa_graph_update_proposer()
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    metadata = [{"main": object(), "raw": object(), "compressed": object()}]
    callback = MagicMock(side_effect=AssertionError("QSA has no FIA graph-update callback"))
    base_globals = AscendSpecDecodeBaseProposer._update_full_graph_params.__globals__
    with patch.dict("sys.modules", modules), patch.dict(base_globals, {"update_full_graph_params": callback}):
        if entry == "runtime_FULL":
            # Exercise the real inherited entry used after _runnable in _propose.
            proposer._update_full_graph_params_if_needed(context, 8, metadata)
        else:
            # dummy_run calls this method directly after a non-capture FULL run.
            proposer._update_full_graph_params(context, 8, metadata)
    callback.assert_not_called()
    state_backend.get_impl_cls.assert_not_called()


@pytest.mark.parametrize("with_state_group", [False, True])
def test_non_qsa_primary_preserves_real_base_full_graph_update(with_state_group):
    proposer, modules, _, _, fa_backend = _qsa_graph_update_proposer()
    proposer.draft_attn_groups[0].backend = fa_backend
    if not with_state_group:
        proposer.draft_attn_groups = proposer.draft_attn_groups[:1]
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    metadata = [{"fa": object()}]
    callback = MagicMock()
    base_globals = AscendSpecDecodeBaseProposer._update_full_graph_params.__globals__
    with patch.dict("sys.modules", modules), patch.dict(base_globals, {"update_full_graph_params": callback}):
        proposer._update_full_graph_params_if_needed(context, 4, metadata)
    callback.assert_called_once_with(
        fa_backend, proposer.update_stream, context, 4, proposer.vllm_config,
        proposer.vllm_config.speculative_config, draft_attn_metadatas=metadata,
    )


@pytest.mark.parametrize("other_backend", ["FA", "unknown", "QSA_name_only"])
def test_qsa_primary_rejects_mixed_or_unrecognized_graph_backends(other_backend):
    proposer, modules, qsa_backend, state_backend, fa_backend = _qsa_graph_update_proposer()
    if other_backend == "FA":
        backend = fa_backend
    elif other_backend == "unknown":
        backend = object()
    else:
        backend = type("NameOnlyQSA", (qsa_backend,), {})
    proposer.draft_attn_groups.append(SimpleNamespace(backend=backend))
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    callback = MagicMock()
    base_globals = AscendSpecDecodeBaseProposer._update_full_graph_params.__globals__
    with patch.dict("sys.modules", modules), patch.dict(base_globals, {"update_full_graph_params": callback}):
        with pytest.raises(NotImplementedError, match="only Ascend QSA and cache-only QSA state"):
            proposer._update_full_graph_params_if_needed(context, 8, [{}])
    callback.assert_not_called()
    state_backend.get_impl_cls.assert_not_called()


def test_qsa_backend_name_cannot_bypass_backend_and_impl_identity_checks():
    proposer, modules, qsa_backend, _, _ = _qsa_graph_update_proposer()
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    impostor = type("NameOnlyQSA", (qsa_backend,), {})
    proposer.draft_attn_groups[0].backend = impostor
    with patch.dict("sys.modules", modules), pytest.raises(NotImplementedError):
        proposer._update_full_graph_params_if_needed(context, 8, [{}])
    proposer.draft_attn_groups[0].backend = qsa_backend
    with patch.dict("sys.modules", modules), patch.object(qsa_backend, "get_impl_cls", return_value=object):
        with pytest.raises(NotImplementedError):
            proposer._update_full_graph_params_if_needed(context, 8, [{}])


def test_non_full_runtime_does_not_enter_graph_parameter_update():
    proposer, _, _, _, _ = _qsa_graph_update_proposer()
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    with patch.object(proposer, "_update_full_graph_params", side_effect=AssertionError("Not a FULL graph")):
        proposer._update_full_graph_params_if_needed(context, 8, [{}])


def _attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )


def test_map_draft_layer_accepts_direct_group_spec() -> None:
    layer_name = "model.mtp.0.self_attn"
    spec = _attention_spec()
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )
    proposer = SimpleNamespace(_draft_attn_layer_names={layer_name})

    layer_to_gid, layer_to_spec = Qwen4ExpMTPProposer._map_draft_layers_to_groups(proposer, config)

    assert layer_to_gid == {layer_name: 0}
    assert layer_to_spec == {layer_name: spec}


def test_map_draft_layer_accepts_uniform_group_spec() -> None:
    layer_name = "model.mtp.0.self_attn"
    spec = _attention_spec()
    uniform = UniformTypeKVCacheSpecs.from_specs({layer_name: spec})
    assert uniform is not None
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=uniform)],
    )
    proposer = SimpleNamespace(_draft_attn_layer_names={layer_name})

    layer_to_gid, layer_to_spec = Qwen4ExpMTPProposer._map_draft_layers_to_groups(proposer, config)

    assert layer_to_gid == {layer_name: 0}
    assert layer_to_spec == {layer_name: spec}


def _draft_group(gid: int, layer_name: str, metadata: object):
    builder = MagicMock()
    builder.build_for_drafting.return_value = metadata
    group = SimpleNamespace(
        kv_cache_group_id=gid,
        layer_names=[layer_name],
        get_metadata_builder=MagicMock(return_value=builder),
    )
    return group, builder


def test_ascend_step_zero_uses_qwen_per_group_block_tables() -> None:
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    main_metadata = object()
    raw_metadata = object()
    main_group, main_builder = _draft_group(0, "draft.main", main_metadata)
    raw_group, raw_builder = _draft_group(1, "draft.raw", raw_metadata)
    proposer.draft_attn_groups = [main_group, raw_group]
    proposer.kv_cache_gid = 0
    raw_table = torch.tensor([[23, 24]], dtype=torch.int32)
    raw_slots = torch.tensor([23 * 128 + 7, -1], dtype=torch.int32)
    proposer._per_group_block_tables = {1: raw_table}
    proposer._per_group_slot_mappings = {1: raw_slots}
    proposer._get_primary_draft_attn_group = MagicMock(return_value=main_group)
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        block_table_tensor=torch.tensor([[3, 4]], dtype=torch.int32),
        slot_mapping=torch.tensor([3 * 128 + 7], dtype=torch.int32),
    )

    multi_step, primary = proposer.build_draft_attn_metadata(
        common,
        num_input_tokens=4,
        num_actual_tokens=2,
    )

    main_common = main_builder.build_for_drafting.call_args.kwargs["common_attn_metadata"]
    raw_common = raw_builder.build_for_drafting.call_args.kwargs["common_attn_metadata"]
    assert main_common is common
    assert main_common.block_table_tensor.tolist() == [[3, 4]]
    assert raw_common is not common
    assert raw_common.block_table_tensor.data_ptr() == raw_table.data_ptr()
    assert raw_common.slot_mapping.data_ptr() == raw_slots.data_ptr()
    assert multi_step == [{"draft.main": main_metadata, "draft.raw": raw_metadata}]
    assert primary is main_metadata


def test_ascend_later_step_uses_qwen_cache_only_group_block_table() -> None:
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    proposer.kv_cache_gid = 0
    raw_table = torch.tensor([[41, 42]], dtype=torch.int32)
    proposer._per_group_block_tables = {1: raw_table}
    proposer._per_group_kernel_block_sizes = {0: 16, 1: 128}
    proposer._per_group_next_step_slot_mappings = {}
    proposer.slot_mapping_group = [
        torch.full((8,), -1, dtype=torch.int32) for _ in range(3)
    ]
    primary_metadata = object()
    raw_metadata = object()
    primary_group, _ = _draft_group(0, "draft.main", primary_metadata)
    raw_group, raw_builder = _draft_group(1, "draft.raw", raw_metadata)
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        positions=torch.tensor([129], dtype=torch.int64),
        block_table_tensor=torch.tensor([[7, 8]], dtype=torch.int32),
    )

    per_layer = proposer._build_cache_only_group_next_step_attn_metadata(
        common,
        draft_index=2,
        num_input_tokens=8,
        primary_group=primary_group,
        primary_metadata=primary_metadata,
        cache_only_groups=[raw_group],
    )

    raw_common, draft_index = raw_builder.build_for_drafting.call_args.args
    assert draft_index == 2
    assert raw_common is not common
    assert raw_common.block_table_tensor.data_ptr() == raw_table.data_ptr()
    assert raw_common.slot_mapping.tolist() == [42 * 128 + 1] + [-1] * 7
    assert per_layer == {
        "draft.main": primary_metadata,
        "draft.raw": raw_metadata,
    }


def test_qwen_mtp_uses_primary_group_kernel_block_size_for_later_steps() -> None:
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    proposer.has_gdn = True
    proposer.kernel_block_size = 16
    proposer.block_size = 128

    # Qwen is a hybrid GDN/QSA model, but its draft layer writes the QSA main
    # cache. The target-wide GDN flag must not select the GDN kernel size.
    assert proposer._get_draft_slot_block_size() == 128


def _graph_proposer():
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    proposer.kv_cache_gid = 0
    proposer.num_speculative_tokens = 3
    proposer._qwen_graph_configured_k = 3
    proposer._qwen_graph_metadata_buffers = {}
    proposer._qwen_graph_metadata_active = False
    proposer._per_group_next_step_slot_mappings = {}
    proposer.uses_mrope = False
    proposer.runner = SimpleNamespace(
        cudagraph_dispatcher=SimpleNamespace(dispatch=MagicMock(return_value=(CUDAGraphMode.FULL, SimpleNamespace(num_tokens=8)))),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
    )
    proposer._per_group_kernel_block_sizes = {0: 16, 1: 128, 2: 64}
    proposer._per_group_block_tables = {
        1: torch.tensor([[41, 42, 43], [51, 52, 53]], dtype=torch.int32),
        2: torch.tensor([[71, 72, 73], [81, 82, 83]], dtype=torch.int32),
    }
    proposer._per_group_slot_mappings = {
        1: torch.arange(8, dtype=torch.int32) + 4100,
        2: torch.arange(8, dtype=torch.int32) + 7100,
    }
    proposer.slot_mapping_group = [torch.full((8,), -1, dtype=torch.int32) for _ in range(3)]
    proposer.positions = torch.tensor([129, 130, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    groups = []
    for gid, name in ((0, "draft.main"), (1, "draft.raw"), (2, "draft.compressed")):
        scratch = torch.zeros(8, dtype=torch.int32)

        def build(common, draft_index, *, scratch=scratch):
            # This deliberately models QSA's builder-owned scratch reuse.
            scratch.copy_(common.slot_mapping)
            return SimpleNamespace(
                block_table=common.block_table_tensor,
                seq_lens=common.seq_lens,
                slot_mapping=scratch,
                query_start_loc=common.query_start_loc,
                query_start_loc_cpu=common.query_start_loc_cpu,
                num_actual_tokens=common.num_actual_tokens,
                num_reqs=common.num_reqs,
                max_query_len=common.max_query_len,
            )

        builder = SimpleNamespace(build_for_drafting=build)
        groups.append(SimpleNamespace(
            kv_cache_group_id=gid,
            layer_names=[name],
            get_metadata_builder=lambda builder=builder: builder,
        ))
    proposer.draft_attn_groups = groups
    proposer._get_primary_draft_attn_group = lambda: groups[0]
    for step in range(3):
        for group in groups:
            proposer._qwen_graph_query_layout(
                (8, step, group.kv_cache_group_id, "common"),
                2 if step == 0 else 8,
                2,
                step,
                torch.device("cpu"),
            )
    return proposer


def _graph_common(step):
    later = step > 0
    return SimpleNamespace(
        num_reqs=8 if later else 2,
        num_actual_tokens=2 if later else 8,
        num_input_tokens=8,
        max_query_len=1 if later else 4,
        positions=torch.tensor([129, 130, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[3, 4, 5], [7, 8, 9]], dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int32) + 100,
        query_start_loc=torch.arange(9 if later else 3, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(9 if later else 3, dtype=torch.int32),
        seq_lens=torch.tensor([130 + step, 131 + step], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([130 + step, 131 + step], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([130 + step, 131 + step], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([129 + step, 130 + step], dtype=torch.int32),
    )


def _real_graph_metadata(proposer, common, step):
    proposer._qwen_graph_metadata_active = True
    if step == 0:
        return proposer.build_draft_attn_metadata(common, 8, 8)[0][0]
    primary = proposer.draft_attn_groups[0]
    group_common = proposer._common_attn_metadata_for_draft_group(common, primary, 8, step)
    metadata = primary.get_metadata_builder().build_for_drafting(group_common, step)
    metadata = proposer._qwen_graph_metadata_snapshot(metadata, (8, step, 0, tuple(primary.layer_names), "output"))
    return proposer._build_cache_only_group_next_step_attn_metadata(
        common, step, 8, primary, metadata, proposer.draft_attn_groups[1:]
    )


@pytest.mark.parametrize("step", [0, 1, 2])
def test_graph_capture_and_real_metadata_share_group_and_step_addresses(step):
    """A successful capture must bind the same group allocations as replay."""
    proposer = _graph_proposer()
    common = _graph_common(step)
    captured = proposer._build_multi_group_graph_capture_metadata(common, step)
    addresses = {
        (name, field): getattr(metadata, field).data_ptr()
        for name, metadata in captured.items()
        for field in ("block_table", "slot_mapping", "seq_lens", "query_start_loc")
    }
    actual = _real_graph_metadata(proposer, _graph_common(step), step)
    for name, metadata in actual.items():
        for field in ("block_table", "slot_mapping", "seq_lens", "query_start_loc"):
            assert getattr(metadata, field).data_ptr() == addresses[name, field]
    assert actual["draft.main"].block_table[:2].tolist() == [[3, 4, 5], [7, 8, 9]]
    assert actual["draft.raw"].block_table[:2].tolist() == [[41, 42, 43], [51, 52, 53]]
    assert actual["draft.compressed"].block_table[:2].tolist() == [[71, 72, 73], [81, 82, 83]]
    expected_query = [0, 4, 8] if step == 0 else [0, 1, 2, 2, 2, 2, 2, 2, 2]
    assert actual["draft.raw"].query_start_loc_cpu.tolist() == expected_query
    assert actual["draft.raw"].num_actual_tokens == (8 if step == 0 else 2)
    if step > 0:
        assert actual["draft.raw"].slot_mapping.tolist() == [42 * 128 + 1, 52 * 128 + 2] + [-1] * 6
        assert actual["draft.compressed"].slot_mapping.tolist() == [73 * 64 + 1, 83 * 64 + 2] + [-1] * 6


def test_graph_steps_do_not_alias_builder_scratch_or_other_groups():
    proposer = _graph_proposer()
    per_step = [_real_graph_metadata(proposer, _graph_common(step), step) for step in range(3)]
    pointers = [
        metadata.slot_mapping.data_ptr()
        for metadata_by_layer in per_step
        for metadata in metadata_by_layer.values()
    ]
    assert len(set(pointers)) == 9
    first_slots = per_step[0]["draft.raw"].slot_mapping.clone()
    proposer._per_group_slot_mappings[1].add_(100)
    _real_graph_metadata(proposer, _graph_common(2), 2)
    assert torch.equal(per_step[0]["draft.raw"].slot_mapping, first_slots)


def test_graph_builders_in_the_same_cache_group_have_independent_outputs():
    proposer = _graph_proposer()
    first = proposer.draft_attn_groups[1]

    def build(common, draft_index):
        result = first.get_metadata_builder().build_for_drafting(common, draft_index)
        return SimpleNamespace(**{**vars(result), "slot_mapping": result.slot_mapping + 77})

    second = SimpleNamespace(
        kv_cache_group_id=1,
        layer_names=["draft.another_raw_owner"],
        get_metadata_builder=lambda: SimpleNamespace(build_for_drafting=build),
    )
    proposer.draft_attn_groups.append(second)
    metadata = proposer._build_multi_group_graph_capture_metadata(_graph_common(0), 0)
    assert metadata["draft.raw"].slot_mapping.data_ptr() != metadata["draft.another_raw_owner"].slot_mapping.data_ptr()
    assert torch.equal(metadata["draft.raw"].slot_mapping + 77, metadata["draft.another_raw_owner"].slot_mapping)


@pytest.mark.parametrize("missing", ["table", "slots"])
def test_graph_capture_fails_closed_when_cache_group_addressing_is_missing(missing):
    proposer = _graph_proposer()
    if missing == "table":
        del proposer._per_group_block_tables[1]
    else:
        del proposer._per_group_slot_mappings[1]
    with pytest.raises(RuntimeError, match="Missing Qwen MTP graph"):
        proposer._build_multi_group_graph_capture_metadata(_graph_common(0), 0)


def test_graph_metadata_keeps_padded_rows_null_and_k_steps_independent():
    proposer = _graph_proposer()
    result = _real_graph_metadata(proposer, _graph_common(1), 1)
    for metadata in result.values():
        assert torch.count_nonzero(metadata.block_table[2:]) == 0
        assert torch.count_nonzero(metadata.seq_lens[2:]) == 0
        assert metadata.slot_mapping[2:].tolist() == [-1] * 6


def test_graph_side_group_preserves_invalid_primary_slots():
    proposer = _graph_proposer()
    common = _graph_common(1)
    common.slot_mapping[0] = -1
    result = _real_graph_metadata(proposer, common, 1)
    assert result["draft.raw"].slot_mapping[0] == -1
    assert result["draft.compressed"].slot_mapping[0] == -1


def test_graph_metadata_refuses_to_change_an_existing_allocation():
    proposer = _graph_proposer()
    proposer._qwen_graph_tensor((8, 0, 0, "test"), torch.zeros(2))
    with pytest.raises(RuntimeError, match="allocation changed"):
        proposer._qwen_graph_tensor((8, 0, 0, "test"), torch.zeros(3))


@pytest.mark.parametrize(
    ("uniform", "num_tokens", "use_graph", "expected"),
    [(True, 8, True, True), (False, 8, True, False), (True, 64, True, False), (True, 8, False, False)],
)
def test_only_captured_uniform_graph_buckets_use_fixed_metadata(uniform, num_tokens, use_graph, expected):
    """Mixed/prefill, oversized and eager batches retain their existing path."""
    proposer = _graph_proposer()
    proposer.use_cuda_graph = use_graph
    proposer.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[4, 8, 12, 16, 20, 24, 28, 32])
    descriptor = SimpleNamespace(uniform=uniform, num_tokens=num_tokens)

    def base_propose(self, **kwargs):
        assert self._qwen_graph_metadata_active is expected
        return "result"

    with patch.object(AscendSpecDecodeBaseProposer, "_propose", base_propose):
        assert proposer._propose(target_model_batch_desc=descriptor, common_attn_metadata=_graph_common(0)) == "result"
    assert proposer._qwen_graph_metadata_active is False


def test_graph_metadata_fails_closed_if_uniform_bucket_requires_request_padding():
    proposer = _graph_proposer()
    proposer.use_cuda_graph = True
    proposer.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[4, 8])
    common = _graph_common(0)
    common.num_reqs = 1
    with pytest.raises(RuntimeError, match="exact uniform decode bucket"):
        proposer._propose(target_model_batch_desc=SimpleNamespace(uniform=True, num_tokens=8), common_attn_metadata=common)
    assert proposer._qwen_graph_metadata_active is False


def test_graph_metadata_fails_closed_if_scheduled_draft_count_changes():
    proposer = _graph_proposer()
    proposer.use_cuda_graph = True
    proposer.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[4, 8])
    with pytest.raises(RuntimeError, match="configured fixed draft count"):
        proposer._propose(num_speculative_tokens=2, target_model_batch_desc=SimpleNamespace(uniform=True, num_tokens=8), common_attn_metadata=_graph_common(0))
    assert proposer._qwen_graph_metadata_active is False


def test_graph_metadata_checks_actual_dispatch_mode_before_changing_layout():
    proposer = _graph_proposer()
    proposer.use_cuda_graph = True
    proposer.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[4, 8])
    proposer.runner.cudagraph_dispatcher.dispatch.return_value = (CUDAGraphMode.NONE, SimpleNamespace(num_tokens=8))

    def base_propose(self, **kwargs):
        assert self._qwen_graph_metadata_active is False
        return "eager-fallback"

    with patch.object(AscendSpecDecodeBaseProposer, "_propose", base_propose):
        result = proposer._propose(target_model_batch_desc=SimpleNamespace(uniform=True, num_tokens=8), common_attn_metadata=_graph_common(0))
    assert result == "eager-fallback"


def test_graph_metadata_uses_configured_k_after_an_empty_draft_step():
    proposer = _graph_proposer()
    proposer.use_cuda_graph = True
    proposer.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[4, 8])
    proposer.num_speculative_tokens = 0

    def base_propose(self, **kwargs):
        assert self._qwen_graph_metadata_active is True
        return "restored-k3"

    with patch.object(AscendSpecDecodeBaseProposer, "_propose", base_propose):
        result = proposer._propose(num_speculative_tokens=3, target_model_batch_desc=SimpleNamespace(uniform=True, num_tokens=8), common_attn_metadata=_graph_common(0))
    assert result == "restored-k3"


@pytest.mark.parametrize("bucket", [4, 8])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_static_query_layout_transfers_once_before_capture_and_never_during_replay(bucket, step):
    proposer = _graph_proposer()
    key = (bucket, step, 99, "common")
    batch_size = bucket // 4
    num_reqs = batch_size if step == 0 else bucket
    original_to = torch.Tensor.to
    transfers = []

    def tracked_to(tensor, *args, **kwargs):
        transfers.append((args, kwargs))
        return original_to(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "to", tracked_to):
        cpu, device = proposer._qwen_graph_query_layout(key, num_reqs, batch_size, step, torch.device("cpu"))
        proposer._qwen_graph_metadata_active = True
        repeated_cpu, repeated_device = proposer._qwen_graph_query_layout(key, num_reqs, batch_size, step, torch.device("cpu"))
    assert len(transfers) == 1
    assert repeated_cpu.data_ptr() == cpu.data_ptr()
    assert repeated_device.data_ptr() == device.data_ptr()
    expected = [index * 4 for index in range(num_reqs + 1)] if step == 0 else [min(index, batch_size) for index in range(num_reqs + 1)]
    assert cpu.tolist() == expected
    assert device.tolist() == expected


def test_static_query_layout_refuses_lazy_initialization_during_real_replay():
    proposer = _graph_proposer()
    proposer._qwen_graph_metadata_active = True
    with patch.object(torch.Tensor, "to", side_effect=AssertionError("No runtime H2D allowed")):
        with pytest.raises(RuntimeError, match="not initialized before replay"):
            proposer._qwen_graph_query_layout((8, 1, 99, "common"), 8, 2, 1, torch.device("cpu"))


def test_static_query_layout_refuses_a_partial_cache():
    proposer = _graph_proposer()
    key = (8, 1, 99, "common")
    proposer._qwen_graph_metadata_buffers[(*key, "query_start_loc_cpu")] = torch.arange(9, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="cache is incomplete"):
        proposer._qwen_graph_query_layout(key, 8, 2, 1, torch.device("cpu"))


@pytest.mark.parametrize("step", [1, 2])
def test_mrope_capture_uses_first_logical_row_without_a_positions_attribute(step):
    """Qwen text M-RoPE stores only mrope_positions in the actual proposer."""
    proposer = _graph_proposer()
    logical_positions = proposer.positions.clone()
    del proposer.positions
    proposer.uses_mrope = True
    proposer.mrope_positions = torch.stack([logical_positions, logical_positions + 64, logical_positions + 128])
    proposer.slot_mapping_group[step][:2] = torch.tensor([5, 6], dtype=torch.int32)
    captured = proposer._build_multi_group_graph_capture_metadata(_graph_common(step), step)
    assert captured["draft.raw"].slot_mapping.tolist() == [42 * 128 + 1, 52 * 128 + 2] + [-1] * 6
    assert captured["draft.compressed"].slot_mapping.tolist() == [73 * 64 + 1, 83 * 64 + 2] + [-1] * 6
    actual = _real_graph_metadata(proposer, _graph_common(step), step)
    assert not hasattr(proposer, "positions")
    assert captured["draft.raw"].slot_mapping.data_ptr() == actual["draft.raw"].slot_mapping.data_ptr()
    assert actual["draft.raw"].slot_mapping.tolist() == [42 * 128 + 1, 52 * 128 + 2] + [-1] * 6
    assert actual["draft.compressed"].slot_mapping.tolist() == [73 * 64 + 1, 83 * 64 + 2] + [-1] * 6
