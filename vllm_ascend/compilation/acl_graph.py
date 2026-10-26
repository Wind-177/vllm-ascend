# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

from ..utils import super_kernel_scope, use_updatable_graph, weak_ref_tensors
from .updatable_graph import (
    ContextSource,
    SharedSource,
    UpdatableGraph,
)

_acl_graph_wrappers: weakref.WeakSet[Any] = weakref.WeakSet()
_STREAM_RESOURCE_ERROR_CODE = "207008"
_STREAM_RESOURCE_ERROR_MARKERS = (
    "insufficient_stream_resources",
    "stream resources are insufficient",
)
_OLD_HDK_CAPTURE_ERROR_MARKERS = ("alloc sq cq fail",)


def _h_native_input_signature(args, kwargs) -> tuple | None:
    """Record graph input addresses/metadata without reading tensor data."""
    entries = []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            shape, stride = tuple(value.shape), tuple(value.stride())
            # Symbolic values must not be materialized while guarding replay.
            if not all(type(item) is int and item >= 0 for item in (*shape, *stride)):
                entries.append(("unsupported", path, "non-concrete tensor layout"))
                return True
            storage = value.untyped_storage()
            storage_offset = value.storage_offset()
            element_size = value.element_size()
            required_elements = storage_offset
            if all(size > 0 for size in shape):
                required_elements += 1 + sum((size - 1) * step for size, step in zip(shape, stride))
            entries.append((
                "tensor", path, value.data_ptr(), shape, stride,
                str(value.dtype), value.device.type, value.device.index,
                storage.data_ptr(), storage.nbytes(), storage_offset, element_size,
                required_elements * element_size,
            ))
        elif value is None or type(value) in (bool, int, float, str):
            entries.append(("value", path, type(value), value))
        elif type(value) in (tuple, list):
            entries.append(("container", path, type(value), len(value)))
            for index, item in enumerate(value):
                if not visit(item, (*path, index)):
                    return False
        elif type(value) is dict and all(type(key) is str for key in value):
            keys = tuple(sorted(value))
            entries.append(("dict", path, keys))
            for key in keys:
                if not visit(value[key], (*path, key)):
                    return False
        else:
            entries.append(("unsupported", path, f"{type(value).__module__}.{type(value).__qualname__}"))
        return True

    try:
        if not visit(args, ("args",)) or not visit(kwargs, ("kwargs",)):
            return None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    return tuple(entries)


def _h_native_input_metadata(signature) -> dict:
    """Serialize only pointer and layout metadata, never tensor contents."""
    tensors, other = [], []
    if signature is None:
        return {"supported": False, "tensors": tensors, "other_entries": other}
    supported = True
    for item in signature:
        kind, path = item[:2]
        if kind == "tensor":
            tensors.append({
                "path": list(path), "data_ptr": item[2], "shape": list(item[3]), "stride": list(item[4]),
                "dtype": item[5], "device": f"{item[6]}:{item[7]}", "device_type": item[6],
                "storage_data_ptr": item[8], "storage_nbytes": item[9], "storage_offset": item[10],
                "element_size": item[11], "required_storage_bytes": item[12],
            })
        elif kind == "value":
            other.append({"kind": kind, "path": list(path), "type": item[2].__name__, "value": item[3]})
        elif kind == "container":
            other.append({"kind": kind, "path": list(path), "type": item[2].__name__, "length": item[3]})
        elif kind == "dict":
            other.append({"kind": kind, "path": list(path), "keys": list(item[2])})
        else:
            supported = False
            other.append({"kind": kind, "path": list(path), "type": item[2]})
    return {"supported": supported, "tensors": tensors, "other_entries": other}


def _h_unused_inputs_embeds_diagnostics(before: dict, after: dict, proof: dict | None) -> dict:
    """Prove the one live MM-runner buffer absent from a captured raw-text graph."""
    path = ("kwargs", "inputs_embeds")
    captured = {tuple(item["path"]): item for item in before["tensors"]}
    runtime = {tuple(item["path"]): item for item in after["tensors"]}
    values = {tuple(item["path"]): item for item in before["other_entries"] if item["kind"] == "value"}
    captured_value = values.get(path)
    proof = proof if type(proof) is dict else {}
    owner = proof.get("owner")
    owner = owner if type(owner) is dict else {}
    actual = runtime.get(path)
    expected_paths = {("kwargs", name) for name in ("input_ids", "ngram_context", "positions", "query_start_loc")}
    owner_fields = (
        "path", "data_ptr", "storage_data_ptr", "storage_offset", "dtype", "device", "device_type",
        "element_size", "shape", "stride", "storage_nbytes", "required_storage_bytes",
    )
    owner_matches = actual is not None and all(name in owner and actual[name] == owner[name] for name in owner_fields)
    valid_span = actual is not None and (
        actual["device_type"] == "npu" and actual["data_ptr"] > 0 and actual["storage_data_ptr"] > 0
        and actual["storage_offset"] >= 0 and actual["element_size"] > 0
        and 0 < actual["required_storage_bytes"] <= actual["storage_nbytes"]
    )
    shape_matches = actual is not None and len(actual["shape"]) == 2 and (
        actual["shape"][0] == proof.get("num_tokens") and actual["shape"][1] > 0
    )
    disjoint = actual is not None and all(
        actual["storage_data_ptr"] + actual["storage_nbytes"] <= item["storage_data_ptr"]
        or item["storage_data_ptr"] + item["storage_nbytes"] <= actual["storage_data_ptr"]
        for item in captured.values()
    )
    checks = {
        "strict_text_forward_authorized": proof.get("strict_text_forward_authorized") is True,
        "exact_qwen_raw_runnable": proof.get("exact_qwen_raw_runnable") is True,
        "raw_input_tokens_required": proof.get("requires_raw_input_tokens") is True,
        "pp1": proof.get("pp1") is True,
        "runner_owned_buffer": proof.get("owner_name") == "runner.inputs_embeds.gpu",
        "captured_explicit_none": captured_value is not None
        and captured_value["type"] == "NoneType" and captured_value["value"] is None,
        "audited_captured_paths": set(captured) == expected_paths,
        "captured_and_live_input_ids_present": ("kwargs", "input_ids") in captured
        and ("kwargs", "input_ids") in runtime,
        "only_extra_tensor_is_inputs_embeds": runtime.keys() - captured.keys() == {path},
        "no_missing_captured_tensor": captured.keys() <= runtime.keys(),
        "owned_view_identity_matches": owner_matches,
        "valid_owned_npu_span": valid_span,
        "owned_view_shape_matches_bucket": shape_matches,
        "disjoint_from_all_captured_storage": disjoint,
    }
    return {
        "allowed": all(checks.values()), "checks": checks, "path": list(path),
        "reason": "captured explicit None on audited raw-token text graph; taskless replay does not consume live inputs_embeds",
        "owner": owner, "live": actual, "captured_value": captured_value,
        "authorization": {key: value for key, value in proof.items() if key != "owner"},
    }


def _h_native_input_diagnostics(captured, live, optional_inputs_embeds_proof: dict | None = None) -> dict:
    """Guard captured GPU storage; replay does not consume live view/scalar metadata."""
    before, after = _h_native_input_metadata(captured), _h_native_input_metadata(live)
    captured_tensors = {tuple(item["path"]): item for item in before["tensors"]}
    live_tensors = {tuple(item["path"]): item for item in after["tensors"]}
    raw_paths_match = bool(captured_tensors) and captured_tensors.keys() == live_tensors.keys()
    optional_policy = _h_unused_inputs_embeds_diagnostics(before, after, optional_inputs_embeds_proof)
    ignored_uncaptured_tensors = []
    if optional_policy["allowed"]:
        live_tensors.pop(("kwargs", "inputs_embeds"))
        ignored_uncaptured_tensors.append(optional_policy)
    paths_match = bool(captured_tensors) and captured_tensors.keys() == live_tensors.keys()
    identity_differences, ignored_metadata_differences = [], []
    for path in sorted(captured_tensors.keys() | live_tensors.keys(), key=repr):
        old, new = captured_tensors.get(path), live_tensors.get(path)
        if old is None or new is None:
            identity_differences.append({
                "path": list(path), "field": "tensor_path", "captured": old, "live": new,
            })
            continue
        for name in ("data_ptr", "storage_data_ptr", "storage_offset", "dtype", "device", "element_size"):
            if old[name] != new[name]:
                identity_differences.append({
                    "path": list(path), "field": name, "captured": old[name], "live": new[name],
                })
        if old["device_type"] != "npu" or new["device_type"] != "npu":
            identity_differences.append({
                "path": list(path), "field": "npu_input_required",
                "captured": old["device_type"], "live": new["device_type"],
            })
        if (
            old["data_ptr"] <= 0 or old["storage_data_ptr"] <= 0
            or old["storage_offset"] < 0 or new["storage_offset"] < 0
            or old["element_size"] <= 0 or old["storage_nbytes"] < 0
            or old["required_storage_bytes"] > old["storage_nbytes"]
            or new["storage_nbytes"] < max(old["storage_nbytes"], old["required_storage_bytes"])
        ):
            identity_differences.append({
                "path": list(path), "field": "captured_storage_capacity",
                "captured": old["storage_nbytes"], "live": new["storage_nbytes"],
                "required_storage_bytes": old["required_storage_bytes"],
            })
        for name in ("shape", "stride"):
            if old[name] != new[name]:
                ignored_metadata_differences.append({
                    "path": list(path), "field": name, "captured": old[name], "live": new[name],
                })
    old_other = {(item["kind"], tuple(item["path"])): item for item in before["other_entries"]}
    new_other = {(item["kind"], tuple(item["path"])): item for item in after["other_entries"]}
    for key in sorted(old_other.keys() | new_other.keys(), key=repr):
        old, new = old_other.get(key), new_other.get(key)
        if old != new:
            ignored_metadata_differences.append({
                "path": list(key[1]), "field": key[0], "captured": old, "live": new,
            })
    supported = before["supported"] and after["supported"]
    return {
        "match": supported and paths_match and not identity_differences,
        "supported": supported, "tensor_paths_match": paths_match,
        "raw_tensor_paths_match": raw_paths_match,
        "captured_tensors": before["tensors"], "live_tensors": after["tensors"],
        "captured_other_entries": before["other_entries"], "live_other_entries": after["other_entries"],
        "identity_differences": identity_differences,
        "ignored_metadata_differences": ignored_metadata_differences,
        "ignored_uncaptured_tensors": ignored_uncaptured_tensors,
        "unused_inputs_embeds_policy": optional_policy,
        "comparison_contract": "same captured NPU Tensor paths, view/base addresses, dtype/device and storage capacity",
    }


def _h_stream_signature(stream) -> dict | None:
    if stream is None:
        return None
    return {
        "device": str(getattr(stream, "device", None)),
        "npu_stream": getattr(stream, "npu_stream", None),
        "stream_id": getattr(stream, "stream_id", None),
    }


def _is_stream_resource_capture_error(exc: RuntimeError) -> bool:
    message = str(exc)
    lowered_message = message.lower()
    has_error_code = _STREAM_RESOURCE_ERROR_CODE in message
    has_stream_resource_marker = any(marker in lowered_message for marker in _STREAM_RESOURCE_ERROR_MARKERS)
    return has_stream_resource_marker or (has_error_code and "stream resource" in lowered_message)


def _is_old_hdk_capture_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _OLD_HDK_CAPTURE_ERROR_MARKERS)


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: torch.npu.NPUGraph | None = None
    output: Any | None = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None
    h_input_signature: tuple | None = None
    h_last_replay_stream: Any | None = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
        update_stream: torch.npu.Stream | None = None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode

        # A NONE runtime mode does not use ACL Graph, so avoid initializing
        # graph-specific state (including the Ascend config lookup).
        assert self.runtime_mode != CUDAGraphMode.NONE

        self.compilation_config = vllm_config.compilation_config
        self.enable_super_kernel = get_ascend_config().ascend_compilation_config.enable_super_kernel

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.enable_enpu = enable_enpu
        self.use_eagle = use_eagle
        self.update_stream = update_stream
        self.attn_backend = None
        self.draft_model_metadata: list[dict[str, Any]] = []
        self._h_native_queue_enabled = ascend_envs.VLLM_ASCEND_EARLY_DECODE_QUEUE
        self.h_native_queue_stats = {"bypassed": 0, "retained": 0}
        self.h_native_queue_stats["unused_inputs_embeds_ignored"] = 0
        self.h_native_queue_last_guard: dict | None = None
        _acl_graph_wrappers.add(self)

    def set_update_stream(self, update_stream):
        self.update_stream = update_stream

    def set_attn_backend(self, attn_backend):
        self.attn_backend = attn_backend

    def update_draft_model_metadata(self, draft_model_metadata: list[dict[str, Any]]):
        # This has been prepared for the update full graph of MRV1.
        self.draft_model_metadata = draft_model_metadata

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of aclgraph wrapper: {self._runnable_str}"
            )
        raise AttributeError(f"Attribute {key} not found. Set VLLM_LOGGING_LEVEL=DEBUG for more details.")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _h_native_queue_entry_diagnostics(self, entry: ACLGraphEntry) -> dict:
        """Inspect captured graph tasks and pointer metadata; never query NPU work."""
        graph = entry.aclgraph
        native = type(graph) is torch.npu.NPUGraph
        updatable = type(graph) is UpdatableGraph
        tasks = getattr(graph, "tasks", None) if updatable else [] if native else None
        providers = getattr(graph, "provider_sizes", None) if updatable else {} if native else None
        bucket = entry.batch_descriptor.num_tokens
        counts = {
            "tasks": len(tasks) if type(tasks) is list else None,
            "providers": len(providers) if type(providers) is dict else None,
        }
        for name in ("events", "handles", "attn_params"):
            table = getattr(_graph_params, name, None)
            values = table.get(bucket) if type(table) is dict else None
            counts[f"legacy_{name}"] = len(values) if type(values) is list else None
        signature = entry.h_input_signature
        captured_inputs = _h_native_input_metadata(signature)
        captured_identity = _h_native_input_diagnostics(signature, signature)
        tensor_count = len(captured_inputs["tensors"])
        checks = {
            "gate_enabled": self._h_native_queue_enabled,
            "full_wrapper": self.runtime_mode == CUDAGraphMode.FULL,
            "enpu_disabled": not self.enable_enpu,
            "exact_native_or_taskless_updatable": (native or updatable) and counts["tasks"] == 0
            and counts["providers"] == 0,
            "empty_legacy_events": counts["legacy_events"] == 0,
            "empty_legacy_handles": counts["legacy_handles"] == 0,
            "empty_legacy_attn_params": counts["legacy_attn_params"] == 0,
            "captured_tensor_inputs": captured_identity["match"] and tensor_count > 0,
        }
        return {
            "ready": all(checks.values()),
            "checks": checks,
            "counts": counts,
            "graph_type": f"{type(graph).__module__}.{type(graph).__qualname__}",
            "use_updatable_graph": use_updatable_graph(self.attn_backend),
            "input_tensor_count": tensor_count,
            "captured_inputs": captured_inputs,
        }

    def _h_can_replay_without_barrier(self, forward_context, entry: ACLGraphEntry, args, kwargs) -> bool:
        report = self._h_native_queue_entry_diagnostics(entry)
        checks = dict(report["checks"])
        current_stream = torch.npu.current_stream()
        authorized_stream = getattr(forward_context, "_h_native_queue_stream", None)
        signature = _h_native_input_signature(args, kwargs)
        optional_proof = getattr(forward_context, "_h_native_queue_optional_inputs_embeds_proof", None)
        if optional_proof is not None:
            from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForConditionalGeneration

            optional_proof = dict(optional_proof)
            optional_proof["exact_qwen_raw_runnable"] = type(self.runnable) is AscendQwen4ExpForConditionalGeneration
            optional_proof["runnable_type"] = f"{type(self.runnable).__module__}.{type(self.runnable).__qualname__}"
            optional_proof["strict_text_forward_authorized"] = (
                getattr(forward_context, "_h_native_queue_authorized", False) is True
            )
        input_diagnostics = _h_native_input_diagnostics(entry.h_input_signature, signature, optional_proof)
        signature_matches = input_diagnostics["match"]
        checks.update({
            "strict_forward_authorized": getattr(forward_context, "_h_native_queue_authorized", False) is True,
            "main_model": not _EXTRA_CTX.is_draft_model and not getattr(forward_context, "is_draft_model", True),
            "not_capturing": not getattr(forward_context, "capturing", True),
            "full_forward": forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL,
            "no_device_metadata_executor": getattr(forward_context, "device_metadata_executor", None) is None,
            "same_authorized_stream": authorized_stream is not None and authorized_stream == current_stream,
            # The first replay keeps the original barrier. Once a replay has
            # established stream ownership, subsequent H inputs/replays are
            # ordered on that same stream, after the previous MTP graph.
            "same_previous_replay_stream": entry.h_last_replay_stream is not None
            and entry.h_last_replay_stream == current_stream,
            "input_signature_match": signature_matches,
        })
        failed = [name for name, passed in checks.items() if not passed]
        bypass = not failed
        report.update({
            "ready": bypass,
            "checks": checks,
            "failed": failed,
            "input_signature_match": signature_matches,
            "input_signature_diagnostics": input_diagnostics,
            "authorized_stream": _h_stream_signature(authorized_stream),
            "current_stream": _h_stream_signature(current_stream),
            "previous_replay_stream": _h_stream_signature(entry.h_last_replay_stream),
        })
        self.h_native_queue_last_guard = report
        self.h_native_queue_stats["bypassed" if bypass else "retained"] += 1
        optional_ignored = bypass and bool(input_diagnostics["ignored_uncaptured_tensors"])
        if optional_ignored:
            self.h_native_queue_stats["unused_inputs_embeds_ignored"] += 1
        for name in failed:
            key = f"fallback_reason:{name}"
            self.h_native_queue_stats[key] = self.h_native_queue_stats.get(key, 0) + 1
        runner_stats = getattr(forward_context, "_h_native_queue_runner_stats", None)
        if runner_stats is not None:
            key = "native_barrier_bypassed" if bypass else "native_barrier_retained"
            runner_stats[key] += 1
            if optional_ignored:
                runner_stats["unused_inputs_embeds_ignored"] += 1
        observation = getattr(forward_context, "_h_native_queue_observation", None)
        if observation is not None:
            observation["last_guard"] = report
            if bypass:
                observation["last_bypass"] = report
        return bypass

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing ACL graph (%s, %s)", self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            entry.input_addresses = input_addresses
            if self._h_native_queue_enabled:
                entry.h_input_signature = _h_native_input_signature(args, kwargs)
            aclgraph = UpdatableGraph()

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.

                # Sync offloader's copy stream before capture.
                # Ensure any pre-capture prefetches from offloader are complete.
                from vllm.model_executor.offloader.base import get_offloader

                get_offloader().sync_prev_onload()
                forward_context.capturing = True
                try:
                    with torch.npu.graph(aclgraph, pool=self.graph_pool):
                        # `output` is managed by pytorch's aclgraph pool
                        with super_kernel_scope("full_model", self.enable_super_kernel):
                            output = self.runnable(*args, **kwargs)
                        # Join offloader's copy stream after forward to avoid
                        # unjoined stream error. The last layer's start_prefetch
                        # forks copy_stream, but wait_prefetch only happens in
                        # the next forward pass.
                        get_offloader().join_after_forward()
                        if self.aclgraph_options.weak_ref_output:
                            # by converting it to weak ref,
                            # the original `output` will immediately be released
                            # to save memory. It is only safe to do this for
                            # the last graph in piecewise aclgraph mode, because
                            # the output of the last graph will not be used by
                            # any other acl graph.
                            output = weak_ref_tensors(output)
                except RuntimeError as exc:
                    if _is_old_hdk_capture_error(exc):
                        raise RuntimeError(
                            "ACL graph capture failed with an old Ascend HDK/CANN stack "
                            "signature (`Alloc sq cq fail`). Please upgrade Ascend HDK to "
                            "25.5.1 or later and use the matching CANN stack.\n"
                            f"Original error:\n{exc}"
                        ) from exc
                    elif _is_stream_resource_capture_error(exc):
                        raise RuntimeError(
                            "ACL graph capture failed with a known stream-resource exhaustion "
                            "signature. Consider reducing cudagraph_capture_sizes, lowering "
                            "max_cudagraph_capture_size, preferring FULL or FULL_DECODE_ONLY for "
                            "mostly uniform decode workloads, or temporarily disabling graph mode "
                            "to confirm the failure is capture-related.\n"
                            f"Original error:\n{exc}"
                        ) from exc
                    raise

            if self.enable_super_kernel:
                aclgraph.super_kernel_optimize(
                    optimize_options={
                        "dcci_after_kernel_end": [".*"],
                    },
                )
                logger.info_once("Super kernel optimization is enabled for ACL graph capture.")

            # here we always use weak ref for the workspaces
            # to save memory
            global _graph_params
            global _draft_graph_params
            global _draft_graph_prefill_params
            weak_ref_workspaces(_graph_params)
            weak_ref_workspaces(_draft_graph_params)
            weak_ref_workspaces(_draft_graph_prefill_params)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        # If we do not in main model and in full-graph mode when using merge-eagle-graph,
        # we do not need to synchronize.
        # When enable_enpu is on, model_runner orders update vs replay; skip here.
        # When FULL + EAGLE draft (merge path), replay does not need this barrier.
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        if not self.enable_enpu and need_sync:
            bypass = self._h_native_queue_enabled and self._h_can_replay_without_barrier(
                forward_context, entry, args, kwargs
            )
            if bypass:
                with torch.profiler.record_function("qsa_async/H_native_barrier_bypass"):
                    pass
            else:
                torch.npu.current_stream().synchronize()
        if self.runtime_mode == CUDAGraphMode.FULL and use_updatable_graph(self.attn_backend):
            self._updatable_graph_replay(forward_context, entry.aclgraph)
        else:
            entry.aclgraph.replay()
        if self._h_native_queue_enabled:
            entry.h_last_replay_stream = torch.npu.current_stream()
        logger.info_once("ACL graph replay is active (logged once).")
        return entry.output

    def _updatable_graph_replay(
        self,
        forward_context,
        graph: UpdatableGraph,
    ):
        assert self.update_stream is not None
        if _EXTRA_CTX.is_draft_model:
            resolved_tasks = graph.resolve_tasks(SharedSource(self.draft_model_metadata))
        else:
            resolved_tasks = graph.resolve_tasks(ContextSource(forward_context.attn_metadata))
        if self.enable_enpu:
            graph.update(self.update_stream, resolved_tasks)
            graph.replay()
        else:
            graph.replay()
            graph.update(self.update_stream, resolved_tasks)


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(params.workspaces[num_tokens])


def update_full_graph_params(
    attn_backend,
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    draft_attn_metadatas=None,
):
    if use_updatable_graph(attn_backend):
        return

    # vLLM >= 0.27.1 (main) makes get_current_vllm_config() raise
    # AssertionError outside set_current_vllm_config(); the SFA backend
    # resolution in get_impl_cls() needs the config.
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        impl_cls = attn_backend.get_impl_cls()
        impl_cls.update_graph_params(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            draft_attn_metadatas=draft_attn_metadatas,
        )


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]


_graph_params: GraphParams | None = None


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace


def get_graph_params():
    return _graph_params


_draft_graph_params: GraphParams | None = None


def set_draft_graph_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params():
    return _draft_graph_params


_draft_graph_prefill_params: GraphParams | None = None


def set_draft_graph_prefill_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph preill parameters have already been set!")
    _draft_graph_prefill_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_prefill_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        _draft_graph_prefill_params.workspaces[num_tokens] = workspace


def get_draft_graph_prefill_params():
    return _draft_graph_prefill_params
