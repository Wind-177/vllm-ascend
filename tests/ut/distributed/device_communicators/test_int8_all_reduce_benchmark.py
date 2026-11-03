# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the standalone benchmark; no vLLM/NPU imports required."""

import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_SCRIPT = Path(__file__).resolve().parents[4] / "benchmarks/benchmark_int8_all_reduce.py"
_SPEC = importlib.util.spec_from_file_location("int8_all_reduce_benchmark", _SCRIPT)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


class TestInt8Benchmark(unittest.TestCase):
    def test_default_and_invalid_arguments(self):
        args = bench._parse_args([])
        self.assertEqual((args.quant_backend, args.dequant_backend), ("npu", "npu"))
        self.assertEqual(args.hidden_sizes, [2048, 5120])
        for argv in (["--tokens", "0"], ["--hidden-sizes", "-1"], ["--iterations", "0"], ["--rounds", "0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                bench._parse_args(argv)
        for size in (1, 3, 16):
            with self.assertRaises(ValueError):
                bench.Int8AllReduce(size, bench.QuantizationOps("torch", "torch"))

    def test_native_operator_signatures_and_missing_operators(self):
        backend = SimpleNamespace(npu_quantize=MagicMock(), npu_anti_quant=MagicMock())
        ops = bench.QuantizationOps("npu", "npu", backend)
        x = torch.ones(4, 64, dtype=torch.bfloat16)
        scale = torch.ones(64, dtype=torch.float32)
        ops.quantize(x, scale)
        backend.npu_quantize.assert_called_once_with(x, scale, None, torch.int8, 1, True)
        ops.dequantize(x, scale)
        backend.npu_anti_quant.assert_called_once_with(x, scale, dst_dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "npu_quantize is unavailable"):
            bench.QuantizationOps("npu", "torch")
        with self.assertRaisesRegex(RuntimeError, "npu_anti_quant is unavailable"):
            bench.QuantizationOps("torch", "npu")

    def test_int8_bounds_and_zero_channels_on_two_four_eight_ranks(self):
        for world_size in (2, 4, 8):
            for value in (0.0, 1.0):
                for method in ("abs", "norm"):
                    pipeline = bench.Int8AllReduce(world_size, bench.QuantizationOps("torch", "torch"), method)

                    def collective(tensor, op=None, group=None, size=world_size):
                        if op != bench.dist.ReduceOp.MAX:
                            tensor.mul_(size)

                    with (
                        self.subTest(world_size=world_size, value=value, method=method),
                        patch.object(bench.dist, "all_reduce", side_effect=collective),
                    ):
                        source = torch.full((4, 64), value, dtype=torch.bfloat16)
                        result = bench._check(pipeline, source)
                        self.assertEqual(result["int8_relative_l2_vs_fp32"], 0)

    def test_bad_quantization_range_fails_correctness_check(self):
        ops = bench.QuantizationOps("torch", "torch")
        ops.quantize = lambda x, scale: torch.full_like(x, 32, dtype=torch.int8)
        pipeline = bench.Int8AllReduce(4, ops)

        def collective(tensor, op=None, group=None):
            if op != bench.dist.ReduceOp.MAX:
                tensor.mul_(4)

        with (
            patch.object(bench.dist, "all_reduce", side_effect=collective),
            self.assertRaisesRegex(AssertionError, "dtype/range/INT8 SUM"),
        ):
            bench._check(pipeline, torch.ones(4, 64, dtype=torch.bfloat16))

    def test_distinct_rank_inputs_share_channel_scales(self):
        for world_size in (2, 4, 8):
            with self.subTest(world_size=world_size):
                self._run_ranks(world_size)

    def _run_ranks(self, world_size):
        inputs = []
        for rank in range(world_size):
            generator = torch.Generator().manual_seed(rank)
            x = (torch.randn(5, 64, generator=generator) * (rank + 1)).to(torch.bfloat16)
            x[:, 0] = 0
            inputs.append(x)
        max_q = 127 // world_size
        global_amax = torch.stack([x.abs().amax(0).float() for x in inputs]).amax(0)
        scale = (global_amax / max_q).clamp_min(1e-8)
        qs = [(x.float() / scale).round().to(torch.int32) for x in inputs]
        expected = (torch.stack(qs).sum(0).float() * scale).to(torch.bfloat16)
        state = threading.local()
        slots = [None] * world_size

        def complete():
            tensors = torch.stack([slot[1] for slot in slots])
            if slots[0][2] == bench.dist.ReduceOp.MAX:
                result = tensors.amax(0)
            else:
                result = tensors.to(torch.int32).sum(0).to(tensors.dtype)
            for tensor, _, _ in slots:
                tensor.copy_(result)

        barrier = threading.Barrier(world_size, action=complete, timeout=15)

        def collective(tensor, op=None, group=None):
            slots[state.rank] = (tensor, tensor.clone(), op)
            barrier.wait()

        def rank_worker(rank):
            state.rank = rank
            x = inputs[rank].clone()
            pipeline = bench.Int8AllReduce(world_size, bench.QuantizationOps("torch", "torch"))
            self.assertIs(pipeline(x), x)
            torch.testing.assert_close(x, expected, rtol=0, atol=0)

        with (
            patch.object(bench.dist, "all_reduce", side_effect=collective),
            ThreadPoolExecutor(max_workers=world_size) as pool,
        ):
            list(pool.map(rank_worker, range(world_size)))

    def test_measurement_resets_input_and_excludes_it_from_events(self):
        clock = SimpleNamespace(value=0)

        class Event:
            def __init__(self, enable_timing):
                self.timestamp = None

            def record(self):
                self.timestamp = clock.value

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        source = torch.ones(4, 64, dtype=torch.bfloat16)
        calls = []

        def operation(working):
            torch.testing.assert_close(working, source, rtol=0, atol=0)
            calls.append(1)
            working.mul_(4)
            clock.value += 3

        fake_npu = SimpleNamespace(
            Event=Event,
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            memory_allocated=lambda: 100,
            max_memory_allocated=lambda: 100 + 1024 * 1024,
        )
        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(bench.dist, "barrier"),
            patch.object(bench, "_max_across_ranks", side_effect=lambda values, device: values),
        ):
            result = bench._measure(operation, source, iterations=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["samples_ms"], [3, 3, 3])
        self.assertEqual(result["extra_peak_allocated_mib"], 1)

    def test_alternating_order_and_full_path_speedup(self):
        calls = []
        pipeline = MagicMock()

        def measure(op, source, iterations):
            kind = "int8" if op is pipeline else "bf16"
            calls.append(kind)
            return {
                "samples_ms": [2 if kind == "int8" else 4] * iterations,
                "wall_ms_including_reset": 5,
                "extra_peak_allocated_mib": 1,
            }

        with (
            patch.object(bench, "_check", return_value={}),
            patch.object(bench, "_measure", side_effect=measure),
            patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True),
        ):
            args = bench._parse_args(["--warmup", "0", "--rounds", "2", "--iterations", "3"])
            result = bench._benchmark_case(pipeline, torch.ones(4, 64, dtype=torch.bfloat16), args)
        self.assertEqual(calls, ["bf16", "int8", "int8", "bf16"])
        self.assertEqual(result["speedup"], 2)
        self.assertEqual(result["latency_reduction_pct"], 50)

    def test_percentiles(self):
        summary = bench.summarize(list(range(1, 21)))
        self.assertEqual(summary["median_ms"], 10.5)
        self.assertEqual(summary["p95_ms"], 19)
        with self.assertRaises(ValueError):
            bench.summarize([])
