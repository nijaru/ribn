#!/usr/bin/env python3
"""Pinned SGLang Gemma E4B one-batch diagnostic, not serving qualification.

Run in an isolated Linux/CUDA SGLang 0.5.22 environment. --validate-only needs
only Python 3.13 and reads the reference without importing Torch or SGLang.
"""

import argparse
import dataclasses
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import tomllib
from array import array
from pathlib import Path

from gemma4_reference import (
    MANIFEST,
    require,
    sha256,
    verify_files,
    verify_reference_sources,
    write_json,
)

UPSTREAM = "baa9befa381511cd4d56f83980951e5a45fffc6d"
SOURCE_HASHES = {
    "sglang/benchmark/one_batch.py": "f98a8a74300544792d339885ed82d3c8b8d7df6d27534d8bd1b48148cf7669de",
    "sglang/srt/managers/schedule_batch.py": "c382bf0066d70e19b4b9e38c92ff0029d39996238f8f0685f4c1dadbfa4e8a96",
    "sglang/srt/models/gemma4_mm.py": "7af9d68a58db47e8a02c036a1a03079723c734fe55c93154b0c56a5db5f1fe95",
    "sglang/srt/models/gemma4_causal.py": "f506b5e0f1c3976693a60892fdd7c3f51576e27a657819fb5140b905ca0bc6cc",
    "sglang/srt/layers/logits_processor.py": "5643e7542ea6540b1f95f0e6c9aaa1b6f8eb1ea49dc8075d6dd327839ff8aa82",
}
PACKAGES = {
    "sglang": "0.5.22",
    "sglang-kernel": "0.4.9",
    "flashinfer-python": "0.7.1",
    "torch": "2.14.1",
    "transformers": "5.19.0",
    "tokenizers": "0.23.2",
}
VOCAB = 262144
CAPACITY = 4096
LENGTHS = (1, 8, 18, 32, 64, 65, 128)


def token_ids(value):
    require(
        isinstance(value, list)
        and bool(value)
        and all(type(token) is int and 0 <= token < VOCAB for token in value),
        "expected nonempty in-vocabulary integer token IDs",
    )
    return list(value)


def reference_cases(reference, manifest):
    """Reject broken histories before loading a model; never follow new predictions."""
    require(reference["complete"] is True, "incomplete reference")
    require(reference["checkpoint"] == manifest["checkpoint"], "wrong checkpoint")
    require(
        reference["precision"] == "bf16"
        and reference["device"] == "cpu"
        and reference["attention"] == "eager",
        "expected CPU eager BF16 reference",
    )
    cases = reference["cases"]
    require(
        [case["name"] for case in cases] == ["chat", "window_511", "window_513"],
        "expected the three archived reference cases in order",
    )
    result = []
    for case, length in zip(cases, (18, 511, 513), strict=True):
        prompt = token_ids(case["prompt_ids"])
        require(len(prompt) == length, "wrong reference prompt length")
        require(len(case["steps"]) == 4, "expected prefill and three continuations")
        history = []
        inputs = []
        for index, step in enumerate(case["steps"]):
            require(step["index"] == index, "wrong step index")
            current = token_ids(step["input_ids"])
            require(
                current == prompt if index == 0 else len(current) == 1,
                "wrong input extent",
            )
            history.extend(current)
            require(
                history == token_ids(step["history_ids"]),
                "discontinuous reference history",
            )
            require(
                step["position_ids"]
                == list(range(len(history) - len(current), len(history))),
                "wrong positions",
            )
            require(len(history) <= CAPACITY, "reference exceeds prepared reach")
            token_ids([step["chosen"]])
            inputs.append(current)
        result.append({"name": case["name"], "inputs": inputs})
    return result


def timing_cases(cases):
    chat = cases[0]
    prompt = chat["inputs"][0]
    # The old native/stock sweep used the first three archived chat winners,
    # which are exactly the three consumed continuation inputs, not the last winner.
    continuation = chat["inputs"][1:]
    return [
        {
            "name": f"timing_{length}",
            "inputs": [[prompt[i % len(prompt)] for i in range(length)]]
            + [list(ids) for ids in continuation],
        }
        for length in LENGTHS
    ]


def check_installation():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == PACKAGES, f"wrong engine environment: {versions}")
    spec = importlib.util.find_spec("sglang")
    require(spec is not None and spec.origin is not None, "SGLang source not found")
    verify_reference_sources(Path(spec.origin).parent.parent, SOURCE_HASHES)
    return {
        item.metadata["Name"]: item.version
        for item in importlib.metadata.distributions()
    }


def load_runner(snapshot, eager):
    import torch
    from sglang.benchmark import one_batch as upstream
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank

    require(torch.cuda.is_available(), "CUDA device required")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    args = upstream.ServerArgs(
        model_path=str(snapshot.resolve()),
        dtype="bfloat16",
        model_impl="sglang",
        trust_remote_code=False,
        tp_size=1,
        context_length=CAPACITY,
        max_total_tokens=CAPACITY,
        max_running_requests=1,
        mem_fraction_static=0.85,
        attention_backend="triton",
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        cuda_graph_config={
            "decode": {
                "backend": "disabled" if eager else "full",
                "bs": [1],
                "max_bs": 1,
            },
            "prefill": {"backend": "disabled"},
        },
    )
    args.resolve_once()
    upstream._set_envs_and_config(args)
    ports = upstream.PortArgs.init_new(args)
    publish(
        args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(args, tp_rank=0, pp_rank=0),
            gpu_id=0,
        ),
    )
    upstream._init_process_global_configs()
    wrapped, _tokenizer = upstream.load_model(args, ports, 0, 0)
    runner = wrapped.torch_runner
    require(
        type(runner.model).__name__ == "Gemma4ForConditionalGeneration",
        "wrong model implementation",
    )
    require(runner.model.quant_config is None, "unexpected quantization")
    require(
        runner.max_total_num_tokens >= 516, "insufficient pool for reference histories"
    )
    require(
        runner.model.lm_head.weight.data_ptr()
        == runner.model.language_model.embed_tokens.weight.data_ptr(),
        "untied head",
    )
    cfg = resolving_view(args)
    settings = {
        key: getattr(cfg, key)
        for key in (
            "dtype",
            "attention_backend",
            "context_length",
            "max_total_tokens",
            "max_running_requests",
            "mem_fraction_static",
            "page_size",
            "enable_torch_compile",
        )
    }
    settings["cuda_graph_config"] = dataclasses.asdict(cfg.cuda_graph_config)
    settings["max_total_num_tokens"] = runner.max_total_num_tokens
    settings["model_class"] = type(runner.model).__name__
    settings["decode_graph_runner"] = (
        type(runner.decode_cuda_graph_runner).__name__
        if runner.decode_cuda_graph_runner is not None
        else None
    )
    settings["numerical_flags"] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }
    return torch, upstream, wrapped, settings


def forward(torch, upstream, runner, ids, batch):
    """Use the pinned one_batch preparation path, omitting its sampler.

    Sampler output must neither choose the history nor mutate captured logits.
    All mutable batch/cache state stays with this single sequential runner.
    """
    if batch is None:
        req = upstream.Req(
            rid="ribn-baseline",
            origin_input_text="",
            origin_input_ids=array("q", ids),
            sampling_params=upstream.SamplingParams(
                temperature=0, max_new_tokens=4, ignore_eos=True
            ),
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.extend_end = len(ids)
        cache = upstream.TreeCacheNamespace(
            page_size=upstream.get_schedule().page_size,
            device=runner.device,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        )
        batch = upstream.ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=cache,
            model_config=runner.model_config,
            enable_overlap=False,
            spec_algorithm=upstream.SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
    else:
        req = batch.reqs[0]
        req.output_ids.extend(ids)
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        req.extend_end = len(req.full_untruncated_fill_ids)
        batch.input_ids = torch.tensor(ids, dtype=torch.int64, device=runner.device)
        batch.prepare_for_decode()
    upstream._maybe_prepare_mlp_sync_batch(batch, runner)
    if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(
            batch.device, non_blocking=True
        )
        batch.prefill_input_ids_cpu = None
    forward_batch = upstream.ForwardBatch.init_new(
        batch, runner, return_hidden_states_before_norm=False
    )
    result = runner.forward(forward_batch)
    return result, batch, forward_batch


def sequence(torch, upstream, wrapped, case, output, report, trial=None):
    run = {
        "name": case["name"],
        "trial": trial,
        "warmup": trial == 0,
        "complete": False,
        "steps": [],
    }
    report["runs"].append(run)
    # clear() is legal only after the preceding sequence has physically completed.
    wrapped.synchronize()
    wrapped.clear()
    batch = None
    history = []
    for index, ids in enumerate(case["inputs"]):
        start = time.perf_counter_ns()
        result, batch, forward_batch = forward(
            torch, upstream, wrapped.torch_runner, ids, batch
        )
        # Preserve the engine's FP32 softcap, not a BF16 rounding.
        host = result.logits_output.next_token_logits.cpu()
        wrapped.synchronize()
        elapsed = time.perf_counter_ns() - start
        history.extend(ids)
        record = {
            "step": index,
            "input_ids": list(ids),
            "history_length": len(history),
            "elapsed_ns": elapsed,
            "engine_can_run_graph": result.can_run_graph,
            "shape": list(host.shape),
            "dtype": str(host.dtype),
        }
        # Keep completed samples even if a later step or validation fails.
        run["steps"].append(record)
        require(tuple(host.shape) == (1, VOCAB), "wrong full-vocabulary shape")
        require(host.dtype == torch.float32, "expected native FP32 engine logits")
        record["finite"] = bool(torch.isfinite(host).all().item())
        if trial is None or not record["finite"]:
            suffix = "" if trial is None else f".trial{trial}"
            path = output / f"{case['name']}{suffix}.{index}.f32le"
            path.write_bytes(host.numpy().astype("<f4", copy=False).tobytes())
            record["logits"] = {
                "path": path.name,
                "sha256": sha256(path),
                "dtype": "float32-le",
                "elements": VOCAB,
            }
        require(record["finite"], "nonfinite logits")
        record["chosen"] = int(host[0].argmax())
        record["position_ids"] = forward_batch.positions.cpu().tolist()
        require(
            record["position_ids"]
            == list(range(len(history) - len(ids), len(history))),
            "forward positions differ from the common history",
        )
        require(
            batch.seq_lens_cpu.tolist() == [len(history)],
            "cache position did not advance",
        )
        write_json(output / "report.json", report)
    wrapped.cleanup(batch)
    run["complete"] = True
    write_json(output / "report.json", report)


def collect(snapshot, reference_path, output, validate_only=False, eager=False):
    require(sys.version_info[:2] == (3, 13), "use Python 3.13")
    require(
        not output.resolve().is_relative_to(snapshot.resolve()),
        "keep evidence outside package",
    )
    output.mkdir(parents=True)  # Never overwrite evidence, including failed runs.
    report = {
        "complete": False,
        "upstream_commit": UPSTREAM,
        "script_sha256": sha256(Path(__file__)),
        "clock": "input preparation through completed full FP32 host logits; no sampling",
        "scope": "bs1 one-batch diagnostic, not HTTP/SLO/capacity or task-quality acceptance",
        "validation_only": validate_only,
        "python": sys.version,
        "platform": platform.platform(),
        "runs": [],
    }
    try:
        report["reference_sha256"] = sha256(reference_path)
        report["manifest_sha256"] = sha256(MANIFEST)
        manifest = tomllib.loads(MANIFEST.read_text())
        reference = json.loads(reference_path.read_text())
        cases = reference_cases(reference, manifest)
        report["checkpoint"] = manifest["checkpoint"]
        report["plan"] = {
            "quality": cases,
            "timing": timing_cases(cases),
            "trials": 4,
            "warmup_trial": 0,
        }
        if validate_only:
            report["validated_inputs"] = True
            return
        report["packages"] = check_installation()
        report["files"] = verify_files(snapshot, manifest["files"])
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[key] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        started = time.perf_counter_ns()
        torch, upstream, wrapped, report["settings"] = load_runner(snapshot, eager)
        wrapped.synchronize()
        report["prepare_ns"] = time.perf_counter_ns() - started
        report["device"] = torch.cuda.get_device_name(0)
        report["torch_cuda"] = torch.version.cuda
        with torch.inference_mode():
            for case in cases:
                sequence(torch, upstream, wrapped, case, output, report)
            for case in timing_cases(cases):
                for trial in range(4):
                    sequence(torch, upstream, wrapped, case, output, report, trial)
            wrapped.synchronize()
        report["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        report["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["complete"] = True
    except Exception as error:
        # Never clear/reuse pools after a failed forward or failed synchronization.
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="disable decode graphs too; prefill graphs are always disabled",
    )
    args = parser.parse_args()
    collect(args.snapshot, args.reference, args.output, args.validate_only, args.eager)


if __name__ == "__main__":
    main()
