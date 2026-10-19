#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.14.0", "transformers==5.20.0", "safetensors==0.8.0", "tokenizers==0.23.3"]
# ///
"""Collect rotary frequencies or stock CUDA BF16 rotation operands, without weights.

Uses the same model/config/source pins as gemma4_reference.py. Output must be a new
JSON file. CUDA mode preserves CPU FP32 frequency preparation, then records eager
BF16 factors and application. Neither mode qualifies native phase or whole models.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tomllib
from pathlib import Path

from gemma4_reference import MANIFEST, require, sha256, verify_reference_sources


def apply_case(torch, config, rotary, kind, heads):
    """Stock CUDA eager application; row-major storage matches the native gate."""
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

    width = config.per_layer_config[kind].head_dim
    positions = [
        4095,
        0,
        511,
        config.max_position_embeddings - 1,
        512,
        513,
        config.max_position_embeddings // 2 - 1,
        4095,
        1,
    ]
    require(max(positions) < config.max_position_embeddings, "positions outside config")
    count = len(positions) * heads * width
    index = torch.arange(count, dtype=torch.int64)
    values = ((index * 17 + index // width * 3) % 65 - 32).float() / 16
    values *= torch.tensor([1.0, 1 / 256, 16.0])[index % 3]
    values[0], values[1] = -0.0, 0.0
    x = values.to(device="cuda", dtype=torch.bfloat16).view(
        1, len(positions), heads, width
    )
    cos, sin = rotary(x, torch.tensor([positions], device="cuda"), kind)
    query = x.transpose(1, 2)
    result = apply_rotary_pos_emb(query, cos, sin)
    # Counterexample protection: a single final BF16 rounding is not this contract.
    half = torch.cat((-query[..., width // 2 :], query[..., : width // 2]), dim=-1)
    fused = (
        query.float() * cos.unsqueeze(1).float()
        + half.float() * sin.unsqueeze(1).float()
    ).to(torch.bfloat16)
    different = int((result != fused).sum().item())
    require(different > 0, "fixture does not expose missing product rounding")

    def bits(value):
        require(
            value.dtype == torch.bfloat16 and value.isfinite().all().item(),
            "nonfinite/wrong dtype rotary operand",
        )
        return [
            int(v) & 0xFFFF
            for v in value.detach().cpu().contiguous().view(torch.int16).flatten()
        ]

    return {
        "heads": heads,
        "positions": positions,
        "input_bits": bits(x),
        "cos_bits": bits(cos),
        "sin_bits": bits(sin),
        "output_bits": bits(result.transpose(1, 2)),
        "single_rounding_mismatches": different,
    }


def collect(output, apply_cuda=False):
    require(not output.exists(), "output already exists; preserve earlier evidence")
    manifest = tomllib.loads(MANIFEST.read_text())
    config_path = MANIFEST.with_name("config.json")
    config_bytes = config_path.read_bytes()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    require(
        config_digest == manifest["files"]["config.json"]["sha256"],
        "config differs from the checkpoint pin",
    )
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "safetensors", "tokenizers")
    }
    for name, version in packages.items():
        require(version == manifest["reference"][name], f"use pinned {name}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if apply_cuda:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import transformers
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    verified = verify_reference_sources(
        Path(transformers.__file__).parent, manifest["reference"]["source_sha256"]
    )
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    cuda = None
    if apply_cuda:
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        cuda = {
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": False,
            "allow_fp16_reduced_precision_reduction": False,
        }
    variants = [
        ("e4b", {}),
        (
            "alternate_geometry",
            {
                "head_dim": 10,
                "global_head_dim": 16,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 32768.0,
                    },
                    "full_attention": {
                        "rope_type": "proportional",
                        "rope_theta": 1048576.0,
                        "partial_rotary_factor": 0.25,
                    },
                },
            },
        ),
    ]
    cases = []
    for name, overrides in variants:
        config = copy.deepcopy(json.loads(config_bytes)["text_config"])
        config.update(overrides)
        config = Gemma4TextConfig(**config)
        # Match the stock probe: CPU FP32 preparation, then move without a dtype cast.
        rotary = Gemma4TextRotaryEmbedding(config, device="cpu")
        if apply_cuda:
            rotary = rotary.to("cuda")
        kinds = {}
        for kind in ("sliding_attention", "full_attention"):
            values = getattr(rotary, f"{kind}_inv_freq")
            require(
                values.device.type == ("cuda" if apply_cuda else "cpu")
                and values.dtype == torch.float32
                and values.ndim == 1
                and values.isfinite().all().item(),
                "invalid rotary reference",
            )
            require(
                getattr(rotary, f"{kind}_attention_scaling") == 1.0,
                "unexpected scaling",
            )
            kinds[kind] = {
                "head_dim": config.per_layer_config[kind].head_dim,
                "inverse_frequency_bits": [
                    int(v) & 0xFFFFFFFF for v in values.cpu().view(torch.int32)
                ],
            }
            if apply_cuda:
                kinds[kind]["applications"] = [
                    apply_case(torch, config, rotary, kind, heads)
                    for heads in ([2, 8] if name == "e4b" else [1, 3])
                ]
        cases.append({"name": name, "text_overrides": overrides, "kinds": kinds})
    report = {
        "schema_version": 1,
        "kind": "cuda_bf16_rotary_apply" if apply_cuda else "cpu_rotary_frequencies",
        "scope": (
            "stock CUDA eager BF16 rotation with CPU-prepared FP32 frequencies; no native factor generation or whole-model qualification"
            if apply_cuda
            else "CPU FP32 inverse frequencies only; no device or whole-model qualification"
        ),
        "cuda": cuda,
        "deterministic_algorithms": True,
        "frequency_preparation": "CPU FP32, transfer without dtype conversion",
        "torch_build": torch.__config__.show(),
        "declared_checkpoint": manifest["checkpoint"],
        "verified_checkpoint_files": ["config.json"],
        "config_sha256": config_digest,
        "generator_sha256": sha256(Path(__file__)),
        "packages": packages,
        "python": sys.version,
        "platform": platform.platform(),
        "transformers_commit": manifest["reference"]["transformers_commit"],
        "verified_source_sha256": verified,
        "whole_installation_identity_verified": False,
        "cases": cases,
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--apply-cuda",
        action="store_true",
        help="also archive stock CUDA BF16 factors, inputs and split-half application",
    )
    args = parser.parse_args()
    try:
        collect(args.output, args.apply_cuda)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"rotary reference collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
