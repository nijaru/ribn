#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.14.0", "transformers==5.20.0", "safetensors==0.8.0", "tokenizers==0.23.3"]
# ///
"""Collect small CPU rotary-frequency fixtures without loading checkpoint weights.

Uses the same model/config/source pins as gemma4_reference.py. Output must be a new
JSON file. This reference does not establish device RoPE or whole-model parity.
"""

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import sys
import tomllib
from pathlib import Path

from gemma4_reference import MANIFEST, require, sha256, verify_reference_sources


def collect(output):
    require(not output.exists(), "output already exists; preserve earlier evidence")
    manifest = tomllib.loads(MANIFEST.read_text())
    config_path = MANIFEST.with_name("config.json")
    require(
        sha256(config_path) == manifest["files"]["config.json"]["sha256"],
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
    import torch
    import transformers
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    verified = verify_reference_sources(
        Path(transformers.__file__).parent, manifest["reference"]["source_sha256"]
    )
    torch.set_num_threads(1)
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
        config = copy.deepcopy(json.loads(config_path.read_text())["text_config"])
        config.update(overrides)
        config = Gemma4TextConfig(**config)
        rotary = Gemma4TextRotaryEmbedding(config, device="cpu")
        kinds = {}
        for kind in ("sliding_attention", "full_attention"):
            values = getattr(rotary, f"{kind}_inv_freq")
            require(
                values.device.type == "cpu"
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
                    int(v) & 0xFFFFFFFF for v in values.view(torch.int32)
                ],
            }
        cases.append({"name": name, "text_overrides": overrides, "kinds": kinds})
    report = {
        "schema_version": 1,
        "scope": "CPU FP32 inverse frequencies only; no device or whole-model qualification",
        "declared_checkpoint": manifest["checkpoint"],
        "verified_checkpoint_files": ["config.json"],
        "config_sha256": sha256(config_path),
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
    args = parser.parse_args()
    try:
        collect(args.output)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"rotary reference collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
