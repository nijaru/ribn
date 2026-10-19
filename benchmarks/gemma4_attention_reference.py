#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.14.0", "transformers==5.20.0", "safetensors==0.8.0", "tokenizers==0.23.3"]
# ///
"""Per-position attention operands from the pinned CPU model; no Ribn claims."""

import argparse
import platform
import sys
import tomllib
from pathlib import Path

from gemma4_reference import (
    MANIFEST,
    prepare_cpu_reference,
    require,
    sha256,
    verify_files,
    write_json,
)

PROMPT = "The quick brown fox jumps over the lazy dog."
LAYERS = (0, 5, 22, 23)


def collect(snapshot, output, threads):
    require(not output.exists(), "output already exists; preserve earlier evidence")
    output.mkdir(parents=True)
    manifest = tomllib.loads(MANIFEST.read_text())
    report = {
        "schema_version": 1,
        "kind": "cpu_bf16_attention_operands",
        "complete": False,
        "checkpoint": manifest["checkpoint"],
        "manifest_sha256": sha256(MANIFEST),
        "generator_sha256": sha256(Path(__file__)),
        "precision": "bf16",
        "device": "cpu",
        "threads": threads,
        "platform": platform.platform(),
        "python": sys.version,
        "scope": "one pinned CPU prompt with per-position attention operands; no Ribn or full-model acceptance",
        "layers": list(LAYERS),
        "prompt_text": PROMPT,
        "cases": [],
    }
    write_json(output / "reference.json", report)
    tensors = {}
    try:
        report["files"] = verify_files(snapshot, manifest["files"])
        torch = prepare_cpu_reference(report, manifest["reference"], threads)
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
        from transformers.models.gemma4 import modeling_gemma4
        from transformers.models.gemma4.modeling_gemma4 import repeat_kv

        model, loading = Gemma4ForConditionalGeneration.from_pretrained(
            snapshot,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        require(not any(loading.values()), "reference checkpoint did not load cleanly")
        model.eval()
        config = model.config.get_text_config()
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
        require(4 <= len(ids) <= 32, "reference prompt length is out of range")
        report["prompt_ids"] = ids

        captured = {}
        original_eager = modeling_gemma4.eager_attention_forward

        def capture(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=None,
            softcap=None,
            **kwargs,
        ):
            if module.layer_idx in LAYERS:
                require(
                    module.layer_idx not in captured,
                    "attention layer ran twice in one forward",
                )
                captured[module.layer_idx] = {
                    name: tensor.detach().clone()
                    for name, tensor in (
                        ("query", query),
                        ("key", key),
                        ("value", value),
                    )
                }
            output_states, weights = original_eager(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                softcap=softcap,
                **kwargs,
            )
            if module.layer_idx in captured:
                # The pinned expression before the additive mask: BF16 matmul,
                # scaling 1.0. Captured so an audit can derive score-quantization
                # bounds instead of assuming a fixed relative allowance.
                grouped = repeat_kv(key, module.num_key_value_groups)
                scores = torch.matmul(query, grouped.transpose(2, 3)) * (
                    scaling if scaling is not None else module.head_dim**-0.5
                )
                captured[module.layer_idx]["scores"] = scores.detach().clone()
                captured[module.layer_idx]["weights"] = weights.detach().clone()
            return output_states, weights

        # Keep the stock `eager` implementation selected so the model materializes
        # its real causal/sliding masks; wrap only the eager function itself.
        modeling_gemma4.eager_attention_forward = capture
        core = {}
        hooks = []
        for layer in LAYERS:
            projection = model.model.language_model.layers[layer].self_attn.o_proj

            def tap(_module, inputs, _output, name=layer):
                core[name] = inputs[0].detach().clone()

            hooks.append(projection.register_forward_hook(tap))
        try:
            with torch.inference_mode():
                model(
                    input_ids=torch.tensor([ids]),
                    use_cache=True,
                    return_dict=True,
                )
        finally:
            modeling_gemma4.eager_attention_forward = original_eager
            for hook in hooks:
                hook.remove()

        def bits(tensor, expected_shape):
            require(
                tuple(tensor.shape) == expected_shape
                and tensor.dtype == torch.bfloat16
                and tensor.isfinite().all().item(),
                f"unexpected attention operand {tuple(tensor.shape)} {tensor.dtype}",
            )
            return [
                int(value) & 0xFFFF
                for value in tensor.detach()
                .cpu()
                .contiguous()
                .view(torch.int16)
                .flatten()
            ]

        sequence = len(ids)
        for layer in LAYERS:
            geometry = config.per_layer_config[layer]
            width = geometry.head_dim
            heads = config.num_attention_heads
            kv = geometry.num_key_value_heads
            entry = captured[layer]
            query = entry["query"].transpose(1, 2).reshape(sequence, heads * width)
            key = entry["key"].transpose(1, 2).reshape(sequence, kv * width)
            value = entry["value"].transpose(1, 2).reshape(sequence, kv * width)
            weights = entry["weights"].reshape(heads, sequence, sequence)
            scores = entry["scores"].reshape(heads, sequence, sequence)
            core_output = core[layer].reshape(sequence, heads * width)
            local = "sliding" in config.layer_types[layer]
            prefix = f"layer.{layer}"
            for name, tensor, shape in (
                ("query", query, (sequence, heads * width)),
                ("key", key, (sequence, kv * width)),
                ("value", value, (sequence, kv * width)),
                ("scores", scores, (heads, sequence, sequence)),
                ("weights", weights, (heads, sequence, sequence)),
                ("output", core_output, (sequence, heads * width)),
            ):
                tensors[f"{prefix}.{name}"] = torch.tensor(
                    bits(tensor, shape), dtype=torch.int32
                )
            report["cases"].append(
                {
                    "layer": layer,
                    "heads": heads,
                    "kv_heads": kv,
                    "head_dim": width,
                    "sequence": sequence,
                    "window": config.sliding_window if local else None,
                }
            )
        save_file(tensors, output / "tensors.safetensors")
        report["tensor_file"] = {
            "sha256": sha256(output / "tensors.safetensors"),
            "size_bytes": (output / "tensors.safetensors").stat().st_size,
        }
        report["complete"] = True
    finally:
        write_json(output / "reference.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    try:
        collect(args.snapshot, args.output, args.threads)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"attention reference collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
