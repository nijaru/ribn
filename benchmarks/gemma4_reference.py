#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.14.0", "transformers==5.20.0", "safetensors==0.8.0", "tokenizers==0.23.3"]
# ///
"""Collect pinned Gemma 4 E4B CPU reference tensors, not Ribn support evidence.

Inputs must already exist locally; this tool does not download or execute Hub
code. Output must be a new directory outside the package. FP32 is a diagnostic
arithmetic policy, not a bit-identical reproduction of BF16 execution.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import tomllib
from pathlib import Path

MANIFEST = Path(__file__).with_name("gemma4-reference") / "model.toml"
CHAT = "Explain why a mutex protects shared mutable state."
FILLER = (
    "A mutex protects shared mutable state by allowing one owner to access it at a time. "
    "Ownership, cancellation, and completion are separate concerns. "
    "A request must retain its resources until the operation that uses them has completed.\n"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(snapshot, files):
    """Verify bytes, not a directory name or model-family assertion."""
    actual = {entry.name for entry in snapshot.iterdir()}
    require(
        actual == set(files),
        f"package members differ: extra={sorted(actual - set(files))}, missing={sorted(set(files) - actual)}",
    )
    result = {}
    for name, expected in files.items():
        require(Path(name).name == name, "manifest members must be flat filenames")
        path = snapshot / name
        require(path.stat().st_size == expected["size_bytes"], f"wrong size: {name}")
        digest = sha256(path)
        require(digest == expected["sha256"], f"wrong SHA256: {name}")
        result[name] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return result


def verify_reference_sources(base, expected):
    observed = {name: sha256(base / name) for name in expected}
    require(
        observed == expected, "installed reference source differs from the upstream pin"
    )
    return observed


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def retain(tensors, name, tensor):
    require(
        tensor.device.type == "cpu" and tensor.isfinite().all().item(),
        f"nonfinite/non-CPU reference tensor: {name}",
    )
    tensors[name] = tensor.detach().contiguous().clone()


def capture_cache(tensors, prefix, cache, config, history_length):
    owners = config.num_hidden_layers - config.num_kv_shared_layers
    require(len(cache.layers) == owners, "wrong KV-owner count")
    shapes = []
    for layer, state in enumerate(cache.layers):
        local = config.layer_types[layer] == "sliding_attention"
        length = (
            min(history_length, config.sliding_window - 1) if local else history_length
        )
        # The reference deliberately rejects global head_dim access when layer
        # geometry differs. Its prepared per-layer config is authoritative.
        geometry = config.per_layer_config[layer]
        expected = (1, geometry.num_key_value_heads, length, geometry.head_dim)
        require(
            tuple(state.keys.shape) == expected
            and tuple(state.values.shape) == expected,
            "wrong persistent KV shape",
        )
        shapes.append({"layer": layer, "shape": list(expected)})
        # Full persistent donor caches expose sliding truncation; transient spans
        # consumed by shared-tail layers are captured separately by attention hooks.
        # other owners retain only their newest K/V row.
        for kind, value in [("keys", state.keys), ("values", state.values)]:
            retain(
                tensors,
                f"{prefix}.kv.{layer}.{kind}",
                value if layer in (22, 23) else value[:, :, -1, :],
            )
    return shapes


def cases(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": CHAT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    chat_ids = tokenizer.encode(rendered, add_special_tokens=False)
    filler_ids = [tokenizer.bos_token_id] + tokenizer.encode(
        FILLER * 100, add_special_tokens=False
    )
    forced = tokenizer.encode(
        " A mutex protects shared state.", add_special_tokens=False
    )[:3]
    require(len(forced) == 3 and len(filler_ids) >= 513, "insufficient boundary input")
    return [
        {
            "name": "chat",
            "prompt_ids": chat_ids,
            "rendered_prompt": rendered,
            "policy": "greedy",
        },
        *[
            {
                "name": f"window_{length}",
                "prompt_ids": filler_ids[:length],
                "policy": "forced",
                "forced_ids": forced,
            }
            for length in (511, 513)
        ],
    ]


def capture_cases(model, tokenizer, report, tensors, output):
    import torch

    config = model.config.get_text_config()
    stops = set(model.generation_config.eos_token_id)
    language = model.model.language_model
    hooks = []
    tap_prefix = ""
    span_shapes = {}
    span_lengths = {}
    tap_shapes = {}
    seen_taps = set()
    for name, module in [
        ("embedding", language.embed_tokens),
        ("ple_lookup", language.embed_tokens_per_layer),
        ("ple_projection_norm", language.per_layer_projection_norm),
        ("raw_logits", model.lm_head),
    ]:

        def tap(_module, _inputs, value, name=name):
            require(
                tuple(value.shape) == tap_shapes[name],
                f"wrong auxiliary tap shape: {name}",
            )
            seen_taps.add(name)
            retain(tensors, f"{tap_prefix}.{name}", value[0, -1])

        hooks.append(module.register_forward_hook(tap))

    def donor_tap(module, _inputs, kwargs, _value):
        key, value = kwargs["shared_kv_states"][module.layer_type]
        geometry = config.per_layer_config[module.layer_idx]
        expected = (
            1,
            geometry.num_key_value_heads,
            span_lengths[module.layer_type],
            geometry.head_dim,
        )
        require(
            tuple(key.shape) == expected and tuple(value.shape) == expected,
            "wrong donor attention span",
        )
        span_shapes[str(module.layer_idx)] = list(expected)
        retain(tensors, f"{tap_prefix}.donor_span.{module.layer_idx}.keys", key)
        retain(tensors, f"{tap_prefix}.donor_span.{module.layer_idx}.values", value)

    for layer in (22, 23):
        hooks.append(
            language.layers[layer].self_attn.register_forward_hook(
                donor_tap, with_kwargs=True
            )
        )
    try:
        with torch.inference_mode():
            for case in cases(tokenizer):
                report["cases"].append(case)
                case["steps"] = []
                history = list(case["prompt_ids"])
                inputs = list(history)
                cache = None
                for step in range(4):
                    tap_prefix = f"{case['name']}.{step}"
                    positions = list(range(len(history) - len(inputs), len(history)))
                    seen_taps.clear()
                    tap_shapes = {
                        "embedding": (1, len(inputs), config.hidden_size),
                        "ple_lookup": (
                            1,
                            len(inputs),
                            config.num_hidden_layers
                            * config.hidden_size_per_layer_input,
                        ),
                        "ple_projection_norm": (
                            1,
                            len(inputs),
                            config.num_hidden_layers,
                            config.hidden_size_per_layer_input,
                        ),
                        "raw_logits": (1, 1, config.vocab_size),
                    }
                    span_shapes.clear()
                    span_lengths["sliding_attention"] = min(
                        len(history) - len(inputs), config.sliding_window - 1
                    ) + len(inputs)
                    span_lengths["full_attention"] = len(history)
                    result = model(
                        input_ids=torch.tensor(
                            [inputs], dtype=torch.long, device="cpu"
                        ),
                        attention_mask=torch.ones(
                            (1, len(history)), dtype=torch.long, device="cpu"
                        ),
                        position_ids=torch.tensor(
                            [positions], dtype=torch.long, device="cpu"
                        ),
                        past_key_values=cache,
                        use_cache=True,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                    require(
                        seen_taps == set(tap_shapes), "missing required auxiliary taps"
                    )
                    require(
                        set(span_shapes) == {"22", "23"},
                        "missing donor attention taps",
                    )
                    require(
                        tuple(result.logits.shape) == (1, 1, config.vocab_size),
                        "wrong logit shape",
                    )
                    retain(tensors, f"{tap_prefix}.logits", result.logits[0, -1])
                    require(
                        len(result.hidden_states) == config.num_hidden_layers + 1,
                        "wrong hidden-state count",
                    )
                    for layer, hidden in enumerate(result.hidden_states):
                        require(
                            tuple(hidden.shape) == (1, len(inputs), config.hidden_size),
                            "wrong hidden-state shape",
                        )
                        retain(tensors, f"{tap_prefix}.hidden.{layer}", hidden[0, -1])
                    cache = result.past_key_values
                    cache_shapes = capture_cache(
                        tensors, tap_prefix, cache, config, len(history)
                    )
                    chosen = int(result.logits[0, -1].argmax())
                    logprobs = result.logits[0, -1].float().log_softmax(dim=-1)
                    top_values, top_ids = logprobs.topk(5)
                    case["steps"].append(
                        {
                            "index": step,
                            "history_ids": list(history),
                            "input_ids": inputs,
                            "position_ids": positions,
                            "attention_mask": "all ones over history_ids",
                            "chosen": chosen,
                            "cache": cache_shapes,
                            "donor_spans": dict(span_shapes),
                            "top_logprobs": [
                                {"id": key, "logprob": value}
                                for key, value in zip(
                                    top_ids.tolist(), top_values.tolist()
                                )
                            ],
                        }
                    )
                    if step == 3 or (case["policy"] == "greedy" and chosen in stops):
                        break
                    # Boundary cases teacher-force public fixed IDs, even if the
                    # reference winner differs. They are not serving/EOS traces.
                    next_id = (
                        chosen
                        if case["policy"] == "greedy"
                        else case["forced_ids"][step]
                    )
                    inputs = [next_id]
                    history.append(next_id)
                write_json(output / "reference.json", report)
                print(
                    f"collected {case['name']}: {len(case['steps'])} rows",
                    flush=True,
                )
    finally:
        for hook in hooks:
            hook.remove()


def collect(snapshot, output, precision, threads):
    manifest = tomllib.loads(MANIFEST.read_text())
    require(threads > 0, "thread count must be positive")
    require(precision in ("fp32", "bf16"), "unsupported reference precision")
    require(
        not output.resolve().is_relative_to(snapshot.resolve()),
        "keep evidence outside the package",
    )
    output.mkdir(
        parents=True
    )  # Refuse to replace earlier evidence, including failures.
    report = {
        "schema_version": 1,
        "complete": False,
        "checkpoint": manifest["checkpoint"],
        "manifest_sha256": sha256(MANIFEST),
        "generator_sha256": sha256(Path(__file__)),
        "precision": precision,
        "device": "cpu",
        "attention": "eager",
        "threads": threads,
        "platform": platform.platform(),
        "python": sys.version,
        "scope": "text-only independent reference; no Ribn, GPU-fit, numerical or task-quality acceptance",
        "cases": [],
    }
    write_json(output / "reference.json", report)
    tensors = {}
    started = time.monotonic()
    try:
        report["files"] = verify_files(snapshot, manifest["files"])
        report["packages"] = {}
        for name in ("torch", "transformers", "safetensors", "tokenizers"):
            version = importlib.metadata.version(name)
            require(
                version == manifest["reference"][name],
                f"use pinned {name}=={manifest['reference'][name]}, got {version}",
            )
            report["packages"][name] = version
        # Local-only loading is explicit as well as enforced in the library calls.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

        torch.set_num_threads(threads)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        report["torch_build"] = torch.__config__.show()
        import transformers

        base = Path(transformers.__file__).parent
        report["implementation"] = {
            "declared_upstream_commit": manifest["reference"]["transformers_commit"],
            "whole_installation_identity_verified": False,
            "verified_file_sha256": verify_reference_sources(
                base, manifest["reference"]["source_sha256"]
            ),
        }
        report["environment"] = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
        )
        catalog = {}
        with safe_open(
            snapshot / manifest["checkpoint"]["weights"], framework="pt", device="cpu"
        ) as artifact:
            names = artifact.keys()
            require(
                len(names) == manifest["checkpoint"]["tensor_count"],
                "wrong tensor catalog",
            )
            for name in names:
                view = artifact.get_slice(name)
                require(
                    view.get_dtype() == manifest["checkpoint"]["scalar_type"],
                    f"unexpected dtype: {name}",
                )
                catalog[name] = {"shape": view.get_shape(), "dtype": view.get_dtype()}
        write_json(output / "catalog.json", catalog)

        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        model, loading = Gemma4ForConditionalGeneration.from_pretrained(
            snapshot,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        # Stock loading intentionally ignores the 54 stored shared-tail K/V/K-norm
        # tensors. Do not tolerate any additional loader mismatch or random init.
        report["loading"] = {key: sorted(value) for key, value in loading.items()}
        require(not any(loading.values()), "reference checkpoint did not load cleanly")
        model.eval()
        require(
            all(p.device.type == "cpu" for p in model.parameters()),
            "reference must stay on CPU",
        )
        require(
            model.lm_head.weight.data_ptr()
            == model.get_input_embeddings().weight.data_ptr(),
            "output embedding is not tied",
        )
        config = model.config.get_text_config()
        require(
            config._attn_implementation == "eager",
            "unexpected attention implementation",
        )
        report["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        require(
            report["parameter_dtypes"] == [str(dtype)], "unexpected parameter precision"
        )
        report["generation_config"] = model.generation_config.to_dict()
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        report["tokenizer_class"] = type(tokenizer).__name__

        capture_cases(model, tokenizer, report, tensors, output)
        save_file(tensors, output / "tensors.safetensors")
        report["tensor_file"] = {
            "sha256": sha256(output / "tensors.safetensors"),
            "size_bytes": (output / "tensors.safetensors").stat().st_size,
        }
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "reference.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    try:
        collect(args.snapshot, args.output, args.precision, args.threads)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"reference collection failed: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}; independent reference only, not Ribn qualification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
