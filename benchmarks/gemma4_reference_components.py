"""RMS/QKV/RoPE reference recomputation; invoked by gemma4_reference.py only.

Consumes archived full-model hidden inputs but recomputes one token at a time.
It stops before attention, cache updates and output projection. These are component
operands, not new full-model taps or numerical acceptance for Ribn.
"""

import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

from gemma4_reference import (
    MANIFEST,
    prepare_cpu_reference,
    prompt_chunks,
    require,
    retain,
    sha256,
    verify_files,
    write_json,
)

LAYERS = (0, 5, 22, 23)  # Local/global, including both shared-tail KV donors.


def validated_rows(source, manifest, precision):
    """Reject incompatible provenance or internally inconsistent token histories."""
    require(
        source.get("schema_version") == 1 and source.get("complete") is True,
        "incomplete/unknown corpus",
    )
    require(
        source.get("device") == "cpu" and source.get("attention") == "eager",
        "expected eager CPU corpus",
    )
    require(
        source.get("precision") == precision,
        "corpus precision differs from requested policy",
    )
    require(source.get("checkpoint") == manifest["checkpoint"], "wrong checkpoint")
    require(
        source.get("manifest_sha256") == sha256(MANIFEST), "wrong manifest identity"
    )
    require(source.get("files") == manifest["files"], "wrong package provenance")
    require(
        source.get("packages")
        == {
            name: manifest["reference"][name]
            for name in ("torch", "transformers", "safetensors", "tokenizers")
        },
        "wrong reference package versions",
    )
    require(
        source.get("implementation", {}).get("verified_file_sha256")
        == manifest["reference"]["source_sha256"],
        "wrong implementation provenance",
    )
    config_path = MANIFEST.with_name("config.json")
    require(
        sha256(config_path) == manifest["files"]["config.json"]["sha256"],
        "wrong config identity",
    )
    vocab = json.loads(config_path.read_text())["text_config"]["vocab_size"]
    rows, names = [], set()
    for case in source["cases"]:
        name = case["name"]
        require(
            isinstance(name, str) and name and name not in names,
            "duplicate/invalid case",
        )
        names.add(name)
        require(case["policy"] in ("greedy", "forced"), "unknown history policy")
        previous = None
        require(case["steps"], "empty case")
        chunks = prompt_chunks(case["prompt_ids"], source.get("prefill_chunk_size"))
        intermediates = case.get("prefill_intermediates", [])
        require(len(intermediates) == len(chunks) - 1, "missing/extra prefill chunks")
        for call, step in enumerate(intermediates + case["steps"]):
            prediction = call - len(intermediates)
            index = call if prediction < 0 else prediction
            suffix = f"prefill.{call}" if prediction < 0 else str(prediction)
            history, inputs = step["history_ids"], step["input_ids"]
            require(
                type(step["index"]) is int and step["index"] == index,
                "noncontiguous row indices",
            )
            require(
                type(step["chosen"]) is int and 0 <= step["chosen"] < vocab,
                "invalid chosen ID",
            )
            require(
                history
                and inputs
                and all(
                    type(token) is int and 0 <= token < vocab
                    for token in history + inputs
                ),
                "invalid history IDs",
            )
            require(
                history[-len(inputs) :] == inputs, "inputs are not the history suffix"
            )
            require(
                step["position_ids"]
                == list(range(len(history) - len(inputs), len(history))),
                "wrong input positions",
            )
            if call < len(chunks):
                end = sum(map(len, chunks[: call + 1]))
                require(
                    history == case["prompt_ids"][:end] and inputs == chunks[call],
                    "wrong prefill history",
                )
            else:
                next_id = (
                    previous["chosen"]
                    if case["policy"] == "greedy"
                    else case["forced_ids"][index - 1]
                )
                require(
                    history == previous["history_ids"] + [next_id]
                    and inputs == [next_id],
                    "wrong continuation history",
                )
            rows.append(
                {
                    "name": f"{name}.{suffix}",
                    "history_ids": history,
                    "position": len(history) - 1,
                    "original_input_tokens": len(inputs),
                }
            )
            previous = step
    require(rows, "no component inputs")
    return rows


def collect_components(snapshot, corpus, output, precision, threads):
    require(threads > 0, "thread count must be positive")
    require(
        not output.resolve().is_relative_to(snapshot.resolve()),
        "keep evidence outside the package",
    )
    require(
        not output.resolve().is_relative_to(corpus.resolve()),
        "keep evidence outside the input corpus",
    )
    output.mkdir(parents=True)  # Never overwrite earlier evidence, including failures.
    manifest = tomllib.loads(MANIFEST.read_text())
    report = {
        "schema_version": 1,
        "kind": "gemma4_attention_components",
        "complete": False,
        "checkpoint": manifest["checkpoint"],
        "manifest_sha256": sha256(MANIFEST),
        "generator_sha256": sha256(Path(__file__)),
        "entrypoint_sha256": sha256(Path(__file__).with_name("gemma4_reference.py")),
        "precision": precision,
        "device": "cpu",
        "threads": threads,
        "python": sys.version,
        "scope": "one-token recomputation from archived hidden inputs; stops before attention/cache/output projection; no Ribn or full-model acceptance",
        "layers": list(LAYERS),
    }
    started = time.monotonic()
    write_json(output / "reference.json", report)
    try:
        source_bytes = (corpus / "reference.json").read_bytes()
        source = json.loads(source_bytes)
        report["rows"] = validated_rows(source, manifest, precision)
        tensor_path = corpus / "tensors.safetensors"
        require(
            tensor_path.stat().st_size == source["tensor_file"]["size_bytes"]
            and sha256(tensor_path) == source["tensor_file"]["sha256"],
            "input tensor file differs from corpus",
        )
        report["input_trace"] = {
            "report_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "tensor_file": source["tensor_file"],
            "original_execution_independently_reverified": False,
        }
        report["files"] = verify_files(snapshot, manifest["files"])
        torch = prepare_cpu_reference(report, manifest["reference"], threads)
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4RMSNorm,
            Gemma4TextAttention,
            Gemma4TextRotaryEmbedding,
        )

        config = Gemma4TextConfig(
            **json.loads((snapshot / "config.json").read_text())["text_config"]
        )
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        tensors = {}
        prefix = None
        shapes = {}

        def save(stage, tensor):
            require(
                tuple(tensor.shape) == shapes[stage] and tensor.dtype == dtype,
                f"wrong component shape/dtype: {prefix}.{stage}",
            )
            key = f"{prefix}.{stage}"
            require(key not in tensors, f"duplicate component: {key}")
            retain(tensors, key, tensor)

        class Captured(Exception):
            pass

        def stop_before_attention(_module, query, key, value, _mask, **_kwargs):
            save("rotated_query", query)
            save("rotated_key", key)
            save("normalized_value", value)
            raise Captured

        # The pinned attention forward does all projections, Q/K/V norms and RoPE.
        # Only its attention boundary is replaced; no fake attention result escapes.
        ALL_ATTENTION_FUNCTIONS.register(
            "ribn_component_capture", stop_before_attention
        )
        config._attn_implementation = "ribn_component_capture"
        rotary = Gemma4TextRotaryEmbedding(config, device="cpu")
        with (
            torch.inference_mode(),
            safe_open(
                snapshot / manifest["checkpoint"]["weights"],
                framework="pt",
                device="cpu",
            ) as weights,
            safe_open(tensor_path, framework="pt", device="cpu") as inputs,
        ):
            for layer in LAYERS:
                geometry = config.per_layer_config[layer]
                dim, heads, kv = (
                    geometry.head_dim,
                    config.num_attention_heads,
                    geometry.num_key_value_heads,
                )
                shapes = {
                    "hidden": (1, 1, config.hidden_size),
                    "input_norm": (1, 1, config.hidden_size),
                    "q_proj": (1, 1, heads * dim),
                    "k_proj": (1, 1, kv * dim),
                    "v_proj": (1, 1, kv * dim),
                    "q_norm": (1, 1, heads, dim),
                    "k_norm": (1, 1, kv, dim),
                    "v_norm": (1, 1, kv, dim),
                    "cos": (1, 1, dim),
                    "sin": (1, 1, dim),
                    "rotated_query": (1, heads, 1, dim),
                    "rotated_key": (1, kv, 1, dim),
                    "normalized_value": (1, kv, 1, dim),
                }
                with torch.device("meta"):
                    normalizer = Gemma4RMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps
                    )
                    attention = Gemma4TextAttention(config, layer)
                root = f"model.language_model.layers.{layer}."
                for module, weight_prefix in [
                    (normalizer, root + "input_layernorm."),
                    (attention, root + "self_attn."),
                ]:
                    state = {}
                    for name in module.state_dict():
                        value = weights.get_tensor(weight_prefix + name)
                        require(
                            value.dtype == torch.bfloat16
                            and value.isfinite().all().item(),
                            "invalid source weight",
                        )
                        state[name] = value.to(dtype)
                    module.load_state_dict(state, strict=True, assign=True)
                    module.eval()
                hooks = []
                for stage in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "q_norm",
                    "k_norm",
                    "v_norm",
                ):

                    def tap(_module, _inputs, value, stage=stage):
                        save(stage, value)

                    hooks.append(getattr(attention, stage).register_forward_hook(tap))
                try:
                    for row in report["rows"]:
                        prefix = f"{row['name']}.layer.{layer}"
                        hidden = inputs.get_tensor(f"{row['name']}.hidden.{layer}")
                        require(
                            tuple(hidden.shape) == (config.hidden_size,),
                            "wrong archived hidden shape",
                        )
                        hidden = hidden.view(1, 1, config.hidden_size)
                        save("hidden", hidden)
                        normalized = normalizer(hidden)
                        save("input_norm", normalized)
                        cos, sin = rotary(
                            normalized,
                            torch.tensor(
                                [[row["position"]]], dtype=torch.long, device="cpu"
                            ),
                            attention.layer_type,
                        )
                        save("cos", cos)
                        save("sin", sin)
                        try:
                            attention(
                                normalized,
                                (cos, sin),
                                attention_mask=None,
                                shared_kv_states={},
                                past_key_values=None,
                            )
                        except Captured:
                            pass
                        else:
                            raise ValueError(
                                "reference did not stop at attention boundary"
                            )
                        require(
                            all(f"{prefix}.{stage}" in tensors for stage in shapes),
                            "missing component",
                        )
                finally:
                    for hook in hooks:
                        hook.remove()
        report["tensors"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in tensors.items()
        }
        save_file(tensors, output / "tensors.safetensors")
        report["tensor_file"] = {
            "size_bytes": (output / "tensors.safetensors").stat().st_size,
            "sha256": sha256(output / "tensors.safetensors"),
        }
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "reference.json", report)
    return report
