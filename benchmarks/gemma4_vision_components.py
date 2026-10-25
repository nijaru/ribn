#!/usr/bin/env python3
"""Replay vision layer 0 on frozen CPU operands; no decoder or parity promotion.

Use the pinned gemma4_vision_reference.py environment. The reference must reproduce
all four archived layer outputs exactly before its component operands are usable.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib

import gemma4_reference as reference
from gemma4_reference import require, sha256, write_json
from gemma4_vision_reference import Archive, runtime

CASES = ("red", "blue", "landscape", "two_images")
NORMS = {
    "input_layernorm": "input_norm",
    "self_attn.q_norm": "q_norm",
    "self_attn.k_norm": "k_norm",
    "self_attn.v_norm": "v_norm",
    "post_attention_layernorm": "post_attention_norm",
    "pre_feedforward_layernorm": "pre_ffn_norm",
    "post_feedforward_layernorm": "post_ffn_norm",
}
LINEARS = {
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gate",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
}


def one_layer_package(snapshot, output):
    """Copy only the real first-layer/patch tensors, not random or text weights."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    raw = json.loads((snapshot / "config.json").read_text())["vision_config"]
    raw["num_hidden_layers"] = 1
    config = Gemma4VisionConfig(**raw)
    package = output / "one-layer-package"
    package.mkdir()
    config.save_pretrained(package)
    prefix = "model.vision_tower."
    with safe_open(
        snapshot / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        weights = {
            name.removeprefix(prefix): source.get_tensor(name)
            for name in source.keys()
            if name.startswith(prefix + "patch_embedder.")
            or name.startswith(prefix + "encoder.layers.0.")
        }
    require(len(weights) == 43, "expected complete real first-layer inventory")
    save_file(weights, package / "model.safetensors")
    return package, weights


def taps(layer, archive):
    """Observe actual stock modules, including both sides of activation clipping."""
    handles = []
    modules = dict(layer.named_modules())

    def output(name):
        return lambda module, args, value: archive.put(name, value)

    def input_value(name):
        return lambda module, args: archive.put(name, args[0])

    for path, name in NORMS.items():
        handles.append(modules[path].register_forward_hook(output(name)))
    for path, name in LINEARS.items():
        module = modules[path]
        handles.append(module.register_forward_hook(output(name)))
        handles.append(
            module.linear.register_forward_pre_hook(input_value(name + ".linear_input"))
        )
        handles.append(
            module.linear.register_forward_hook(output(name + ".linear_output"))
        )
    handles.append(
        layer.pre_feedforward_layernorm.register_forward_pre_hook(
            input_value("residual_attention")
        )
    )
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(input_value("gated")))
    handles.append(layer.mlp.act_fn.register_forward_hook(output("gelu")))
    return handles


def collect(snapshot, corpus, output, threads):
    require(threads > 0, "positive threads required")
    for source in (snapshot, corpus):
        require(
            not output.resolve().is_relative_to(source.resolve()),
            "output must be outside inputs",
        )
    output.mkdir(parents=True)  # Preserve prior attempts, even incomplete ones.
    started = time.monotonic()
    report = {
        "complete": False,
        "kind": "gemma4_vision_layer0_components",
        "scope": "common BF16 operands, full padded batches; no decoder, tolerance or full-model acceptance",
        "generator_sha256": sha256(Path(__file__)),
        "precision": "bf16",
        "threads": threads,
        "cases": [],
    }
    write_json(output / "reference.json", report)
    try:
        manifest = tomllib.loads(reference.MANIFEST.read_text())
        report["files"] = reference.verify_files(snapshot, manifest["files"])
        source_report = corpus / "bf16/capture/reference.json"
        source = json.loads(source_report.read_text())
        require(
            source["complete"] and source["precision"] == "bf16",
            "need complete BF16 corpus",
        )
        require(source["files"] == report["files"], "different checkpoint assets")
        require(
            tuple(case["name"] for case in source["cases"]) == CASES,
            "different frozen cohort",
        )
        prepared = json.loads((corpus / "prepared-v1/inputs.json").read_text())
        require(
            sha256(corpus / "prepared-v1/inputs.json") == source["inputs_sha256"],
            "different input report",
        )
        input_path = corpus / "prepared-v1/inputs.safetensors"
        require(
            sha256(input_path) == prepared["tensor_file"]["sha256"],
            "different input tensors",
        )
        report["source_corpus_sha256"] = sha256(source_report)
        report["inputs_sha256"] = sha256(input_path)
        torch = runtime(report, manifest, threads)
        from safetensors import safe_open
        from transformers.models.gemma4 import modeling_gemma4 as modeling

        package, weights = one_layer_package(snapshot, output)
        model, loading = modeling.Gemma4VisionModel.from_pretrained(
            package,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        require(not any(loading.values()), f"unclean loading: {loading}")
        require(
            all(
                torch.equal(t.view(torch.uint16), weights[name].view(torch.uint16))
                for name, t in model.state_dict().items()
            ),
            "loaded persistent bytes changed",
        )
        require(
            model.encoder.rotary_emb.inv_freq.dtype == torch.float32,
            "rotary buffer must stay FP32",
        )
        del weights
        model.eval()
        layer = model.encoder.layers[0]
        archive = Archive(output, limit=1900 * 1024**2)
        handles = taps(layer, archive)
        original_rope = modeling.apply_multidimensional_rope
        original_attention = modeling.eager_attention_forward
        rotations = []

        def rope(*args, **kwargs):
            value = original_rope(*args, **kwargs)
            name = ("q_rope", "k_rope")[len(rotations)]
            rotations.append(name)
            archive.put(name, value)
            return value

        def attention(*args, **kwargs):
            value, probabilities = original_attention(*args, **kwargs)
            archive.put("attention", value)
            return value, probabilities

        modeling.apply_multidimensional_rope = rope
        modeling.eager_attention_forward = attention
        try:
            with (
                torch.inference_mode(),
                safe_open(input_path, framework="pt", device="cpu") as inputs,
            ):
                for case in source["cases"]:
                    name = case["name"]
                    path = corpus / "bf16/capture" / case["archive"]["file"]
                    require(
                        sha256(path) == case["archive"]["sha256"],
                        "changed frozen capture",
                    )
                    with safe_open(path, framework="pt", device="cpu") as frozen:
                        hidden = frozen.get_tensor("vision.patch_embeddings")
                        expected = frozen.get_tensor("vision.layer.0")
                    positions = inputs.get_tensor(name + ".image_position_ids")
                    require(
                        hidden.shape == (*positions.shape[:2], 768),
                        "wrong physical geometry",
                    )
                    rotations.clear()
                    archive.put("hidden", hidden)
                    archive.put("positions", positions)
                    archive.put("inv_freq", model.encoder.rotary_emb.inv_freq)
                    cos, sin = model.encoder.rotary_emb(hidden, positions)
                    archive.put("cos", cos)
                    archive.put("sin", sin)
                    mask = modeling.create_bidirectional_mask(
                        config=model.config,
                        inputs_embeds=hidden,
                        attention_mask=~(positions == -1).all(dim=-1),
                    )
                    actual = layer(
                        hidden,
                        position_embeddings=(cos, sin),
                        attention_mask=mask,
                        position_ids=positions,
                    )
                    exact = torch.equal(
                        actual.view(torch.uint16), expected.view(torch.uint16)
                    )
                    archive.put("output", actual)
                    require(
                        rotations == ["q_rope", "k_rope"], "missing/extra rotary calls"
                    )
                    report["cases"].append(
                        {
                            "name": name,
                            "images": hidden.shape[0],
                            "layer0_exact": exact,
                            "archive": archive.flush(name),
                        }
                    )
                    write_json(output / "reference.json", report)
                    require(exact, f"layer0 replay not exact: {name}")
                    print(name, "layer0 exact", flush=True)
        finally:
            modeling.apply_multidimensional_rope = original_rope
            modeling.eager_attention_forward = original_attention
            for handle in handles:
                handle.remove()
        report["tensor_bytes"] = archive.used
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "reference.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        collect(args.snapshot, args.corpus, args.output, args.threads)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"vision components failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
