#!/usr/bin/env python3
# /// script
# requires-python = "==3.13.11"
# dependencies = ["torch==2.14.0", "transformers==5.20.0", "safetensors==0.8.0", "tokenizers==0.23.3", "numpy==2.5.4", "pillow==13.0.0", "torchvision==0.29.1"]
# ///
"""Canonical synthetic-RGB Gemma 4 reference, not Ribn multimodal qualification.

Prepare inputs without loading weights, then collect stock CPU encoder/projector
and image+text forwards. No encoded image files, Arrow, network or Hub code.
Existing evidence is never replaced. FP32 is diagnostic, not ground truth.
"""

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import tomllib

import gemma4_reference as reference
from gemma4_reference import require, sha256, write_json

VISION_MANIFEST = reference.MANIFEST.with_name("vision.toml")
MAX_BYTES = 1024**3 - 16 * 1024**2  # Leave room for archive headers/reports.
QUESTION = "What is the main color in this image? Answer briefly."


def runtime(report, manifest, threads):
    torch = reference.prepare_cpu_reference(report, manifest["reference"], threads)
    vision = tomllib.loads(VISION_MANIFEST.read_text())
    for name, version in vision["packages"].items():
        require(
            importlib.metadata.version(name) == version, f"use pinned {name}=={version}"
        )
    import transformers

    report["vision_sources"] = reference.verify_reference_sources(
        Path(transformers.__file__).parent,
        vision["source_sha256"],
    )
    return torch


def image_spans(ids, image_id):
    """Half-open contiguous image-placeholder ranges, not enclosing BOI/EOI."""
    spans = []
    for index, token in enumerate(ids):
        if token == image_id:
            if spans and spans[-1][1] == index:
                spans[-1][1] += 1
            else:
                spans.append([index, index + 1])
    return spans


def fixtures():
    import numpy as np
    from PIL import Image

    red = np.zeros((48, 48, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros_like(red)
    blue[:, :, 2] = 255
    landscape = np.concatenate((red, blue), axis=1)
    arrays = {"red": red, "blue": blue, "landscape": landscape}
    images = {name: Image.fromarray(array) for name, array in arrays.items()}

    def image(name):
        return {"type": "image", "image": images[name]}

    def text(value):
        return {"type": "text", "text": value}

    cases = [
        ("red", ["red"], [image("red"), text(QUESTION)]),
        ("blue", ["blue"], [image("blue"), text(QUESTION)]),
        (
            "landscape",
            ["landscape"],
            [image("landscape"), text("What color is on the left? Answer briefly.")],
        ),
        (
            "two_images",
            ["red", "blue"],
            [
                text("First image:"),
                image("red"),
                text("Second image:"),
                image("blue"),
                text(
                    "What is the main color in the first image and in the second image? Answer briefly."
                ),
            ],
        ),
    ]
    return arrays, cases


def validate_inputs(inputs, image_id, pooling_kernel):
    """Validate the canonical processor/model boundary before any model launch."""
    import torch

    require(
        set(inputs)
        == {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_position_ids",
        },
        "unexpected processor keys",
    )
    ids, positions, pixels = (
        inputs["input_ids"],
        inputs["image_position_ids"],
        inputs["pixel_values"],
    )
    require(
        ids.dtype == torch.int64 and ids.ndim == 2 and ids.shape[0] == 1,
        "invalid token batch",
    )
    for name in ("attention_mask", "mm_token_type_ids"):
        require(
            inputs[name].dtype == torch.int64 and inputs[name].shape == ids.shape,
            f"invalid {name}",
        )
    require(bool((inputs["attention_mask"] == 1).all()), "unexpected text padding")
    require(
        pixels.dtype == torch.float32 and pixels.ndim == 3 and pixels.shape[2] == 768,
        "invalid canonical patch tensor",
    )
    require(
        positions.dtype == torch.int64 and positions.shape == (*pixels.shape[:2], 2),
        "invalid patch positions",
    )
    require(
        all(
            v.device.type == "cpu" and bool(v.isfinite().all()) for v in inputs.values()
        ),
        "nonfinite/non-CPU processor output",
    )
    require(bool(((pixels >= 0) & (pixels <= 1)).all()), "patches are not rescaled RGB")
    padding = (positions == -1).all(-1)
    require(
        bool(((positions >= 0).all(-1) | padding).all()), "partial padding position"
    )
    require(bool((pixels[padding] == 0).all()), "nonzero padding patches")
    counts = (~padding).sum(-1).tolist()
    area = pooling_kernel**2
    require(all(n > 0 and n % area == 0 for n in counts), "incomplete pooling cells")
    soft = [n // area for n in counts]
    spans = image_spans(ids[0].tolist(), image_id)
    require(
        [end - start for start, end in spans] == soft,
        "image features and placeholder spans disagree",
    )
    return {"patch_counts": counts, "soft_token_counts": soft, "image_spans": spans}


def prepare(snapshot, output, threads):
    require(
        not output.resolve().is_relative_to(snapshot.resolve()),
        "keep evidence outside package",
    )
    output.mkdir(parents=True)
    report = {
        "complete": False,
        "scope": "canonical generated RGB; no model weights loaded",
        "cases": [],
    }
    try:
        manifest = tomllib.loads(reference.MANIFEST.read_text())
        report["files"] = reference.verify_files(snapshot, manifest["files"])
        torch = runtime(report, manifest, threads)
        from safetensors.torch import save_file
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        require(
            processor.image_processor.max_soft_tokens == 280,
            "unexpected default image budget",
        )
        arrays, cases = fixtures()
        tensors = {
            f"rgb.{name}": torch.from_numpy(array.copy())
            for name, array in arrays.items()
        }
        report.update(
            image_token_id=processor.tokenizer.image_token_id,
            pooling_kernel=3,
            max_soft_tokens=280,
        )
        for name, images, content in cases:
            messages = [{"role": "user", "content": content}]
            options = {"add_generation_prompt": True, "enable_thinking": False}
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **options,
            )
            layout = validate_inputs(
                inputs, report["image_token_id"], report["pooling_kernel"]
            )
            tensors.update(
                {f"{name}.{key}": value.contiguous() for key, value in inputs.items()}
            )
            report["cases"].append(
                {
                    "name": name,
                    "images": images,
                    "rendered_template": processor.apply_chat_template(
                        messages, tokenize=False, **options
                    ),
                    "prompt_ids": inputs["input_ids"][0].tolist(),
                    **layout,
                }
            )
        require(
            torch.equal(tensors["red.input_ids"], tensors["blue.input_ids"]),
            "color controls have different text",
        )
        save_file(tensors, output / "inputs.safetensors")
        report["tensor_file"] = {
            "sha256": sha256(output / "inputs.safetensors"),
            "size_bytes": (output / "inputs.safetensors").stat().st_size,
        }
        report["manifest_sha256"] = sha256(reference.MANIFEST)
        report["vision_manifest_sha256"] = sha256(VISION_MANIFEST)
        report["generator_sha256"] = sha256(Path(__file__))
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "inputs.json", report)
    return report


class Archive:
    """Bound retained CPU tensors across per-case archives; never change arithmetic."""

    def __init__(self, output, limit=MAX_BYTES):
        self.output, self.limit, self.used = output, limit, 0
        self.tensors = {}

    def put(self, name, value):
        require(name not in self.tensors, f"duplicate tap: {name}")
        require(
            value.device.type == "cpu" and bool(value.isfinite().all()),
            f"nonfinite/non-CPU tap: {name}",
        )
        size = value.numel() * value.element_size()
        require(self.used + size <= self.limit, "tensor byte budget exceeded")
        self.tensors[name] = value.detach().contiguous().clone()
        self.used += size

    def flush(self, name):
        from safetensors.torch import save_file

        path = self.output / f"{name}.safetensors"
        require(not path.exists(), "refuse to replace tensor evidence")
        metadata = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in self.tensors.items()
        }
        save_file(self.tensors, path)
        result = {
            "file": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "tensors": metadata,
        }
        self.tensors.clear()
        return result


@contextmanager
def vision_taps(model, archive):
    """Observe the actual prefill, including inputs passed to the text decoder."""
    hooks = []

    def output(name):
        return lambda module, args, value: archive.put(name, value)

    try:
        tower = model.model.vision_tower
        hooks.append(
            tower.patch_embedder.register_forward_hook(
                output("vision.patch_embeddings")
            )
        )
        for index, layer in enumerate(tower.encoder.layers):
            hooks.append(layer.register_forward_hook(output(f"vision.layer.{index}")))

        def pool(module, args, value):
            archive.put("vision.pooled_padded", value[0])
            archive.put("vision.pooler_mask", value[1])

        hooks.append(tower.pooler.register_forward_hook(pool))
        hooks.append(
            tower.register_forward_hook(
                lambda module, args, value: archive.put(
                    "vision.valid", value.last_hidden_state
                )
            )
        )
        hooks.append(
            model.model.embed_vision.embedding_pre_projection_norm.register_forward_hook(
                output("vision.projection_norm")
            )
        )
        hooks.append(
            model.model.embed_vision.register_forward_hook(output("vision.projected"))
        )

        def text_input(module, args, kwargs):
            archive.put("text.inputs_embeds", kwargs["inputs_embeds"])
            archive.put("text.per_layer_inputs", kwargs["per_layer_inputs"])

        hooks.append(
            model.model.language_model.register_forward_pre_hook(
                text_input, with_kwargs=True
            )
        )
        yield
    finally:
        for hook in hooks:
            hook.remove()


def text_controls(model, tokenizer, path, precision, archive):
    """Optional exact historic rows, on their original chunk/forced-token history."""
    import torch

    controls = json.loads(path.read_text())
    require(
        tokenizer.encode(controls["prompt"], add_special_tokens=False)
        == controls["prompt_ids"],
        "control tokenization differs",
    )
    history, cache, rows = [], None, []
    require(
        len(controls["calls"]) == len(controls[precision]) == 5,
        "expected five text controls",
    )
    require(
        sum(controls["calls"][:2], []) == controls["prompt_ids"],
        "invalid control prompt chunks",
    )
    for index, (tokens, expected) in enumerate(
        zip(controls["calls"], controls[precision], strict=True)
    ):
        start = len(history)
        history.extend(tokens)
        result = model(
            input_ids=torch.tensor([tokens]),
            attention_mask=torch.ones((1, len(history)), dtype=torch.long),
            position_ids=torch.arange(start, len(history))[None],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = result.past_key_values
        value = result.logits[0, -1].detach().contiguous()
        source = path.parent / expected["file"]
        require(sha256(source) == expected["sha256"], "historic control changed")
        require(
            value.view(torch.uint8).numpy().tobytes() == source.read_bytes(),
            f"historic text row differs at {start}",
        )
        archive.put(f"control.{index}.logits", value)
        rows.append(
            {
                "position": start,
                "input_ids": tokens,
                "chosen": int(value.argmax()),
                "source_sha256": expected["sha256"],
            }
        )
    return rows


def capture_case(model, tokenizer, inputs, layout, archive, max_new_tokens):
    import torch

    history = inputs["input_ids"][0].tolist()
    with vision_taps(model, archive):
        result = model(**inputs, use_cache=True, logits_to_keep=1)
    projected = archive.tensors["vision.projected"]
    require(
        torch.equal(projected, result.image_hidden_states),
        "projected output tap differs",
    )
    require(
        list(projected.shape)
        == [sum(layout["soft_token_counts"]), model.config.text_config.hidden_size],
        "wrong projected features",
    )
    mask = inputs["input_ids"][0] == model.config.image_token_id
    require(
        torch.equal(archive.tensors["text.inputs_embeds"][0, mask], projected),
        "image/text insertion differs",
    )
    generated, rows = [], []
    stops = set(model.generation_config.eos_token_id)
    for step in range(max_new_tokens):
        logits = result.logits[0, -1]
        archive.put(f"step.{step}.logits", logits)
        token = int(logits.argmax())
        rows.append({"history_ids": history.copy(), "chosen": token})
        generated.append(token)
        if token in stops or step + 1 == max_new_tokens:
            break
        position = len(history)
        history.append(token)
        result = model(
            input_ids=torch.tensor([[token]]),
            attention_mask=torch.ones((1, len(history)), dtype=torch.long),
            position_ids=torch.tensor([[position]]),
            past_key_values=result.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    return {
        "rows": rows,
        "generated_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "finish": "stop" if generated[-1] in stops else "length",
    }


def collect(snapshot, prepared, output, precision, threads, controls=None):
    require(
        not output.resolve().is_relative_to(snapshot.resolve()),
        "keep evidence outside package",
    )
    require(precision in ("bf16", "fp32"), "unsupported precision")
    output.mkdir(parents=True)
    report = {
        "complete": False,
        "scope": "stock CPU synthetic-RGB integration reference; not Ribn VLM, task-quality or serving acceptance",
        "precision": precision,
        "threads": threads,
        "max_new_tokens": 8,
        "cases": [],
    }
    started = time.monotonic()
    write_json(output / "reference.json", report)
    try:
        manifest = tomllib.loads(reference.MANIFEST.read_text())
        report["files"] = reference.verify_files(snapshot, manifest["files"])
        torch = runtime(report, manifest, threads)
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

        data = json.loads((prepared / "inputs.json").read_text())
        require(
            data["complete"]
            and data["manifest_sha256"] == sha256(reference.MANIFEST)
            and data["vision_manifest_sha256"] == sha256(VISION_MANIFEST)
            and data["generator_sha256"] == sha256(Path(__file__)),
            "input provenance differs",
        )
        require(
            data["tensor_file"]["sha256"] == sha256(prepared / "inputs.safetensors"),
            "prepared tensors changed",
        )
        report["inputs_sha256"] = sha256(prepared / "inputs.json")
        report["generator_sha256"] = sha256(Path(__file__))
        if controls is not None:
            report["text_controls_sha256"] = sha256(controls)
        tensors = load_file(prepared / "inputs.safetensors", device="cpu")
        cases = []
        for case in data["cases"]:
            inputs = {
                key.removeprefix(case["name"] + "."): value
                for key, value in tensors.items()
                if key.startswith(case["name"] + ".")
            }
            layout = validate_inputs(
                inputs, data["image_token_id"], data["pooling_kernel"]
            )
            require(
                all(case[key] == value for key, value in layout.items()),
                "prepared layout differs",
            )
            require(
                inputs["input_ids"][0].tolist() == case["prompt_ids"],
                "prepared tokens differ",
            )
            cases.append((case, inputs))
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        model, loading = Gemma4ForConditionalGeneration.from_pretrained(
            snapshot,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        require(not any(loading.values()), f"unclean loading: {loading}")
        require(
            all(
                p.device.type == "cpu" and p.dtype == dtype for p in model.parameters()
            ),
            "wrong model placement/dtype",
        )
        require(
            model.lm_head.weight.data_ptr()
            == model.get_input_embeddings().weight.data_ptr(),
            "untied output head",
        )
        require(
            model.config.text_config._attn_implementation == "eager"
            and model.config.vision_config._attn_implementation == "eager",
            "non-eager attention",
        )
        model.eval()
        report.update(
            loading={key: sorted(value) for key, value in loading.items()},
            text_bidirectional_attention=model.config.text_config.use_bidirectional_attention,
            vision_config=model.config.vision_config.to_dict(),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        archive = Archive(output)
        with torch.inference_mode():
            if controls is not None:
                report["text_controls"] = text_controls(
                    model, tokenizer, controls, precision, archive
                )
                report["control_archive"] = archive.flush("controls")
                write_json(output / "reference.json", report)
            for case, inputs in cases:
                result = capture_case(
                    model, tokenizer, inputs, case, archive, report["max_new_tokens"]
                )
                report["cases"].append(
                    {**case, **result, "archive": archive.flush(case["name"])}
                )
                write_json(output / "reference.json", report)
                print(case["name"], repr(result["text"]), result["finish"], flush=True)
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
    parser.add_argument("mode", choices=("prepare", "collect"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--text-controls", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            require(
                args.inputs is None and args.text_controls is None,
                "prepare does not consume a corpus",
            )
            prepare(args.snapshot, args.output, args.threads)
        else:
            require(args.inputs is not None, "collect requires --inputs")
            collect(
                args.snapshot,
                args.inputs,
                args.output,
                args.precision,
                args.threads,
                args.text_controls,
            )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"vision reference failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
