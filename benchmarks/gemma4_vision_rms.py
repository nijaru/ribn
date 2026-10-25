#!/usr/bin/env python3
"""Audit vision RMS working-precision stages on frozen common operands.

Use the pinned vision reference environment. This is not a new precision policy,
acceptance tolerance, full-model forward or decoder run.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib

import gemma4_reference as reference
from gemma4_reference import require, sha256, write_json
from gemma4_vision_components import CASES, NORMS, one_layer_package
from gemma4_vision_reference import Archive, runtime

INPUTS = {
    "input_norm": "hidden",
    "q_norm": "q",
    "k_norm": "k",
    "v_norm": "v",
    "post_attention_norm": "o",
    "pre_ffn_norm": "residual_attention",
    "post_ffn_norm": "down",
}


def collect(snapshot, components, output):
    for source in (snapshot, components):
        require(
            not output.resolve().is_relative_to(source.resolve()),
            "output overlaps inputs",
        )
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "complete": False,
        "kind": "gemma4_vision_rms_stages",
        "scope": "FP32 RMS stage diagnostics only; no acceptance-policy change",
        "generator_sha256": sha256(Path(__file__)),
        "cases": [],
    }
    write_json(output / "reference.json", report)
    try:
        manifest = tomllib.loads(reference.MANIFEST.read_text())
        report["files"] = reference.verify_files(snapshot, manifest["files"])
        original = components / "reference.json"
        source = json.loads(original.read_text())
        require(
            source["complete"] and source["kind"] == "gemma4_vision_layer0_components",
            "need complete shared-operand corpus",
        )
        require(source["files"] == report["files"], "different checkpoint")
        require(
            tuple(c["name"] for c in source["cases"]) == CASES, "different frozen cases"
        )
        require(
            all(c["layer0_exact"] for c in source["cases"]),
            "missing full-layer controls",
        )
        report["common_reference_sha256"] = sha256(original)
        torch = runtime(report, manifest, 2)
        from safetensors import safe_open
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        package, weights = one_layer_package(snapshot, output)
        model, loading = Gemma4VisionModel.from_pretrained(
            package,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        require(not any(loading.values()), f"unclean loading: {loading}")
        state = model.state_dict()
        require(state.keys() == weights.keys(), "different persistent inventory")
        require(
            all(
                torch.equal(value.view(torch.uint16), weights[name].view(torch.uint16))
                for name, value in state.items()
            ),
            "persistent bytes changed",
        )
        require(
            model.encoder.rotary_emb.inv_freq.dtype == torch.float32,
            "derived frequencies changed",
        )
        del weights, state
        model.eval()
        modules = dict(model.encoder.layers[0].named_modules())
        archive = Archive(output, limit=1024**3)
        with torch.inference_mode():
            for case in source["cases"]:
                name = case["name"]
                path = components / case["archive"]["file"]
                require(
                    sha256(path) == case["archive"]["sha256"], "common archive changed"
                )
                controls = []
                with safe_open(path, framework="pt", device="cpu") as frozen:
                    for module_path, stage in NORMS.items():
                        module = modules[module_path]
                        expected = frozen.get_tensor(stage)
                        hidden = frozen.get_tensor(INPUTS[stage]).reshape(
                            expected.shape
                        )
                        actual = module(hidden)
                        require(
                            torch.equal(
                                actual.view(torch.uint16), expected.view(torch.uint16)
                            ),
                            f"BF16 control failed: {name}/{stage}",
                        )
                        x = hidden.float()
                        working = module(x)
                        require(
                            torch.equal(
                                working.to(torch.bfloat16).view(torch.uint16),
                                expected.view(torch.uint16),
                            ),
                            f"FP32 return control failed: {name}/{stage}",
                        )
                        mean = x.pow(2).mean(-1, keepdim=True)
                        statistic = mean + module.eps
                        inverse = torch.pow(statistic, -0.5)
                        unscaled = x * inverse
                        scaled = (
                            unscaled * module.weight.float()
                            if module.with_scale
                            else unscaled
                        )
                        require(
                            torch.equal(
                                scaled.view(torch.int32), working.view(torch.int32)
                            ),
                            f"unrounded stage control failed: {name}/{stage}",
                        )
                        for label, tensor in [
                            ("mean", mean),
                            ("statistic", statistic),
                            ("inverse", inverse),
                            ("unscaled", unscaled),
                            ("scaled", scaled),
                        ]:
                            archive.put(stage + "." + label, tensor)
                        controls.append(
                            {
                                "name": stage,
                                "shape": list(expected.shape),
                                "width": expected.shape[-1],
                                "with_scale": module.with_scale,
                                "epsilon": module.eps,
                                "bf16_exact": True,
                                "fp32_return_exact": True,
                                "staged_fp32_exact": True,
                            }
                        )
                report["cases"].append(
                    {
                        "name": name,
                        "images": case["images"],
                        "controls": controls,
                        "archive": archive.flush(name),
                    }
                )
                write_json(output / "reference.json", report)
                print(name, "all7 RMS controls exact", flush=True)
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
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        collect(args.snapshot, args.components, args.output)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"vision RMS audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
