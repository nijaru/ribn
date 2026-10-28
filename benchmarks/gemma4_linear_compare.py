#!/usr/bin/env python3
"""Diagnose stored BF16 linear operands against the frozen binary64 PLE pack.

File consistency is not device-execution attestation. No error budget or promotion
verdict is selected here. Requires the existing CPU Torch/SafeTensors environment.
"""

import argparse
import hashlib
import json
import math
import platform
import time
import tomllib
from pathlib import Path

import gemma4_linear_reference as linear
import gemma4_reference as reference
from gemma4_reference import require, sha256, write_json

REFERENCE_NAMES = ("embedding_rows", "input", "weight", "reference", "sum_abs_products")
CAPTURE_NAMES = ("input", "weight", "output")


def projection_errors(expected, captured, torch):
    """Reject different operands before reducing any output errors."""
    require(
        set(expected) == set(REFERENCE_NAMES),
        "wrong reference tensors",
    )
    require(set(captured) == set(CAPTURE_NAMES), "wrong capture tensors")
    require(
        all(t.device.type == "cpu" for t in (*expected.values(), *captured.values())),
        "expected CPU tensors",
    )
    x, w = expected["input"], expected["weight"]
    require(
        x.ndim == w.ndim == 2
        and 1 <= x.shape[0] <= linear.MAX_ROWS
        and x.shape[1] > 0
        and w.shape[0] > 0
        and x.shape[1] == w.shape[1],
        "wrong reference operand shapes",
    )
    shape = (x.shape[0], w.shape[0])
    for name, tensor in expected.items():
        dtype = (
            torch.float64
            if name in ("reference", "sum_abs_products")
            else torch.bfloat16
        )
        target_shape = (
            shape
            if dtype == torch.float64
            else w.shape
            if name == "weight"
            else x.shape
        )
        require(
            tensor.dtype == dtype and tensor.shape == target_shape,
            f"wrong reference shape/dtype: {name}",
        )
        require(bool(torch.isfinite(tensor).all()), f"nonfinite reference: {name}")
    require(
        bool((expected["sum_abs_products"] >= expected["reference"].abs()).all()),
        "invalid absolute-product sums",
    )
    for name in ("input", "weight"):
        a, b = expected[name], captured[name]
        require(
            b.dtype == torch.bfloat16 and b.shape == a.shape,
            f"wrong captured shape/dtype: {name}",
        )
        require(bool(torch.isfinite(b).all()), f"nonfinite captured: {name}")
        require(
            torch.equal(a.view(torch.int16), b.view(torch.int16)),
            f"operand bits differ: {name}",
        )
    output = captured["output"]
    require(
        output.dtype in (torch.bfloat16, torch.float32) and output.shape == shape,
        "wrong captured output shape/dtype",
    )
    require(bool(torch.isfinite(output).all()), "nonfinite captured output")
    target = expected["reference"]
    delta = output.to(torch.float64) - target
    norm = float(torch.linalg.vector_norm(target))
    errors = {
        "max_abs": float(delta.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "relative_l2": float(torch.linalg.vector_norm(delta)) / norm if norm else None,
        "reference_l2": norm,
    }
    require(
        all(value is None or math.isfinite(value) for value in errors.values()),
        "nonfinite reduced diagnostic",
    )
    return errors


def compare(reference_report, capture, output, threads=1):
    output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "complete": False,
        "numerical_acceptance": "none",
        "native_execution_verified": False,
        "actual_device_operand_capture_verified": False,
        "python": platform.python_version(),
    }
    started = time.monotonic()
    report_path = output / "report.json"
    write_json(report_path, report)
    try:
        require(type(threads) is int and threads > 0, "threads must be positive")
        generators = {
            "generator": Path(linear.__file__),
            "helper": Path(reference.__file__),
            "manifest": reference.MANIFEST,
        }
        paths = {"comparator": Path(__file__), **generators}
        before = {name: sha256(path) for name, path in paths.items()}
        frozen_bytes = reference_report.read_bytes()
        frozen = json.loads(frozen_bytes)
        paths["reference_report"] = reference_report
        before["reference_report"] = hashlib.sha256(frozen_bytes).hexdigest()
        manifest_bytes = reference.MANIFEST.read_bytes()
        require(
            hashlib.sha256(manifest_bytes).hexdigest() == before["manifest"],
            "manifest changed",
        )
        manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
        require(
            frozen.get("schema_version") == 1
            and frozen.get("complete") is True
            and "error" not in frozen,
            "incomplete reference",
        )
        require(
            frozen["checkpoint"] == manifest["checkpoint"]
            and frozen["package_before"]
            == frozen["package_after"]
            == manifest["files"],
            "wrong reference package identity",
        )
        require(
            frozen["source_parameters"]
            == {"embedding_rows": linear.EMBEDDING, "weight": linear.PROJECTION},
            "wrong reference parameters",
        )
        require(
            frozen["generator_sha256"]
            == {path.name: before[name] for name, path in generators.items()},
            "wrong reference generator identity",
        )
        require(
            frozen["packages"]
            == {
                key: manifest["reference"][key]
                for key in ("torch", "transformers", "safetensors", "tokenizers")
            },
            "wrong reference runtime identity",
        )
        descriptor = frozen["artifact"]
        tensor_path = reference_report.parent / descriptor["file"]
        require(
            tensor_path.resolve().parent == reference_report.parent.resolve(),
            "reference artifact escapes directory",
        )
        artifact_digest = sha256(tensor_path)
        require(
            tensor_path.stat().st_size == descriptor["size_bytes"]
            and artifact_digest == descriptor["sha256"],
            "reference artifact changed",
        )
        paths.update(reference_tensors=tensor_path, capture=capture)
        before.update(reference_tensors=artifact_digest, capture=sha256(capture))
        report["inputs"] = {
            name: {"path": str(path.resolve()), "sha256": before[name]}
            for name, path in paths.items()
        }
        import torch
        from safetensors import safe_open

        torch.set_num_threads(threads)
        report["torch"] = torch.__version__
        report["threads"] = threads
        with safe_open(tensor_path, framework="pt", device="cpu") as source:
            require(
                set(source.keys()) == set(REFERENCE_NAMES), "wrong reference tensors"
            )
            expected = {name: source.get_tensor(name) for name in REFERENCE_NAMES}
        with safe_open(capture, framework="pt", device="cpu") as source:
            require(set(source.keys()) == set(CAPTURE_NAMES), "wrong capture tensors")
            captured = {name: source.get_tensor(name) for name in CAPTURE_NAMES}
        linear.validate_tokens(frozen["token_ids"])
        require(
            expected["input"].shape[0] == len(frozen["token_ids"]),
            "reference token/row count differs",
        )
        require(
            frozen["tensors"]
            == {
                name: {"shape": list(t.shape), "dtype": str(t.dtype)}
                for name, t in expected.items()
            },
            "reference tensor metadata differs",
        )
        report["errors"] = projection_errors(expected, captured, torch)
        report["stored_operand_bits_equal"] = True
        report["token_ids"] = frozen["token_ids"]
        report["output_shape"] = list(captured["output"].shape)
        report["output_dtype"] = str(captured["output"].dtype)
        require(
            before == {name: sha256(path) for name, path in paths.items()},
            "comparison input changed",
        )
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    compare(args.reference_report, args.capture, args.output, args.threads)


if __name__ == "__main__":
    main()
