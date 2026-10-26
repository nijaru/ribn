#!/usr/bin/env python3
"""Export expected first PLE-projection operands and a binary64 diagnostic.

Uses an existing pinned package/runtime, CPU only. This is not captured device
state, a BF16 inference oracle, or an acceptance budget for reordered arithmetic.
"""

import argparse
import json
import math
import platform
import time
import tomllib
from pathlib import Path

import gemma4_reference as reference
from gemma4_reference import require, sha256, write_json

MAX_ROWS = 4
PREFIX = "model.language_model."
EMBEDDING = PREFIX + "embed_tokens.weight"
PROJECTION = PREFIX + "per_layer_model_projection.weight"


def validate_tokens(tokens, vocab=None):
    require(1 <= len(tokens) <= MAX_ROWS, "expected one to four token IDs")
    require(
        all(type(token) is int and token >= 0 for token in tokens),
        "token IDs must be nonnegative integers, not bools",
    )
    if vocab is not None:
        require(all(token < vocab for token in tokens), "token ID outside vocabulary")


def fsum_linear(inputs, weights):
    """Return row-major X @ W.T and sum(abs(products)), without BLAS.

    The collector supplies finite, BF16-exact Python floats. Their products are
    exact in binary64. math.fsum is still a binary64 numerical reference, not a
    proof of exact real-number rounding or a BF16 output-rounding oracle.
    Stream weight rows to avoid materializing the matrix as Python floats.
    """
    require(inputs and inputs[0], "empty linear input")
    width = len(inputs[0])
    require(all(len(row) == width for row in inputs), "ragged linear input")
    result = [[] for _ in inputs]
    absolute = [[] for _ in inputs]
    count = 0
    for weight in weights:
        require(len(weight) == width, "wrong linear weight width")
        for row, values in enumerate(inputs):
            products = [x * w for x, w in zip(values, weight, strict=True)]
            result[row].append(math.fsum(products))
            absolute[row].append(math.fsum(map(abs, products)))
        count += 1
    require(count > 0, "empty linear weight")
    return result, absolute


def collect(snapshot, output, token_ids, threads):
    snapshot, output = snapshot.resolve(), output.resolve()
    require(not output.is_relative_to(snapshot), "output must be outside package")
    output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "complete": False,
        "snapshot": str(snapshot),
        "token_ids": token_ids,
        "python": platform.python_version(),
        "threads": threads,
        "operation": "input @ weight.T, before PLE projection scale and norm",
        "reference_math": "exact BF16 products in binary64; math.fsum accumulation",
        "conditioning": "sum_abs_products is descriptive, not an error budget",
        "actual_device_operand_capture_verified": False,
        "native_execution_verified": False,
        "numerical_acceptance": "none",
        "generator_sha256": {
            path.name: sha256(path)
            for path in (Path(__file__), Path(reference.__file__), reference.MANIFEST)
        },
    }
    started = time.monotonic()
    report_path = output / "report.json"
    write_json(report_path, report)
    try:
        validate_tokens(token_ids)
        manifest = tomllib.loads(reference.MANIFEST.read_text())
        report["checkpoint"] = manifest["checkpoint"]
        report["package_before"] = reference.verify_files(snapshot, manifest["files"])
        config = json.loads((snapshot / "config.json").read_text())["text_config"]
        vocab, hidden = config["vocab_size"], config["hidden_size"]
        packed = config["num_hidden_layers"] * config["hidden_size_per_layer_input"]
        validate_tokens(token_ids, vocab)
        torch = reference.prepare_cpu_reference(report, manifest["reference"], threads)
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextScaledWordEmbedding,
        )

        with safe_open(
            snapshot / manifest["checkpoint"]["weights"], framework="pt", device="cpu"
        ) as source:
            embedding = source.get_slice(EMBEDDING)
            require(embedding.get_shape() == [vocab, hidden], "wrong embedding shape")
            rows = torch.cat([embedding[token : token + 1] for token in token_ids])
            weight = source.get_tensor(PROJECTION)
            require(tuple(weight.shape) == (packed, hidden), "wrong projection shape")
            require(
                rows.dtype == weight.dtype == torch.bfloat16, "expected BF16 parameters"
            )
            tensors = {}
            reference.retain(tensors, "embedding_rows", rows)
            reference.retain(tensors, "weight", weight)
        # Compact lookup has the same selected rows and pinned scale operation;
        # local indices are not retokenization or a replacement model vocabulary.
        module = Gemma4TextScaledWordEmbedding(
            len(token_ids), hidden, None, embed_scale=hidden**0.5
        ).to(device="cpu", dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.copy_(tensors["embedding_rows"])
            inputs = module(torch.arange(len(token_ids), device="cpu"))
        require(tuple(inputs.shape) == (len(token_ids), hidden), "wrong input shape")
        require(inputs.dtype == torch.bfloat16, "expected BF16 scaled inputs")
        reference.retain(tensors, "input", inputs)
        report["embedding_scale_bf16"] = module.embed_scale.item()
        values, absolute = fsum_linear(
            tensors["input"].float().tolist(),
            (row.float().tolist() for row in tensors["weight"]),
        )
        for name, data in (("reference", values), ("sum_abs_products", absolute)):
            tensor = torch.tensor(data, dtype=torch.float64, device="cpu")
            require(
                tuple(tensor.shape) == (len(token_ids), packed), "wrong reference shape"
            )
            reference.retain(tensors, name, tensor)
        report["source_parameters"] = {
            "embedding_rows": EMBEDDING,
            "weight": PROJECTION,
        }
        report["tensors"] = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        }
        path = output / "projection.safetensors"
        save_file(tensors, path)
        report["artifact"] = {
            "file": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        report["package_after"] = reference.verify_files(snapshot, manifest["files"])
        require(report["package_before"] == report["package_after"], "package changed")
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
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-ids", type=int, nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    collect(args.snapshot, args.output, args.token_ids, args.threads)


if __name__ == "__main__":
    main()
