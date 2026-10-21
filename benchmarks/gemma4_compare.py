#!/usr/bin/env python3
"""Compare one completed Gemma FP32 capture with the archived CPU BF16 oracle.

Needs NumPy, not Torch/SGLang. Hashes establish artifact consistency, not execution
attestation. Diagnostics do not select a numerical budget or establish task quality.
"""

import argparse
import json
import statistics
import struct
import tomllib
from itertools import pairwise
from pathlib import Path

import numpy as np
from gemma4_reference import MANIFEST, require, sha256
from gemma4_sglang import VOCAB, reference_cases, timing_cases


def metrics(observed, reference):
    require(
        observed.dtype == reference.dtype == np.dtype("float32"),
        "expected decoded FP32 rows",
    )
    require(
        observed.shape == reference.shape and observed.ndim == 1 and observed.size > 0,
        "different or empty logit shapes",
    )
    require(
        bool(np.isfinite(observed).all() and np.isfinite(reference).all()),
        "nonfinite logits",
    )
    a, b = observed.astype(np.float64), reference.astype(np.float64)
    delta = a - b
    lp, lq = b - b.max(), a - a.max()
    lp -= np.log(np.exp(lp).sum())
    lq -= np.log(np.exp(lq).sum())
    p, q = np.exp(lp), np.exp(lq)
    norm = float(np.linalg.norm(b))
    return {
        "fp32_bits_equal": bool(
            np.array_equal(observed.view(np.uint32), reference.view(np.uint32))
        ),
        "winner_equal": int(a.argmax()) == int(b.argmax()),
        "max_abs": float(np.abs(delta).max()),
        "centered_rmse": float(np.sqrt(np.mean((delta - delta.mean()) ** 2))),
        "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
        "kl_reference_to_observed": float(np.sum(p * (lp - lq))),
        "tv": float(np.abs(p - q).sum() / 2),
    }


def reference_rows(path, reference):
    descriptor = reference["tensor_file"]
    size = path.stat().st_size
    require(size == descriptor["size_bytes"], "wrong reference tensor file size")
    require(sha256(path) == descriptor["sha256"], "wrong reference tensor file hash")
    rows = {}
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, "truncated safetensors header length")
        length = struct.unpack("<Q", length_bytes)[0]
        require(0 < length <= min(size - 8, 1 << 20), "invalid safetensors header size")
        header = json.loads(stream.read(length))
        ranges = []
        for case in reference["cases"]:
            for step in case["steps"]:
                name = f"{case['name']}.{step['index']}"
                spec = header[name + ".logits"]
                require(
                    spec["dtype"] == "BF16" and spec["shape"] == [VOCAB],
                    "expected a complete BF16 reference row",
                )
                start, end = spec["data_offsets"]
                require(
                    type(start) is int
                    and type(end) is int
                    and 0 <= start < end <= size - 8 - length
                    and end - start == VOCAB * 2,
                    "invalid reference tensor range",
                )
                ranges.append((start, end))
                stream.seek(8 + length + start)
                bits = np.frombuffer(stream.read(end - start), dtype="<u2")
                require(bits.shape == (VOCAB,), "truncated reference row")
                row = (bits.astype(np.uint32) << 16).view(np.float32)
                require(bool(np.isfinite(row).all()), "nonfinite reference row")
                require(int(row.argmax()) == step["chosen"], "wrong reference winner")
                rows[name] = row
        ordered = sorted(ranges)
        require(
            all(left[1] <= right[0] for left, right in pairwise(ordered)),
            "overlapping reference logit rows",
        )
    return rows


def captured_row(directory, descriptor):
    path = directory / descriptor["path"]
    require(
        path.resolve().parent == directory.resolve() and path.is_file(),
        "logit file escapes capture directory or is missing",
    )
    require(
        descriptor["dtype"] == "float32-le" and descriptor["elements"] == VOCAB,
        "wrong captured logit encoding",
    )
    require(path.stat().st_size == VOCAB * 4, "wrong captured logit byte count")
    require(sha256(path) == descriptor["sha256"], "wrong captured logit hash")
    values = np.fromfile(path, dtype="<f4")
    require(bool(np.isfinite(values).all()), "nonfinite captured logits")
    return values


def compare(report_path, source, reference_path, tensors):
    manifest = tomllib.loads(MANIFEST.read_text())
    reference = json.loads(reference_path.read_text())
    cases = reference_cases(reference, manifest)
    require(
        reference["manifest_sha256"] == sha256(MANIFEST), "wrong reference manifest"
    )
    require(
        reference["packages"]
        == {key: manifest["reference"][key] for key in reference["packages"]}
        and set(reference["packages"])
        == {"torch", "transformers", "safetensors", "tokenizers"},
        "wrong CPU reference packages",
    )
    require(
        reference["implementation"]["declared_upstream_commit"]
        == manifest["reference"]["transformers_commit"]
        and reference["implementation"]["verified_file_sha256"]
        == manifest["reference"]["source_sha256"],
        "wrong CPU reference implementation",
    )
    require(reference["files"] == manifest["files"], "wrong reference checkpoint files")
    report = json.loads(report_path.read_text())
    require(
        report["complete"] is True
        and report.get("validation_only") is False
        and "error" not in report,
        "capture is incomplete, failed or validation-only",
    )
    require(
        report["clock"]
        == "input preparation through completed full FP32 host logits; no sampling",
        "wrong capture clock boundary",
    )
    require(report["script_sha256"] == sha256(source), "wrong capture source")
    require(
        report["reference_sha256"] == sha256(reference_path), "wrong reference identity"
    )
    require(report["manifest_sha256"] == sha256(MANIFEST), "wrong capture manifest")
    require(report["checkpoint"] == manifest["checkpoint"], "wrong captured checkpoint")
    require(report["files"] == manifest["files"], "wrong captured checkpoint files")
    timing = timing_cases(cases)
    plan = {"quality": cases, "timing": timing, "trials": 4, "warmup_trial": 0}
    require(report["plan"] == plan, "capture does not use the common plan")
    expected = [(case, None) for case in cases] + [
        (case, trial) for case in timing for trial in range(4)
    ]
    require(len(report["runs"]) == len(expected), "missing or extra capture sequences")
    rows = reference_rows(tensors, reference)
    quality = []
    measured = {case["name"]: {"prefill_ns": [], "decode_ns": []} for case in timing}
    for run, (case, trial) in zip(report["runs"], expected, strict=True):
        require(
            run["complete"] is True
            and run["name"] == case["name"]
            and run["trial"] == trial
            and (run["trial"] is None or type(run["trial"]) is int)
            and run["warmup"] is (trial == 0)
            and len(run["steps"]) == len(case["inputs"]),
            "incomplete, reordered or wrongly classified sequence",
        )
        position = 0
        for index, (record, ids) in enumerate(
            zip(run["steps"], case["inputs"], strict=True)
        ):
            position += len(ids)
            require(
                type(record["step"]) is int
                and record["step"] == index
                and all(type(token) is int for token in record["input_ids"])
                and record["input_ids"] == ids
                and type(record["history_length"]) is int
                and record["history_length"] == position
                and all(type(value) is int for value in record["position_ids"])
                and record["position_ids"]
                == list(range(position - len(ids), position)),
                "captured input history or positions differ",
            )
            require(
                record["shape"] in ([1, VOCAB], [VOCAB])
                and record["dtype"] in ("torch.float32", "float32")
                and record["finite"] is True,
                "expected a finite complete FP32 host row",
            )
            require(
                type(record["elapsed_ns"]) is int and record["elapsed_ns"] > 0,
                "invalid completed sample clock",
            )
            require(
                type(record["chosen"]) is int and 0 <= record["chosen"] < VOCAB,
                "invalid captured winner",
            )
            if trial is None:
                name = f"{case['name']}.{index}"
                values = captured_row(report_path.parent, record["logits"])
                require(
                    int(values.argmax()) == record["chosen"], "wrong captured winner"
                )
                quality.append({"name": name, **metrics(values, rows[name])})
            elif trial != 0:
                measured[case["name"]][
                    "prefill_ns" if index == 0 else "decode_ns"
                ].append(record["elapsed_ns"])
    return {
        "scope": "12 common histories; one-process direct-call clocks, not serving/task quality",
        "analysis_script_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "capture_sha256": sha256(report_path),
        "capture_source_sha256": sha256(source),
        "reference_sha256": sha256(reference_path),
        "reference_tensor_sha256": sha256(tensors),
        "clock": report["clock"],
        "quality": quality,
        "summary": {
            "rows": len(quality),
            "fp32_bits_equal": sum(row["fp32_bits_equal"] for row in quality),
            "winners_equal": sum(row["winner_equal"] for row in quality),
            "max_over_rows": {
                metric: max(row[metric] for row in quality)
                for metric in (
                    "max_abs",
                    "centered_rmse",
                    "kl_reference_to_observed",
                    "tv",
                )
            },
        },
        "timing": {
            name: {
                key: {
                    "samples": len(samples),
                    "median_ms": statistics.median(samples) / 1e6,
                }
                for key, samples in groups.items()
            }
            for name, groups in measured.items()
        },
        "warmups": "trial zero retained in capture and excluded from these medians",
        "acceptance_budget": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite evidence")
    result = compare(args.report, args.source, args.reference, args.tensors)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
