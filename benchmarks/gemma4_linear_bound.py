#!/usr/bin/env python3
"""Certify a stored E4B BF16 projection against the fixed FP32/BF16 bound.

This is a component gate, not full-model or device-execution attestation. See
numerical assumptions and the enclosure proof in gemma4-cuda-baseline.md.
Requires NumPy; inputs already exist locally. No model loading or code execution.
"""

import argparse
import hashlib
import json
import math
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
from gemma4_reference import require, sha256

GEOMETRIES = {
    (2560, 10752),
    (2560, 2048),
    (2560, 512),
    (2560, 4096),
    (2560, 1024),
    (2560, 2560),
    (2560, 10240),
    (2560, 256),
    (2048, 2560),
    (4096, 2560),
    (10240, 2560),
    (256, 2560),
    (2560, 262144),
}
ROLES = {"ple", "q", "k", "v", "head"}
CHUNK_COLUMNS = 64
MAX_K = 10240
ZERO_QUANTUM = 32767


def encoding_tables():
    """Exact binary64 values and smallest dyadic units, not encoding ULPs.

    Never pass through float32: a host's FP32 denormal mode must not erase a
    BF16 subnormal before the safe-domain check or reference enclosure.
    """
    quantum = np.full(65536, ZERO_QUANTUM, dtype=np.int16)
    decoded = np.full(65536, np.nan, dtype=np.float64)
    for bits in range(65536):
        exponent, fraction = (bits >> 7) & 255, bits & 127
        if exponent == 255:
            continue
        mantissa = 128 + fraction if exponent else fraction
        base = exponent - 134 if exponent else -133
        value = math.ldexp(float(mantissa), base)
        decoded[bits] = -value if bits & 0x8000 else value
        if mantissa:
            quantum[bits] = base + (mantissa & -mantissa).bit_length() - 1
    return quantum, decoded


QUANTUM, DECODED = encoding_tables()


def down(value):
    return np.nextafter(value, -np.inf)


def up(value):
    return np.nextafter(value, np.inf)


def gamma_bounds(k, precision):
    exact = Fraction(k, 2**precision - k)
    return float(down(float(exact))), float(up(float(exact)))


def values(bits):
    return DECODED[bits]


def validate_bits(bits):
    require(
        bits.dtype.kind == "u" and bits.dtype.itemsize == 2, "expected BF16 encodings"
    )
    require(bool(np.all((bits & 0x7F80) != 0x7F80)), "nonfinite BF16 encoding")


def normal_operands(bits):
    return ((bits & 0x7F80) != 0) | ((bits & 0x7FFF) == 0)


def dot_enclosure(x, weight):
    """Binary64 enclosure for one encoded input row against weight rows.

    BF16 products are exact normal binary64 values. Even the smallest nonzero
    product quantum (2^-266) and the largest possible sum for K<=10240 stay
    normal/finite in binary64. Thus any reduction order has |d_hat-d|<=gamma*S.
    Explicit float64 products/sum avoid assumptions about a BLAS accumulator.
    """
    require(
        x.ndim == 1 and weight.ndim == 2 and weight.shape[1] == x.size,
        "wrong dot shapes",
    )
    require(0 < x.size <= MAX_K and weight.shape[0] > 0, "unsupported dot extent")
    validate_bits(x)
    validate_bits(weight)
    products = values(weight) * values(x)
    estimate = np.sum(products, axis=1, dtype=np.float64)
    magnitude = np.sum(np.abs(products), axis=1, dtype=np.float64)
    _, gamma_hi = gamma_bounds(x.size, 52)
    # Each intermediate operation is directed outward by one nextafter step.
    sum_lo = np.maximum(0.0, down(magnitude / up(1.0 + gamma_hi)))
    sum_hi = up(magnitude / down(1.0 - gamma_hi))
    reference_error = up(gamma_hi * sum_hi)
    dot_lo, dot_hi = down(estimate - reference_error), up(estimate + reference_error)
    zero = magnitude == 0
    # Positive products cannot underflow in binary64, so zero here proves S=d=0.
    sum_lo[zero] = sum_hi[zero] = dot_lo[zero] = dot_hi[zero] = 0
    quantum = int(QUANTUM[x].min()) + QUANTUM[weight].min(axis=1).astype(np.int32)
    normal = bool(np.all(normal_operands(x))) & np.all(normal_operands(weight), axis=1)
    return dot_lo, dot_hi, sum_lo, sum_hi, quantum, normal


def classify(x, weight, output):
    """Return sufficient-pass / definite-violation / indeterminate certificates.

    The lower bound on the allowed error, not its optimistic upper bound, must
    cover the worst error over the reference interval. Unsupported safe domains
    never pass. These classifications do not certify an implementation's internals.
    """
    validate_bits(output)
    dot_lo, dot_hi, sum_lo, sum_hi, quantum, normal = dot_enclosure(x, weight)
    require(output.shape == dot_lo.shape, "wrong output shape")
    gamma_lo, gamma_hi = gamma_bounds(x.size, 23)
    accumulated_lo = np.maximum(0.0, down(gamma_lo * sum_lo))
    accumulated_hi = up(gamma_hi * sum_hi)
    abs_dot_lo = np.where(
        (dot_lo <= 0) & (dot_hi >= 0),
        0.0,
        np.maximum(0.0, down(np.minimum(np.abs(dot_lo), np.abs(dot_hi)))),
    )
    abs_dot_hi = up(np.maximum(np.abs(dot_lo), np.abs(dot_hi)))
    bound_lo = np.maximum(
        0.0, down(accumulated_lo + down(2.0**-7 * down(abs_dot_lo + accumulated_lo)))
    )
    bound_hi = up(accumulated_hi + up(2.0**-7 * up(abs_dot_hi + accumulated_hi)))
    actual = values(output)
    error_hi = up(np.maximum(np.abs(actual - dot_lo), np.abs(actual - dot_hi)))
    error_lo = np.where(
        (actual >= dot_lo) & (actual <= dot_hi),
        0.0,
        np.maximum(
            0.0, down(np.minimum(np.abs(actual - dot_lo), np.abs(actual - dot_hi)))
        ),
    )
    # A common integer grid >=2^-126 excludes every nonzero subnormal FP32
    # product/partial sum. Rounding preserves that grid. S<2^127, K<=10240 and
    # gamma_32<1/2 also exclude FP32 accumulation and BF16 conversion overflow.
    safe = normal & (quantum >= -126) & (sum_hi < 2.0**127)
    zero = sum_hi == 0
    safe |= zero
    error_hi[zero] = np.abs(actual[zero])
    error_lo[zero] = np.abs(actual[zero])
    bound_lo[zero] = bound_hi[zero] = 0
    accepted = safe & (error_hi <= bound_lo)
    violated = safe & (error_lo > bound_hi)
    indeterminate = safe & ~accepted & ~violated
    return {
        "accepted": accepted,
        "violated": violated,
        "indeterminate": indeterminate,
        "unsupported": ~safe,
        "dot_lower": dot_lo,
        "dot_upper": dot_hi,
        "error_upper": error_hi,
        "bound_lower": bound_lo,
        "bound_upper": bound_hi,
        "quantum": quantum,
        "sum_upper": sum_hi,
    }


def certify(x, weight, output, chunk_columns=CHUNK_COLUMNS):
    require(
        x.ndim == weight.ndim == output.ndim == 2
        and x.shape[0] in (1, 4)
        and 0 < x.shape[1] <= MAX_K
        and weight.shape[1] == x.shape[1]
        and weight.shape[0] > 0
        and output.shape == (x.shape[0], weight.shape[0]),
        "wrong projection shapes",
    )
    require(
        type(chunk_columns) is int and 0 < chunk_columns <= CHUNK_COLUMNS,
        "invalid chunk size",
    )
    counts = {
        key: 0 for key in ("accepted", "violated", "indeterminate", "unsupported")
    }
    result = {
        "counts": counts,
        "max_error_upper": 0.0,
        "max_sum_upper": 0.0,
        "exceptions": [],
    }
    for start in range(0, weight.shape[0], chunk_columns):
        stop = min(start + chunk_columns, weight.shape[0])
        for row in range(x.shape[0]):
            certificate = classify(x[row], weight[start:stop], output[row, start:stop])
            for name in counts:
                counts[name] += int(certificate[name].sum())
            result["max_error_upper"] = max(
                result["max_error_upper"], float(certificate["error_upper"].max())
            )
            result["max_sum_upper"] = max(
                result["max_sum_upper"], float(certificate["sum_upper"].max())
            )
            for column in np.flatnonzero(~certificate["accepted"]):
                if len(result["exceptions"]) == 16:
                    break
                status = next(name for name in counts if certificate[name][column])
                result["exceptions"].append(
                    {
                        "row": row,
                        "column": start + int(column),
                        "status": status,
                        **{
                            name: float(certificate[name][column])
                            for name in (
                                "dot_lower",
                                "dot_upper",
                                "error_upper",
                                "bound_lower",
                                "bound_upper",
                                "sum_upper",
                            )
                        },
                        "quantum_exponent": int(certificate["quantum"][column]),
                    }
                )
    result["component_bound_acceptance"] = counts["accepted"] == output.size
    result["output_shape"] = list(output.shape)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_capture(path):
    with path.open("rb") as stream:
        payload = stream.read(65537)
    require(len(payload) <= 65536, "oversized capture metadata")
    metadata = json.loads(payload, object_pairs_hook=unique_object)
    index, role = metadata["index"], metadata["role"]
    require(
        type(index) is int and 0 <= index < 20 and role in ROLES,
        "unknown capture index/role",
    )
    require(path.name == f"{index:02}-{role}.json", "wrong capture filename")
    require(metadata["dtype"] == "BF16", "wrong capture dtype")
    shapes = [
        metadata[name] for name in ("input_shape", "weight_shape", "output_shape")
    ]
    require(
        all(
            isinstance(s, list)
            and len(s) == 2
            and all(type(n) is int and n > 0 for n in s)
            for s in shapes
        ),
        "invalid capture dimensions",
    )
    (m, k), (n, wk), ys = shapes
    require(
        m in (1, 4) and wk == k and ys == [m, n] and (k, n) in GEOMETRIES,
        "unsupported capture geometry",
    )
    require(
        metadata.get("acknowledged_native_return") is True,
        "capture declares no completion",
    )
    paths = {
        "input": path.with_name(f"{index:02}-{role}-input.bf16"),
        "weight": path.with_name(f"weight-{role}.bf16"),
        "output": path.with_name(f"{index:02}-{role}-output.bf16"),
    }
    tensors, descriptors = {}, {}
    for (name, tensor_path), shape in zip(paths.items(), shapes, strict=True):
        tensor_path = tensor_path.resolve(strict=True)
        require(
            tensor_path.stat().st_size == shape[0] * shape[1] * 2,
            f"wrong capture byte count: {name}",
        )
        descriptors[name] = {
            "path": str(tensor_path.resolve()),
            "sha256": sha256(tensor_path),
        }
        tensors[name] = np.memmap(
            tensor_path, mode="r", dtype="<u2", shape=tuple(shape)
        )
    descriptors["metadata"] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return tensors, descriptors


def verify(capture, output):
    output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "complete": False,
        "component_bound_acceptance": False,
        "full_model_acceptance": False,
        "device_execution_attested": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "bound": "gamma_K(2^-23)*S + 2^-7*(abs(dot)+gamma_K(2^-23)*S)",
        "reference": "binary64 enclosure, gamma_K(2^-52), outward nextafter",
    }
    sources = [Path(__file__), Path(__file__).with_name("gemma4_reference.py")]
    try:
        source_before = {p.name: sha256(p) for p in sources}
        tensors, report["inputs"] = read_capture(capture)
        verdict = certify(tensors["input"], tensors["weight"], tensors["output"])
        require(
            all(
                sha256(Path(d["path"])) == d["sha256"]
                for d in report["inputs"].values()
            ),
            "capture changed during verification",
        )
        require(
            source_before == {p.name: sha256(p) for p in sources},
            "verifier source changed",
        )
        report.update(verdict, source_sha256=source_before, complete=True)
    except Exception as error:
        report.update(complete=False, component_bound_acceptance=False)
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.capture, args.output)
    raise SystemExit(0 if report["component_bound_acceptance"] else 1)


if __name__ == "__main__":
    main()
