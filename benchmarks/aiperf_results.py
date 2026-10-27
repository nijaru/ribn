"""Offline protocol, usage, prompt-cohort and selected clock audit of AIPerf v0.13."""

import hashlib
import json
from collections import Counter
from itertools import pairwise
from statistics import median

from aiperf_stream import completion_stream, timestamp, validate_clocks
from endpoint_config import (
    AIPERF_VERSION,
    check_config,
    number,
    parse_manifest,
    require,
)

EXPORT_SCHEMA = "1.4"


def read_json(path):
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected a JSON object in {path.name}")
    return value


def records(path):
    result = {}
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            metadata = record["metadata"]
            key = metadata["x_request_id"]
            require(
                isinstance(key, str) and key and key not in result,
                f"missing/duplicate request ID in {path.name}",
            )
            require(
                metadata["benchmark_phase"] in ("warmup", "profiling"),
                "unexpected benchmark phase",
            )
            result[key] = record
    return result


def metric(record, key, field="value", unit="tokens"):
    require(isinstance(record, dict), "invalid metrics object")
    value = record.get(key, {})
    require(
        isinstance(value, dict)
        and value.get("unit") == unit
        and number(value.get(field)),
        f"missing/invalid metric {key}",
    )
    return value[field]


def audit_timing_metrics(metrics, timing, completion_tokens):
    # Match v0.13's internal ns arithmetic, then its display-unit conversion.
    expected = {
        "time_to_first_token": timing["first_content"] * 1e-6,
        "request_latency": timing["last_content"] * 1e-6,
    }
    if completion_tokens >= 2:
        # Exact effective endpoint settings disable per-chunk usage. AIPerf
        # assumes one token in the first content chunk; these are not token taps.
        expected["inter_token_latency"] = (
            (timing["last_content"] - timing["first_content"]) / (completion_tokens - 1)
        ) * 1e-6
    else:
        require(
            "inter_token_latency" not in metrics,
            "inter_token_latency must be absent below two output tokens",
        )
    for name, value in expected.items():
        require(
            metric(metrics, name, unit="ms") == value,
            f"raw clock/metric mismatch: {name}",
        )


def issued_load(raw, profiling, workload):
    """Actual single-host starts, never the rate limiter's unrecorded arrival plan."""
    starts = sorted(timestamp(raw[key].get("start_perf_ns")) for key in profiling)
    gaps = [right - left for left, right in pairwise(starts)]
    return {
        "configured_request_rate": workload["request_rate"],
        "max_concurrency": workload["max_concurrency"],
        "issued_requests": len(starts),
        "start_span_ns": starts[-1] - starts[0],
        "start_interval_ns": {
            "min": min(gaps) if gaps else None,
            "median": median(gaps) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
    }


def prompt_cohort(raw, profiling):
    """Identify the decoded UTF-8 prompt multiset, not an arrival schedule."""
    counts = Counter()
    unavailable = []
    for key in profiling:
        payload = raw[key].get("payload")
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        if not isinstance(prompt, str):
            unavailable.append(key)
            continue
        try:
            counts[hashlib.sha256(prompt.encode("utf-8")).hexdigest()] += 1
        except UnicodeEncodeError:
            unavailable.append(key)
    encoded = json.dumps(sorted(counts.items()), separators=(",", ":")).encode("ascii")
    return {
        "algorithm": "sha256-utf8-prompt-multiset-v1",
        # Never present a partial digest as the identity of all attempted work.
        "sha256": None
        if unavailable
        else hashlib.sha256(
            b"ribn:profiling-prompt-multiset:v1\0" + encoded
        ).hexdigest(),
        "requests": len(profiling),
        "unavailable_request_ids": sorted(unavailable),
    }


def validate(directory):
    snapshot = (directory / "manifest.toml").read_bytes()
    manifest = parse_manifest(snapshot)
    require(
        (directory / "client-exit-code").read_text().strip() == "0",
        "AIPerf process did not succeed",
    )
    provenance = read_json(directory / "invocation.json")
    digest = hashlib.sha256(snapshot).hexdigest()
    require(
        provenance.get("manifest_sha256") == digest, "manifest changed after invocation"
    )
    client = directory / "client"
    summary = read_json(client / "profile_export_aiperf.json")
    require(
        summary.get("aiperf_version") == AIPERF_VERSION
        and summary.get("schema_version") == EXPORT_SCHEMA,
        "unsupported AIPerf version/export schema",
    )
    require(
        summary.get("is_complete") is True and summary.get("was_cancelled") is False,
        "AIPerf run was incomplete/cancelled",
    )
    check_config(summary.get("input_config"), manifest)
    ordinary = records(client / "profile_export.jsonl")
    raw = records(client / "profile_export_raw.jsonl")
    require(ordinary.keys() == raw.keys(), "ordinary/raw request IDs differ")
    expected = manifest["workload"]["request_count"]
    profiling = [
        key
        for key, record in ordinary.items()
        if record["metadata"]["benchmark_phase"] == "profiling"
    ]
    require(
        len(profiling) == expected,
        f"expected {expected} profiling records, got {len(profiling)}",
    )
    require(
        len(ordinary) - len(profiling) == manifest["workload"]["warmup_requests"],
        "warmup count differs",
    )
    load = issued_load(raw, profiling, manifest["workload"])
    timings = {}
    failures = []
    invalid = []
    warmup_errors = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    successful = 0
    for key, record in ordinary.items():
        request = raw[key]
        require(record["metadata"] == request["metadata"], f"metadata mismatch: {key}")
        phase = record["metadata"]["benchmark_phase"]
        if record.get("error") is not None:
            failure = {"request_id": key, "error": record["error"]}
            if phase == "profiling":
                failures.append(failure)
            else:
                warmup_errors.append(failure)
            try:
                validate_clocks(request)
            except (ValueError, KeyError, TypeError) as error:
                if phase == "profiling":
                    invalid.append({"request_id": key, "error": str(error)})
                else:
                    failure["clock_error"] = str(error)
            continue
        try:
            require(
                not record["metadata"].get("was_cancelled")
                and not request.get("error"),
                "request was cancelled/failed",
            )
            require(request.get("status") == 200, "request did not return HTTP 200")
            usage, timing = completion_stream(request, manifest)
            audit_timing_metrics(record["metrics"], timing, usage["completion_tokens"])
            for field, value in usage.items():
                if field in totals:
                    require(
                        metric(record["metrics"], f"usage_{field}") == value,
                        f"usage metric mismatch: {field}",
                    )
            require(
                metric(record["metrics"], "output_sequence_length")
                == usage["completion_tokens"],
                "OSL/usage mismatch",
            )
            if phase == "profiling":
                timings[key] = timing
                successful += 1
                for field in totals:
                    totals[field] += usage[field]
        except (ValueError, KeyError, TypeError) as error:
            target = invalid if phase == "profiling" else warmup_errors
            target.append({"request_id": key, "error": str(error)})
    native_success = metric(summary, "request_count", "avg", "requests")
    native_errors = (
        metric(summary, "error_request_count", "avg", "requests")
        if "error_request_count" in summary
        else 0
    )
    require(
        native_success + native_errors == expected and native_errors == len(failures),
        "aggregate request counts differ",
    )
    if not invalid:
        require(native_success == successful, "aggregate success count differs")
        for field, value in totals.items():
            if successful:
                require(
                    metric(summary, f"total_usage_{field}", "avg") == value,
                    f"aggregate usage mismatch: {field}",
                )
    return {
        "protocol_valid": not invalid and not warmup_errors and successful > 0,
        "quality_scope": manifest["run"]["quality_scope"],
        "attempted_requests": expected,
        "planned_arrivals_audited": False,
        "issued_load": load,
        "prompt_cohort": prompt_cohort(raw, profiling),
        "audited_request_timing_metrics": [
            "time_to_first_token",
            "request_latency",
            "inter_token_latency",
        ],
        "request_timing_ns": timings,
        "timing_source": "client/profile_export_raw.jsonl",
        "successful_requests": successful,
        "failed_requests": len(failures),
        "invalid_requests": invalid,
        "warmup_errors": warmup_errors,
        "failures": failures,
        **totals,
        "slo": manifest.get("slo", {}),
        "metrics_source": "client/profile_export_aiperf.json",
        "performance_metrics_audited": False,
        "readiness": "Automatic probes disabled; endpoint must already be ready. Zero warmup does not prove cold state.",
        "limitations": "Protocol/usage, prompt multiset and specified raw-clock/usage-ratio audit only; not observed token intervals, aggregate performance, planned arrivals, HTTP EOF, numerical, GPU or physical-memory qualification.",
    }
