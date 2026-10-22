"""Offline protocol/accounting audit of AIPerf v0.13 native exports."""

import hashlib
import json

from endpoint_config import (
    AIPERF_VERSION,
    check_config,
    count,
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


def stream_usage(raw, manifest):
    """Validate one-choice OpenAI completion terminal order, not SSE wire framing."""
    payload = raw["payload"]
    require(isinstance(payload, dict), "missing request payload")
    require(
        payload.get("model") == manifest["model"]["id"]
        and payload.get("stream") is True,
        "request model/stream differs from manifest",
    )
    require(
        number(payload.get("temperature"))
        and payload["temperature"] == 0
        and type(payload.get("n", 1)) is int
        and payload.get("n", 1) == 1,
        "request is not one-choice greedy generation",
    )
    cap = manifest["workload"]["output_tokens_mean"]
    require(
        type(payload.get("max_tokens")) is int and payload["max_tokens"] == cap,
        "request output cap differs from manifest",
    )
    require(
        payload.get("ignore_eos", False) is False
        and payload.get("min_tokens", 0) == 0
        and payload.get("min_new_tokens", 0) == 0,
        "request bypasses EOS",
    )
    options = payload.get("stream_options")
    require(
        isinstance(options, dict) and options.get("include_usage") is True,
        "usage was not requested",
    )
    finish = None
    usage = None
    done = False
    for response in raw["responses"]:
        packets = response["packets"]
        require(
            isinstance(packets, list) and all(isinstance(p, dict) for p in packets),
            "invalid SSE packets",
        )
        require(
            not any(
                p.get("name") == "event" and p.get("value") == "error" for p in packets
            ),
            "SSE error event",
        )
        values = [p["value"] for p in packets if p["name"] == "data"]
        if not values:
            continue
        require(
            len(values) == 1 and not done,
            "unexpected data after [DONE] or multiple data packets",
        )
        data = values[0]
        if data == "[DONE]":
            require(
                finish is not None and usage is not None, "[DONE] before finish/usage"
            )
            done = True
            continue
        event = json.loads(data)
        require(isinstance(event, dict) and "error" not in event, "JSON error response")
        require(
            event.get("object") in ("text_completion", "completion"),
            "unexpected completion object",
        )
        choices = event.get("choices")
        require(
            isinstance(choices, list) and len(choices) <= 1,
            "invalid completion choices",
        )
        if choices:
            require(
                finish is None and usage is None, "choice after terminal finish/usage"
            )
            choice = choices[0]
            require(isinstance(choice, dict), "invalid completion choice")
            require(
                choice.get("index") == 0 and isinstance(choice.get("text"), str),
                "invalid completion choice",
            )
            finish = choice.get("finish_reason")
            require(finish in (None, "stop", "length"), "unsupported finish reason")
        if event.get("usage") is not None:
            require(
                finish is not None and usage is None,
                "usage before finish or duplicate usage",
            )
            usage = event["usage"]
            require(isinstance(usage, dict), "invalid usage object")
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                require(count(usage.get(field)), f"invalid usage.{field}")
            require(
                usage["total_tokens"]
                == usage["prompt_tokens"] + usage["completion_tokens"],
                "incoherent token totals",
            )
            require(
                usage["completion_tokens"] <= cap, "completion exceeded requested cap"
            )
            require(
                finish != "length" or usage["completion_tokens"] == cap,
                "length finish below token cap",
            )
    require(done, "missing [DONE] (truncated stream)")
    return usage


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
            if phase == "profiling":
                failures.append({"request_id": key, "error": record["error"]})
            else:
                warmup_errors.append({"request_id": key, "error": record["error"]})
            continue
        try:
            require(
                not record["metadata"].get("was_cancelled")
                and not request.get("error"),
                "request was cancelled/failed",
            )
            require(request.get("status") == 200, "request did not return HTTP 200")
            usage = stream_usage(request, manifest)
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
        "limitations": "Protocol/usage audit only; not numerical, GPU or physical-memory qualification.",
    }
