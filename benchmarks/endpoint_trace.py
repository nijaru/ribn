"""The finite single-turn text trace accepted by the benchmark runner."""

import hashlib
import json
import re
from datetime import datetime

from aiperf_stream import validate_payload
from endpoint_config import count, require


def parse_trace(contents, workload):
    rows = {}
    previous = 0
    # JSONL uses LF/CRLF, not Unicode separators inside JSON strings.
    lines = contents.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    for line in lines:
        row = json.loads(line)
        require(
            isinstance(row, dict)
            and row.keys()
            == {"session_id", "timestamp", "text_input", "output_length"},
            "unsupported trace row fields",
        )
        key = row["session_id"]
        require(
            isinstance(key, str)
            and re.fullmatch(r"[A-Za-z0-9_.:-]+", key)
            and key not in rows,
            "missing, unsafe or duplicate trace session_id",
        )
        offset = row["timestamp"]
        require(
            count(offset) and offset >= previous and (bool(rows) or offset == 0),
            "trace timestamps must be sorted integer milliseconds starting at zero",
        )
        require(
            isinstance(row["text_input"], str) and row["text_input"],
            "trace text_input must be nonempty",
        )
        row["text_input"].encode("utf-8")
        require(
            type(row["output_length"]) is int
            and row["output_length"] == workload["output_tokens_mean"],
            "trace output_length differs from uniform output cap",
        )
        rows[key] = row
        previous = offset
    require(
        len(rows) == workload["request_count"], "trace row count differs from manifest"
    )
    return rows


def trace_snapshot(directory, manifest, provenance):
    workload = manifest["workload"]
    if workload["kind"] != "fixed_schedule":
        return None
    contents = (directory / workload["trace_file"]).read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    require(provenance.get("trace_sha256") == digest, "trace changed after invocation")
    return parse_trace(contents, workload)


def reported_phase_clock(raw, rows, start_time, client_timezone):
    """Bound only the serialized controller epoch; never infer monotonic alignment."""
    require(client_timezone in (None, "UTC"), "unsupported recorded client timezone")
    if client_timezone is None:
        return None
    require(
        isinstance(start_time, str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?",
            start_time,
        ),
        "missing/invalid UTC phase start_time",
    )
    elapsed = datetime.fromisoformat(start_time) - datetime(1970, 1, 1)
    epoch_ns = (elapsed.days * 86400 + elapsed.seconds) * 1_000_000_000
    epoch_ns += elapsed.microseconds * 1000
    require(
        0 <= epoch_ns < (1 << 32) * 1_000_000_000, "phase epoch outside supported range"
    )
    # v0.13 converts integer ns to float seconds, then a microsecond datetime.
    # Below 2**32 seconds, integer/float rounding plus datetime rounding is <2us.
    # This interval does NOT bound distinct worker/controller clock anchors.
    uncertainty = 2000
    low, high = epoch_ns - uncertainty, epoch_ns + uncertainty
    offsets = {}
    for record in raw.values():
        meta = record["metadata"]
        started = meta.get("request_start_ns")
        require(count(started), "missing/invalid epoch request_start_ns")
        offset = rows[meta["conversation_id"]]["timestamp"] * 1_000_000
        offsets[meta["conversation_id"]] = [
            started - high - offset,
            started - low - offset,
        ]
    return {
        "source": "client/profile_export_aiperf.json:start_time",
        "timezone": "UTC",
        "origin_ns_bounds": [low, high],
        "serialization_uncertainty_ns": uncertainty,
        "dispatch_offset_ns_bounds": offsets,
        "monotonic_alignment_audited": False,
    }


def audit_trace(raw, rows, manifest, *, start_time=None, client_timezone=None):
    """Bind every issued attempt, including failures, to its declared input."""
    seen = set()
    for record in raw.values():
        metadata = record["metadata"]
        key = metadata.get("conversation_id")
        require(
            key in rows
            and key not in seen
            and type(metadata.get("turn_index")) is int
            and metadata["turn_index"] == 0
            and metadata.get("benchmark_phase") == "profiling",
            "trace conversation coverage differs",
        )
        validate_payload(record, manifest)
        require(
            record["payload"].get("prompt") == rows[key]["text_input"],
            "trace request prompt differs",
        )
        seen.add(key)
    require(seen == rows.keys(), "trace conversation coverage differs")
    return {
        "declared_requests": len(rows),
        "matched_attempts": len(seen),
        "timestamp_unit": "ms",
        "input_offsets": {key: row["timestamp"] for key, row in rows.items()},
        "absolute_origin_audited": False,
        "reported_phase_clock": reported_phase_clock(
            raw, rows, start_time, client_timezone
        ),
    }
