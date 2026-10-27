"""The finite single-turn text trace accepted by the benchmark runner."""

import hashlib
import json
import re

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


def audit_trace(raw, rows, manifest):
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
    }
