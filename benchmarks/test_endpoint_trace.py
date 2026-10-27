"""Trace snapshots must identify all attempted work, not just surviving requests."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiperf_results import validate
from endpoint import run
from endpoint_config import parse_manifest
from endpoint_trace import parse_trace
from test_endpoint import MANIFEST, export, fixture

TRACE_MANIFEST = MANIFEST.replace(
    'kind = "synthetic"', 'kind = "fixed_schedule"\ntrace_file = "schedule.jsonl"'
)
for line in (
    "input_tokens_mean = 16\n",
    "input_tokens_stddev = 0\n",
    "request_rate = 0.0\n",
    'arrival_pattern = "poisson"\n',
):
    TRACE_MANIFEST = TRACE_MANIFEST.replace(line, "")


def rows(count=2):
    return [
        {
            "session_id": f"req-{i}",
            "timestamp": i * 50,
            "text_input": f"fixture prompt {i}",
            "output_length": 8,
        }
        for i in range(count)
    ]


def trace_fixture(directory):
    ordinary, raw, summary = fixture(directory)
    contents = "".join(json.dumps(row) + "\n" for row in rows()).encode()
    (directory / "schedule.jsonl").write_bytes(contents)
    (directory / "manifest.toml").write_text(TRACE_MANIFEST)
    (directory / "invocation.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(TRACE_MANIFEST.encode()).hexdigest(),
                "trace_sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    )
    for i, (o, r) in enumerate(zip(ordinary, raw, strict=True)):
        o["metadata"].update(conversation_id=f"req-{i}", turn_index=0)
        r["metadata"].update(conversation_id=f"req-{i}", turn_index=0)
    cfg = summary["input_config"]
    cfg["datasets"] = [
        {
            "name": "main",
            "type": "file",
            "path": str((directory / "schedule.jsonl").resolve()),
            "format": "mooncake_trace",
            "sampling": "sequential",
            "random_seed": 42,
        }
    ]
    cfg["phases"][0]["type"] = "fixed_schedule"
    export(directory, ordinary, raw, summary)
    return ordinary, raw, summary


class TraceTests(unittest.TestCase):
    def test_unicode_prompt_separators_are_not_jsonl_boundaries(self):
        data = rows()
        data[0]["text_input"] = "hello\u0085world\u2028again\u2029end"
        contents = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in data
        ).encode("utf-8")
        parsed = parse_trace(
            contents, parse_manifest(TRACE_MANIFEST.encode())["workload"]
        )
        self.assertEqual(parsed["req-0"]["text_input"], data[0]["text_input"])

    def test_rejects_invalid_whole_trace_before_any_client_or_output(self):
        for change in (
            lambda r: r[0].update(timestamp=1),
            lambda r: r[1].update(timestamp=True),
            lambda r: r[1].update(timestamp=-1),
            lambda r: r[1].update(session_id=r[0]["session_id"]),
            lambda r: r[1].update(session_id="bad\nheader"),
            lambda r: r[1].update(output_length=9),
            lambda r: r[1].update(text_input=""),
            lambda r: r[1].update(payload={}),
            lambda r: r.pop(),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                r = rows()
                change(r)
                (d / "manifest.toml").write_text(TRACE_MANIFEST)
                (d / "schedule.jsonl").write_text(
                    "".join(json.dumps(x) + "\n" for x in r)
                )
                with patch("endpoint.subprocess.run") as client:
                    with self.assertRaises(ValueError):
                        run(d / "manifest.toml", d / "out", "aiperf")
                    client.assert_not_called()
                self.assertFalse((d / "out").exists())

    def test_manifest_rejects_ambiguous_controls_and_sidecar_paths(self):
        for old, new in [
            ("warmup_requests = 0", "warmup_requests = 1"),
            ("schedule.jsonl", "../schedule.jsonl"),
            ("schedule.jsonl", "/schedule.jsonl"),
            ("max_concurrency = 1", "max_concurrency = 1\nrequest_rate = 0.0"),
        ]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                parse_manifest(TRACE_MANIFEST.replace(old, new).encode())
        self.assertEqual(
            list(
                parse_trace(
                    "".join(json.dumps(r) + "\n" for r in rows()).encode(),
                    parse_manifest(TRACE_MANIFEST.encode())["workload"],
                )
            ),
            ["req-0", "req-1"],
        )

    def test_all_attempts_include_failed_trace_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            o, r, s = trace_fixture(d)
            o[1]["error"] = {"message": "overload"}
            o[1]["metrics"] = {}
            r[1]["status"] = 429
            r[1]["responses"] = []
            s["request_count"]["avg"] = 1
            s["error_request_count"] = {"unit": "requests", "avg": 1}
            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                s[f"total_usage_{k}"]["avg"] /= 2
            export(d, o, r, s)
            report = validate(d)
            self.assertTrue(report["protocol_valid"])
            self.assertEqual(report["trace_cohort"]["matched_attempts"], 2)
            self.assertEqual(report["failed_requests"], 1)
            self.assertIsNone(report["issued_load"]["configured_request_rate"])
            r[1]["payload"]["prompt"] = "substituted"
            export(d, o, r, s)
            with self.assertRaisesRegex(ValueError, "trace request prompt"):
                validate(d)

    def test_rejects_changed_snapshot_or_any_conversation_mismatch(self):
        for change in ("bytes", "id", "duplicate", "turn", "prompt", "cap", "config"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                o, r, s = trace_fixture(d)
                if change == "bytes":
                    (d / "schedule.jsonl").write_bytes(
                        (d / "schedule.jsonl").read_bytes() + b"\n"
                    )
                elif change == "config":
                    s["input_config"]["datasets"][0]["path"] = "elsewhere.jsonl"
                elif change in ("id", "duplicate", "turn"):
                    r[1]["metadata"].update(
                        {
                            "id": {"conversation_id": "unknown"},
                            "duplicate": {"conversation_id": "req-0"},
                            "turn": {"turn_index": 1},
                        }[change]
                    )
                    o[1]["metadata"] = copy.deepcopy(r[1]["metadata"])
                elif change == "prompt":
                    r[0]["payload"]["prompt"] = "wrong"
                else:
                    r[0]["payload"]["max_tokens"] = 9
                export(d, o, r, s)
                with self.assertRaises(ValueError):
                    validate(d)


if __name__ == "__main__":
    unittest.main()
