"""Explicit CPU-only qualification of AIPerf's finite timestamped replay.

This exercises the snapshotting runner and offline input-cohort validator.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np
from aiperf_results import audit_timing_metrics
from aiperf_stream import completion_stream
from endpoint_integration import AIPERF, ROOT, local_endpoint
from test_endpoint_trace import TRACE_MANIFEST


class ReplayClientTests(unittest.TestCase):
    def test_fixed_schedule_preserves_cohort_under_backpressure(self):
        rows = [
            {
                "session_id": f"req-{i}",
                "timestamp": i * 50,
                "text_input": " ".join(["hello"] * (i + 1))
                + ("\u0085\u2028\u2029" if i == 0 else ""),
                "output_length": 8,
            }
            for i in range(4)
        ]
        expected = {row["session_id"]: row for row in rows}
        with tempfile.TemporaryDirectory(prefix="ribn replay ") as tmp:
            directory = Path(tmp)
            for mode, concurrency in (("timing", 4), ("timing", 1), ("overload", 1)):
                with self.subTest(mode=mode, concurrency=concurrency):
                    case = directory / f"{mode}-c{concurrency}"
                    case.mkdir()
                    (case / "schedule.jsonl").write_text(
                        "".join(
                            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                        )
                    )
                    with local_endpoint(case, mode) as (tokenizer, server):
                        client = case / "run" / "client"
                        manifest = case / "manifest.toml"
                        manifest.write_text(
                            TRACE_MANIFEST.replace(
                                'tokenizer = "fixture"', f'tokenizer = "{tokenizer}"'
                            )
                            .replace(
                                "http://127.0.0.1:18000",
                                f"http://127.0.0.1:{server.server_port}",
                            )
                            .replace("request_count = 2", "request_count = 4")
                            .replace(
                                "max_concurrency = 1",
                                f"max_concurrency = {concurrency}",
                            )
                        )
                        command = [
                            sys.executable,
                            str(ROOT / "endpoint.py"),
                            "run",
                            str(manifest),
                            "--output",
                            str(case / "run"),
                            "--aiperf",
                            AIPERF,
                        ]
                        (case / "command.json").write_text(json.dumps(command))
                        result = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            timeout=60,
                            check=False,
                        )
                        (case / "client.log").write_text(result.stdout + result.stderr)
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        self.assertEqual(server.requests, 4)
                        report = json.loads(
                            (case / "run" / "validation.json").read_text()
                        )
                        self.assertTrue(report["protocol_valid"])
                        self.assertEqual(report["trace_cohort"]["matched_attempts"], 4)
                        self.assertEqual(
                            report["trace_cohort"]["input_offsets"],
                            {r["session_id"]: r["timestamp"] for r in rows},
                        )
                        self.assertFalse(report["planned_arrivals_audited"])
                        self.assertEqual(
                            (case / "run" / "schedule.jsonl").read_bytes(),
                            (case / "schedule.jsonl").read_bytes(),
                        )
                        if mode == "overload":
                            self.assertEqual(
                                (
                                    report["successful_requests"],
                                    report["failed_requests"],
                                ),
                                (3, 1),
                            )
                            continue
                        with (client / "profile_export_raw.jsonl").open(
                            encoding="utf-8"
                        ) as source:
                            raw = [json.loads(line) for line in source]
                        with (client / "profile_export.jsonl").open(
                            encoding="utf-8"
                        ) as source:
                            records = [json.loads(line) for line in source]
                        summary = json.loads(
                            (client / "profile_export_aiperf.json").read_text()
                        )
                        self.assertEqual(summary["aiperf_version"], "0.13.0")
                        self.assertEqual(summary["schema_version"], "1.4")
                        self.assertTrue(summary["is_complete"])
                        self.assertFalse(summary["was_cancelled"])
                        self.assertEqual(summary["request_count"]["avg"], 4)
                        self.assertEqual(
                            summary.get("error_request_count", {}).get("avg", 0), 0
                        )
                        phases = summary["input_config"]["phases"]
                        self.assertEqual(len(phases), 1)
                        self.assertEqual(phases[0]["type"], "fixed_schedule")
                        self.assertEqual(phases[0]["concurrency"], concurrency)
                        ordinary = {r["metadata"]["x_request_id"]: r for r in records}
                        raw_ids = {r["metadata"]["x_request_id"] for r in raw}
                        self.assertEqual(
                            (len(raw), len(records), len(ordinary), len(raw_ids)),
                            (4, 4, 4, 4),
                        )
                        self.assertEqual(raw_ids, ordinary.keys())
                        seen, offsets, intervals = set(), [], []
                        for record in raw:
                            meta = record["metadata"]
                            other = ordinary[meta["x_request_id"]]
                            self.assertEqual(meta, other["metadata"])
                            self.assertEqual(meta["benchmark_phase"], "profiling")
                            self.assertFalse(meta["was_cancelled"])
                            self.assertFalse(other.get("error"))
                            self.assertEqual(record["status"], 200)
                            key = meta["conversation_id"]
                            self.assertNotIn(key, seen)
                            seen.add(key)
                            row = expected[key]
                            self.assertEqual(meta["turn_index"], 0)
                            self.assertEqual(
                                record["payload"]["prompt"], row["text_input"]
                            )
                            usage, clocks = completion_stream(
                                record,
                                {
                                    "model": {"id": "fixture"},
                                    "workload": {"output_tokens_mean": 8},
                                },
                            )
                            self.assertEqual(
                                usage["prompt_tokens"],
                                len(server.tokenizer.encode(row["text_input"]).ids),
                            )
                            self.assertEqual(usage["completion_tokens"], 5)
                            audit_timing_metrics(
                                other["metrics"], clocks, usage["completion_tokens"]
                            )
                            intervals.append(
                                (
                                    record["start_perf_ns"],
                                    record["start_perf_ns"] + clocks["done_receipt"],
                                )
                            )
                            self.assertIs(type(meta["request_start_ns"]), int)
                            offsets.append(
                                meta["request_start_ns"] - row["timestamp"] * 1_000_000
                            )
                        self.assertEqual(seen, expected.keys())
                        if concurrency == 1:
                            # Prove the cap actually held later requests until
                            # the previous stream's DONE, not just in config.
                            for (_, done), (start, _) in pairwise(sorted(intervals)):
                                self.assertGreaterEqual(start, done)
                        # Match the pinned client's float64 epoch-offset column,
                        # including its low-bit loss BEFORE centering. This is
                        # relative lag; the schedule's absolute origin is absent.
                        offsets = np.asarray(offsets, dtype=np.float64)
                        relative_ms = (offsets - offsets.min()) / 1_000_000
                        for percentile in (50, 90, 99):
                            metric = summary[f"replay_sched_lag_p{percentile}"]
                            self.assertEqual(metric["unit"], "ms")
                            self.assertEqual(
                                metric["avg"],
                                float(np.percentile(relative_ms, percentile)),
                            )
                        self.assertEqual(
                            summary["replay_sched_degraded"]["avg"],
                            int(summary["replay_sched_lag_p99"]["avg"] > 500),
                        )
                        # No speed threshold: the cohort must survive either cap,
                        # even if an overloaded test host delays the control too.


if __name__ == "__main__":
    unittest.main()
