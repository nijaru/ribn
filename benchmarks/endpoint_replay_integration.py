"""Explicit CPU-only qualification of AIPerf's finite timestamped replay.

This invokes the native client directly, not endpoint.py's synthetic-only runner.
"""

import json
import subprocess
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np
from aiperf_results import audit_timing_metrics
from aiperf_stream import completion_stream
from endpoint_integration import AIPERF, local_endpoint


class ReplayClientTests(unittest.TestCase):
    def test_fixed_schedule_preserves_cohort_under_backpressure(self):
        rows = [
            {
                "session_id": f"req-{i}",
                "timestamp": i * 50,
                "text_input": " ".join(["hello"] * (i + 1)),
                "output_length": 8,
            }
            for i in range(4)
        ]
        expected = {row["session_id"]: row for row in rows}
        with tempfile.TemporaryDirectory(prefix="ribn replay ") as tmp:
            directory = Path(tmp)
            trace = directory / "schedule.jsonl"
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
            for concurrency in (4, 1):
                with self.subTest(concurrency=concurrency):
                    case = directory / f"c{concurrency}"
                    case.mkdir()
                    with local_endpoint(case, "timing") as (tokenizer, server):
                        client = case / "client"
                        command = [
                            AIPERF,
                            "profile",
                            "--model",
                            "fixture",
                            "--tokenizer",
                            str(tokenizer),
                            "--tokenizer-revision",
                            "fixture",
                            "--endpoint-type",
                            "completions",
                            "--endpoint",
                            "/v1/completions",
                            "--streaming",
                            "--url",
                            f"http://127.0.0.1:{server.server_port}",
                            "--input-file",
                            str(trace),
                            "--custom-dataset-type",
                            "mooncake_trace",
                            "--fixed-schedule",
                            "--dataset-sampling-strategy",
                            "sequential",
                            "--concurrency",
                            str(concurrency),
                            "--random-seed",
                            "42",
                            "--use-server-token-count",
                            "--extra-inputs",
                            '{"temperature":0}',
                            "--connection-reuse-strategy",
                            "never",
                            "--wait-for-model-mode",
                            "inference",
                            "--wait-for-model-timeout",
                            "0",
                            "--request-timeout-seconds",
                            "10",
                            "--export-level",
                            "raw",
                            "--export-http-trace",
                            "--no-server-metrics",
                            "--no-gpu-telemetry",
                            "--ui",
                            "none",
                            "--no-auto-plot",
                            "--output-artifact-dir",
                            str(client),
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
                        raw = [
                            json.loads(line)
                            for line in (client / "profile_export_raw.jsonl")
                            .read_text()
                            .splitlines()
                        ]
                        records = [
                            json.loads(line)
                            for line in (client / "profile_export.jsonl")
                            .read_text()
                            .splitlines()
                        ]
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
