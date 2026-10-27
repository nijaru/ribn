"""Explicit CPU-only acceptance test using installed AIPerf, never a model/GPU.

Run with the AIPerf environment's Python; not part of the dependency-free unit suite.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from test_endpoint import MANIFEST
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parent
AIPERF = os.environ.get("AIPERF", str(Path(sys.executable).with_name("aiperf")))


class EndpointFixture(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt_tokens = len(self.server.tokenizer.encode(payload["prompt"]).ids)
        self.server.prompt_lengths.append(prompt_tokens)
        self.server.requests += 1
        if self.server.mode == "overload" and self.server.requests == 2:
            self.send_error(429, "fixture overload")
            return
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 2,
            "total_tokens": prompt_tokens + 2,
        }
        messages = [
            {
                "object": "text_completion",
                "choices": [
                    {"index": 0, "text": "hello world", "finish_reason": "stop"}
                ],
            },
            {"object": "text_completion", "choices": [], "usage": usage},
        ]
        if self.server.mode == "single-token":
            usage.update(completion_tokens=1, total_tokens=prompt_tokens + 1)
            messages[0]["choices"][0]["text"] = "hello"
        if self.server.mode == "mlx-usage":
            messages[-1]["object"] = "chat.completion"
        if self.server.mode == "timing":
            usage.update(completion_tokens=5, total_tokens=prompt_tokens + 5)
            messages = [
                {
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                }
                for text, finish in (
                    ("", None),
                    ("hello ", None),
                    ("world", None),
                    ("", "stop"),
                )
            ] + [{"object": "text_completion", "choices": [], "usage": usage}]
        if self.server.mode == "late-error":
            messages.insert(1, {"error": {"message": "fixture device fault"}})
        if self.server.mode == "no-usage":
            messages.pop()
        body = "".join(f"data: {json.dumps(message)}\n\n" for message in messages)
        if self.server.mode != "truncated":
            body += "data: [DONE]\n\n"
        body = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.server.mode == "timing":
            events = []
            self.server.wire_events[self.headers["X-Request-ID"]] = events
            for frame in body.splitlines(keepends=True):
                if frame.startswith(b"data:"):
                    time.sleep(0.1)
                    events.append(time.perf_counter_ns())
                self.wfile.write(frame)
                self.wfile.flush()
        else:
            self.wfile.write(body)


class RealClientTests(unittest.TestCase):
    def exercise(
        self,
        mode,
        rate=0,
        arrival="constant",
        input_stddev=0,
        requests=2,
        warmup=1,
        slos=None,
    ):
        if slos is None:
            slos = {"ttft_ms": 1000, "itl_ms": 1000, "last_content_ms": 10000}
        with tempfile.TemporaryDirectory(prefix="ribn endpoint ") as tmp:
            directory = Path(tmp)
            tokenizer_dir = directory / "tokenizer"
            tokenizer_dir.mkdir()
            tokenizer = Tokenizer(
                models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
            (tokenizer_dir / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "PreTrainedTokenizerFast",
                        "unk_token": "[UNK]",
                        "model_max_length": 8192,
                    }
                )
            )
            with ThreadingHTTPServer(("127.0.0.1", 0), EndpointFixture) as server:
                server.mode = mode
                server.requests = 0
                server.tokenizer = tokenizer
                server.prompt_lengths = []
                server.wire_events = {}
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                try:
                    manifest = directory / "manifest.toml"
                    manifest.write_text(
                        MANIFEST.replace(
                            'tokenizer = "fixture"', f'tokenizer = "{tokenizer_dir}"'
                        )
                        .replace(
                            "http://127.0.0.1:18000",
                            f"http://127.0.0.1:{server.server_port}",
                        )
                        .replace("warmup_requests = 0", f"warmup_requests = {warmup}")
                        .replace("request_count = 2", f"request_count = {requests}")
                        .replace(
                            "input_tokens_stddev = 0",
                            f"input_tokens_stddev = {input_stddev}",
                        )
                        .replace("request_rate = 0.0", f"request_rate = {rate}")
                        .replace(
                            'arrival_pattern = "poisson"',
                            f'arrival_pattern = "{arrival}"',
                        )
                        + "\n[slo]\n"
                        + "".join(f"{name} = {value}\n" for name, value in slos.items())
                    )
                    output = directory / "run"
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "endpoint.py"),
                            "run",
                            str(manifest),
                            "--output",
                            str(output),
                            "--aiperf",
                            AIPERF,
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    log = (output / "client.log").read_text()
                    self.assertEqual(
                        (output / "client-exit-code").read_text().strip(), "0", log
                    )
                    self.assertEqual(
                        result.returncode,
                        0
                        if mode
                        in ("valid", "mlx-usage", "overload", "timing", "single-token")
                        else 1,
                        result.stdout + result.stderr + log[-3000:],
                    )
                    report = json.loads((output / "validation.json").read_text())
                    # Readiness probes are disabled: only declared issued attempts.
                    self.assertEqual(server.requests, requests + warmup)
                    client = output / "client"
                    raw = [
                        json.loads(line)
                        for line in (client / "profile_export_raw.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    # Independent reference: count actual profiling prompt digests,
                    # including overload failures, without calling the auditor.
                    digests = [
                        hashlib.sha256(
                            r["payload"]["prompt"].encode("utf-8")
                        ).hexdigest()
                        for r in raw
                        if r["metadata"]["benchmark_phase"] == "profiling"
                    ]
                    pairs = [[d, digests.count(d)] for d in sorted(set(digests))]
                    encoded = json.dumps(pairs, separators=(",", ":")).encode("ascii")
                    self.assertEqual(
                        report["prompt_cohort"],
                        {
                            "algorithm": "sha256-utf8-prompt-multiset-v1",
                            "sha256": hashlib.sha256(
                                b"ribn:profiling-prompt-multiset:v1\0" + encoded
                            ).hexdigest(),
                            "requests": requests,
                            "unavailable_request_ids": [],
                        },
                    )
                    return {
                        "audit": report,
                        "prompt_lengths": server.prompt_lengths[warmup:],
                        "records": [
                            json.loads(line)
                            for line in (client / "profile_export.jsonl")
                            .read_text()
                            .splitlines()
                        ],
                        "raw": raw,
                        "wire_events": server.wire_events,
                    }
                finally:
                    server.shutdown()
                    thread.join()

    def test_real_exports_accept_early_eos_and_reject_false_client_successes(self):
        for mode in ("valid", "mlx-usage", "truncated", "late-error", "no-usage"):
            with self.subTest(mode=mode):
                report = self.exercise(mode)["audit"]
                if mode in ("valid", "mlx-usage"):
                    self.assertTrue(report["protocol_valid"])
                    self.assertEqual(report["completion_tokens"], 4)
                else:
                    self.assertFalse(report["protocol_valid"])
                    self.assertEqual(len(report["invalid_requests"]), 2)
                    self.assertEqual(len(report["warmup_errors"]), 1)

    def test_single_token_response_has_no_itl_metric(self):
        captured = self.exercise("single-token", warmup=0)
        self.assertTrue(captured["audit"]["protocol_valid"])
        self.assertEqual(captured["audit"]["completion_tokens"], 2)
        self.assertIn(
            "inter_token_latency", captured["audit"]["audited_request_timing_metrics"]
        )
        self.assertEqual(captured["audit"]["slo_attainment"]["undefined_requests"], 2)
        self.assertEqual(captured["audit"]["slo_attainment"]["good_requests"], 0)
        for record in captured["records"]:
            self.assertEqual(record["metrics"]["output_sequence_length"]["value"], 1)
            self.assertNotIn("inter_token_latency", record["metrics"])
            self.assertNotIn("good_request_count", record["metrics"])

    def test_single_token_can_meet_slos_without_an_itl_requirement(self):
        captured = self.exercise(
            "single-token",
            warmup=0,
            slos={"ttft_ms": 1000, "last_content_ms": 10000},
        )
        self.assertTrue(captured["audit"]["protocol_valid"])
        attainment = captured["audit"]["slo_attainment"]
        self.assertEqual(attainment["good_requests"], 2)
        self.assertEqual(attainment["undefined_requests"], 0)
        self.assertEqual(attainment["attempted_fraction"], 1.0)

    def test_joint_slo_rejection_is_not_a_protocol_failure(self):
        captured = self.exercise(
            "valid",
            warmup=0,
            slos={"ttft_ms": 0.000001, "itl_ms": 1000},
        )
        self.assertTrue(captured["audit"]["protocol_valid"])
        attainment = captured["audit"]["slo_attainment"]
        self.assertEqual(attainment["good_requests"], 0)
        self.assertEqual(attainment["violated_requests"], 2)
        self.assertEqual(attainment["attempted_fraction"], 0.0)

    def test_variable_inputs_reach_the_endpoint_and_preserve_actual_usage(self):
        captured = self.exercise("valid", input_stddev=8, requests=16)
        report, prompt_lengths = captured["audit"], captured["prompt_lengths"]
        self.assertTrue(report["protocol_valid"])
        self.assertEqual(report["attempted_requests"], 16)
        self.assertEqual(report["successful_requests"], 16)
        self.assertEqual(report["issued_load"]["issued_requests"], 16)
        profiling_ids = {
            r["metadata"]["x_request_id"]
            for r in captured["raw"]
            if r["metadata"]["benchmark_phase"] == "profiling"
        }
        self.assertEqual(set(report["request_timing_ns"]), profiling_ids)
        self.assertGreater(len(set(prompt_lengths)), 1, prompt_lengths)
        self.assertEqual(report["prompt_tokens"], sum(prompt_lengths))
        self.assertEqual(report["completion_tokens"], 32)

    def test_zero_warmup_and_content_clocks_with_delayed_terminal_frames(self):
        captured = self.exercise("timing", warmup=0)
        self.assertTrue(captured["audit"]["protocol_valid"])
        self.assertEqual(captured["audit"]["completion_tokens"], 10)
        ordinary = {r["metadata"]["x_request_id"]: r for r in captured["records"]}
        for raw in captured["raw"]:
            key = raw["metadata"]["x_request_id"]
            metrics = ordinary[key]["metrics"]
            responses = raw["responses"]
            self.assertEqual(len(responses), 6)
            for response, sent in zip(
                responses, captured["wire_events"][key], strict=True
            ):
                # Both processes use the same OS monotonic clock, independently.
                self.assertGreaterEqual(response["perf_ns"], sent)
            first, last = responses[1]["perf_ns"], responses[2]["perf_ns"]
            start = raw["start_perf_ns"]
            self.assertEqual(
                captured["audit"]["request_timing_ns"][key],
                {
                    "first_content": first - start,
                    "last_content": last - start,
                    "done_receipt": responses[-1]["perf_ns"] - start,
                },
            )
            for name, expected in {
                "time_to_first_token": (first - start) * 1e-6,
                "request_latency": (last - start) * 1e-6,
                "inter_token_latency": ((last - first) / 4) * 1e-6,
            }.items():
                self.assertEqual(metrics[name]["unit"], "ms")
                self.assertEqual(metrics[name]["value"], expected)
            # Server usage wins over two visible chunks or retokenized text.
            self.assertEqual(metrics["output_sequence_length"]["value"], 5)
            self.assertEqual(len(metrics["inter_chunk_latency"]["value"]), 1)
            self.assertLess(last, responses[3]["perf_ns"])
            self.assertLess(responses[3]["perf_ns"], responses[-1]["perf_ns"])

    def test_rate_ceiling_does_not_establish_nominal_offered_load(self):
        captured = self.exercise("timing", rate=20, warmup=0)
        report = captured["audit"]
        self.assertTrue(report["protocol_valid"])
        self.assertEqual(report["attempted_requests"], 2)
        self.assertFalse(report["planned_arrivals_audited"])
        starts = sorted(r["start_perf_ns"] for r in captured["raw"])
        # One slot stays occupied through six delayed frames. The requested
        # 50 ms arrival interval is not an independently offered-work ledger.
        gap = starts[1] - starts[0]
        self.assertGreater(gap, 500_000_000)
        self.assertEqual(
            report["issued_load"],
            {
                "configured_request_rate": 20,
                "max_concurrency": 1,
                "issued_requests": 2,
                "start_span_ns": gap,
                "start_interval_ns": {"min": gap, "median": gap, "max": gap},
            },
        )

    def test_rate_control_preserves_partial_overload(self):
        for arrival in ("constant", "poisson"):
            with self.subTest(arrival=arrival):
                captured = self.exercise("overload", rate=2, arrival=arrival)
                report = captured["audit"]
                self.assertTrue(report["protocol_valid"])
                self.assertEqual(
                    (
                        report["attempted_requests"],
                        report["successful_requests"],
                        report["failed_requests"],
                    ),
                    (2, 1, 1),
                )
                self.assertEqual(report["completion_tokens"], 2)
                self.assertEqual(report["slo_attainment"]["good_requests"], 1)
                self.assertEqual(report["slo_attainment"]["failed_requests"], 1)
                self.assertEqual(report["slo_attainment"]["attempted_fraction"], 0.5)
                starts = sorted(
                    r["start_perf_ns"]
                    for r in captured["raw"]
                    if r["metadata"]["benchmark_phase"] == "profiling"
                )
                self.assertEqual(report["issued_load"]["issued_requests"], 2)
                self.assertEqual(
                    report["issued_load"]["start_span_ns"], starts[-1] - starts[0]
                )


if __name__ == "__main__":
    unittest.main()
