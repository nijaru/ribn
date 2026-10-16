"""Explicit CPU-only acceptance test using installed AIPerf, never a model/GPU.

Run with the AIPerf environment's Python; not part of the dependency-free unit suite.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
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
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests += 1
        if self.server.mode == "overload" and self.server.requests == 2:
            self.send_error(429, "fixture overload")
            return
        usage = {"prompt_tokens": 16, "completion_tokens": 2, "total_tokens": 18}
        messages = [
            {
                "object": "text_completion",
                "choices": [
                    {"index": 0, "text": "hello world", "finish_reason": "stop"}
                ],
            },
            {"object": "text_completion", "choices": [], "usage": usage},
        ]
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
        self.wfile.write(body)


class RealClientTests(unittest.TestCase):
    def exercise(self, mode, rate=0, arrival="constant"):
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
                        .replace("warmup_requests = 0", "warmup_requests = 1")
                        .replace("request_rate = 0.0", f"request_rate = {rate}")
                        .replace(
                            'arrival_pattern = "poisson"',
                            f'arrival_pattern = "{arrival}"',
                        )
                        + "\n[slo]\nttft_ms = 1000\nitl_ms = 1000\ne2e_ms = 10000\n"
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
                        0 if mode in ("valid", "overload") else 1,
                        result.stdout + result.stderr + log[-3000:],
                    )
                    report = json.loads((output / "validation.json").read_text())
                    # Readiness probes are disabled: precisely one warmup + two profiling requests.
                    self.assertEqual(server.requests, 3)
                    return report
                finally:
                    server.shutdown()
                    thread.join()

    def test_real_exports_accept_early_eos_and_reject_false_client_successes(self):
        for mode in ("valid", "truncated", "late-error", "no-usage"):
            with self.subTest(mode=mode):
                report = self.exercise(mode)
                if mode == "valid":
                    self.assertTrue(report["protocol_valid"])
                    self.assertEqual(report["completion_tokens"], 4)
                else:
                    self.assertFalse(report["protocol_valid"])
                    self.assertEqual(len(report["invalid_requests"]), 2)
                    self.assertEqual(len(report["warmup_errors"]), 1)

    def test_rate_control_preserves_partial_overload(self):
        for arrival in ("constant", "poisson"):
            with self.subTest(arrival=arrival):
                report = self.exercise("overload", rate=2, arrival=arrival)
                self.assertTrue(report["protocol_valid"])
                self.assertEqual(
                    (
                        report["offered_requests"],
                        report["successful_requests"],
                        report["failed_requests"],
                    ),
                    (2, 1, 1),
                )
                self.assertEqual(report["completion_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
