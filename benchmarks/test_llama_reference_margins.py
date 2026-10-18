"""Common-history comparisons must never follow the reference winner after divergence."""

import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llama_reference_margins import collect, parse_inputs, reference_row

ROOT = Path(__file__).resolve().parent
ARTIFACT = "gguf:sha256:" + "a" * 64
FIXTURE = f"{ARTIFACT}\n1 2\n20 21 22\n".encode()
# Independent native-API contract, not copied from the collector's configuration.
EXPECTED_POLICY = {
    "n_predict": 1,
    "temperature": -1.0,
    "samplers": ["temperature"],
    "n_probs": 5,
    "post_sampling_probs": False,
    "ignore_eos": False,
    "logit_bias": [],
    "cache_prompt": False,
    "return_tokens": True,
    "add_special": False,
    "stream": False,
    "stop": [],
    "seed": 1,
}


def ribn_log():
    lines = [f"margins-context artifact={ARTIFACT} prompt=1,2"]
    for index, chosen in enumerate([10, 11, 14]):
        ids = [chosen] + [key for key in range(10, 16) if key != chosen][:4]
        top = " ".join(
            f"{key}:{value:.4f}"
            for key, value in zip(ids, [-1, -1.2, -2.2, -3.3, -4.4])
        )
        lines.append(f"margins[{index}] chosen={chosen} gap=0.2000 {top}")
    return ("\n".join(lines) + "\nprefill timing is not comparison data\n").encode()


def reference(history):
    top = [
        {"id": key, "logprob": value}
        for key, value in zip([20, 10, 11, 12, 13], [-0.3, -2.3, -3.3, -4.3, -5.3])
    ]
    return {
        "tokens": [20],
        "tokens_evaluated": len(history),
        "tokens_predicted": 1,
        "tokens_cached": 0,
        "truncated": False,
        "stop": True,
        "stop_type": "limit",
        "generation_settings": copy.deepcopy(EXPECTED_POLICY),
        "completion_probabilities": [{"id": 20, "logprob": -0.3, "top_logprobs": top}],
    }


class ReferenceFixture(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.payloads.append(payload)
        body = reference(payload["prompt"])
        if self.server.mode == "truncated" and len(self.server.payloads) == 2:
            body["truncated"] = True
        if self.server.mode == "overload":
            self.send_response(503)
            body = {"error": "fixture overload"}
        else:
            self.send_response(200)
        encoded = json.dumps(body).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class CommonHistoryTests(unittest.TestCase):
    def exercise(self, mode, check):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture, ribn = directory / "input.tokens", directory / "ribn.log"
            fixture.write_bytes(FIXTURE)
            ribn.write_bytes(ribn_log())
            with ThreadingHTTPServer(("127.0.0.1", 0), ReferenceFixture) as server:
                server.payloads = []
                server.mode = mode
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                try:
                    url = f"http://127.0.0.1:{server.server_port}"
                    check(directory, fixture, ribn, server, url)
                finally:
                    server.shutdown()
                    thread.join()

    def test_real_cli_replays_ribn_history_after_reference_divergence(self):
        def check(directory, fixture, ribn, server, url):
            output = directory / "evidence"
            args = [
                sys.executable,
                str(ROOT / "llama_reference_margins.py"),
                "--fixture",
                str(fixture),
                "--ribn-margins",
                str(ribn),
                "--url",
                url,
                "--reference-build",
                "CPU protocol fixture",
                "--output",
                str(output),
            ]
            result = subprocess.run(
                args, check=False, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "comparison.json").read_text())
            self.assertTrue(report["complete"])
            self.assertFalse(report["reference_identity_verified"])
            self.assertEqual(report["winner_disagreements"], [0, 1, 2])
            self.assertEqual(
                [payload["prompt"] for payload in server.payloads],
                [[1, 2], [1, 2, 10], [1, 2, 10, 11]],
            )
            for index, payload in enumerate(server.payloads):
                self.assertEqual(
                    {key: value for key, value in payload.items() if key != "prompt"},
                    EXPECTED_POLICY,
                )
                self.assertEqual(
                    json.loads((output / f"{index:04d}.request.json").read_text()),
                    payload,
                )
                self.assertEqual(
                    json.loads((output / f"{index:04d}.response.json").read_text()),
                    reference(payload["prompt"]),
                )
            first = report["rows"][0]
            self.assertAlmostEqual(
                first["shared_top_logprob_deltas"][0]["ribn_minus_reference"], 1.3
            )
            self.assertEqual(first["ribn_winner_reference_logprob"], -2.3)
            self.assertIsNone(report["rows"][2]["ribn_winner_reference_logprob"])
            evidence = (output / "comparison.json").read_bytes()
            second = subprocess.run(
                args, check=False, capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((output / "comparison.json").read_bytes(), evidence)
            self.assertEqual(len(server.payloads), 3)

        self.exercise("valid", check)

    def test_partial_failure_preserves_evidence_without_success(self):
        def check(directory, fixture, ribn, server, url):
            output = directory / "partial"
            report = collect(fixture, ribn, url, "CPU protocol fixture", output)
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["rows"]), 1)
            self.assertIn("truncation", report["error"])
            self.assertEqual(len(server.payloads), 2)
            self.assertTrue(
                json.loads((output / "0001.response.json").read_text())["truncated"]
            )
            self.assertEqual(
                json.loads((output / "comparison.json").read_text()), report
            )

        self.exercise("truncated", check)

    def test_http_failure_is_not_a_zero_difference_result(self):
        def check(directory, fixture, ribn, server, url):
            output = directory / "failed"
            report = collect(fixture, ribn, url, "CPU protocol fixture", output)
            self.assertFalse(report["complete"])
            self.assertEqual(report["rows"], [])
            self.assertIn("503", report["error"])
            self.assertEqual(
                json.loads((output / "0000.response.json").read_text()),
                {"error": "fixture overload"},
            )

        self.exercise("overload", check)

    def test_rejects_wrong_context_and_noncontiguous_or_nonfinite_margins(self):
        log = ribn_log()
        mutations = [
            log.replace(b"prompt=1,2", b"prompt=2,1"),
            log.replace(ARTIFACT.encode(), ("gguf:sha256:" + "b" * 64).encode()),
            log.replace(b"margins[1]", b"margins[4]"),
            log.replace(b"margins[1]", b"margins[0]"),
            log.split(b"\n", 1)[1],
            log.replace(b"gap=0.2000", b"gap=nan"),
            log.replace(b"10:-1.0000", b"10:nan"),
            log.replace(b"10:-1.0000", b"10:inf"),
            log.replace(b"10:-1.0000", b"10:0.1000"),
            log.replace(b"11:-1.2000", b"10:-1.2000"),
            log.replace(b"10:-1.0000", b"10:-8.0000"),
        ]
        for invalid in mutations:
            with self.subTest(invalid=invalid[:100]), self.assertRaises(ValueError):
                parse_inputs(FIXTURE, invalid)
        artifact, prompt, rows = parse_inputs(FIXTURE, log)
        self.assertEqual((artifact, prompt, len(rows)), (ARTIFACT, [1, 2], 3))

    def test_rejects_transformed_incomplete_or_incoherent_reference_rows(self):
        original = reference([1, 2])
        cases = [
            ("tokens_evaluated", 3),
            ("tokens_predicted", True),
            ("tokens_cached", 1),
            ("tokens", [10]),
            ("tokens", []),
            ("truncated", True),
            ("stop_type", "word"),
            ("completion_probabilities", []),
        ]
        for field, value in cases:
            body = copy.deepcopy(original)
            body[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                reference_row(body, [1, 2])
        for field, value in [
            ("temperature", 0.0),
            ("ignore_eos", True),
            ("samplers", ["top_k", "temperature"]),
            ("post_sampling_probs", True),
        ]:
            body = copy.deepcopy(original)
            body["generation_settings"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                reference_row(body, [1, 2])


if __name__ == "__main__":
    unittest.main()
