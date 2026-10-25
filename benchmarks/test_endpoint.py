"""Protection against accepting incomplete streams or miscounting attempted work."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aiperf_results import validate
from endpoint_config import check_config, parse_manifest

ROOT = Path(__file__).resolve().parent
MANIFEST = """schema_version = 1
[run]
name = "fixture"
engine = "fixture"
engine_version = "1"
engine_commit = "fixture"
benchmark_client_version = "0.13.0"
seed = 42
quality_scope = "host protocol only"
gpu_telemetry = false
[model]
id = "fixture"
revision = "fixture"
artifact_sha256 = "DIGEST"
tokenizer = "fixture"
tokenizer_revision = "fixture"
dtype = "none"
quantization = "none"
[hardware]
device = "CPU fixture"
vram_bytes = 0
driver = "none"
cuda = "none"
[server]
endpoint_type = "completions"
endpoint = "/v1/completions"
base_url = "http://127.0.0.1:18000"
streaming = true
command = "fixture"
[workload]
kind = "synthetic"
input_tokens_mean = 16
input_tokens_stddev = 0
output_tokens_mean = 8
output_tokens_stddev = 0
request_count = 2
warmup_requests = 0
max_concurrency = 1
request_rate = 0.0
arrival_pattern = "poisson"
request_timeout_seconds = 10
[sampling]
temperature = 0.0
ignore_eos = false
""".replace("DIGEST", "0" * 64)


def message(data):
    return {
        "perf_ns": 1_100_000_000,
        "packets": [{"name": "data", "value": json.dumps(data)}],
    }


def fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.toml").write_text(MANIFEST)
    (directory / "client-exit-code").write_text("0\n")
    (directory / "invocation.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(MANIFEST.encode()).hexdigest(),
            }
        )
    )
    client = directory / "client"
    client.mkdir()
    usage = {"prompt_tokens": 16, "completion_tokens": 2, "total_tokens": 18}
    metrics = {
        f"usage_{key}": {"value": value, "unit": "tokens"}
        for key, value in usage.items()
    }
    metrics["output_sequence_length"] = {"value": 2, "unit": "tokens"}
    for name in ("time_to_first_token", "request_latency"):
        metrics[name] = {"value": 100.0, "unit": "ms"}
    raw = []
    ordinary = []
    for index in range(2):
        metadata = {
            "x_request_id": str(index),
            "benchmark_phase": "profiling",
            "was_cancelled": False,
        }
        ordinary.append(
            {"metadata": copy.deepcopy(metadata), "metrics": copy.deepcopy(metrics)}
        )
        raw.append(
            {
                "metadata": metadata,
                "status": 200,
                "start_perf_ns": 1_000_000_000 + index * 2_000_000_000,
                "payload": {
                    "model": "fixture",
                    "stream": True,
                    "temperature": 0,
                    "max_tokens": 8,
                    "stream_options": {"include_usage": True},
                },
                "responses": [
                    message(
                        {
                            "object": "text_completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "text": "hello world",
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ),
                    message(
                        {
                            "object": "text_completion",
                            "choices": [],
                            "usage": copy.deepcopy(usage),
                        }
                    ),
                    {"packets": [{"name": "data", "value": "[DONE]"}]},
                ],
            }
        )
    for request in raw:
        for response in request["responses"]:
            response["perf_ns"] = request["start_perf_ns"] + 100_000_000
    summary = {
        "input_config": {
            "models": {"items": [{"name": "fixture"}]},
            "endpoint": {
                "urls": ["http://127.0.0.1:18000"],
                "type": "completions",
                "path": "/v1/completions",
                "timeout": 10,
                "streaming": True,
                "connection_reuse": "never",
                "use_server_token_count": True,
                "extra": {"temperature": 0},
                "wait_for_model_mode": "inference",
                "wait_for_model_timeout": 0,
            },
            "datasets": [
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 2,
                    "random_seed": 42,
                    "prompts": {
                        "isl": {"mean": 16, "stddev": 0},
                        "osl": {"mean": 8, "stddev": 0},
                    },
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "kind": "profiling",
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 1,
                }
            ],
            "tokenizer": {"name": "fixture", "revision": "fixture"},
            "artifacts": {"records": ["jsonl"], "raw": True, "trace": True},
            "gpu_telemetry": {"enabled": False},
        },
        "schema_version": "1.4",
        "aiperf_version": "0.13.0",
        "is_complete": True,
        "was_cancelled": False,
        "request_count": {"avg": 2, "unit": "requests"},
        **{
            f"total_usage_{key}": {"avg": value * 2, "unit": "tokens"}
            for key, value in usage.items()
        },
    }
    return ordinary, raw, summary


def export(directory, ordinary, raw, summary):
    client = directory / "client"
    for filename, rows in [
        ("profile_export.jsonl", ordinary),
        ("profile_export_raw.jsonl", raw),
    ]:
        (client / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (client / "profile_export_aiperf.json").write_text(json.dumps(summary))


class EndpointTests(unittest.TestCase):
    def test_rejects_end_to_end_label_for_last_content_slo(self):
        with self.assertRaisesRegex(ValueError, "unknown SLO"):
            parse_manifest((MANIFEST + "\n[slo]\ne2e_ms = 1000\n").encode())

    def test_early_eos_uses_actual_usage_and_joins_unordered_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = fixture(directory)
            export(directory, ordinary, raw[::-1], summary)
            report = validate(directory)
            self.assertTrue(report["protocol_valid"])
            self.assertEqual(report["attempted_requests"], 2)
            self.assertFalse(report["planned_arrivals_audited"])
            self.assertEqual(report["issued_load"]["start_span_ns"], 2_000_000_000)
            self.assertEqual(
                report["request_timing_ns"]["0"],
                {
                    "first_content": 100_000_000,
                    "last_content": 100_000_000,
                    "done_receipt": 100_000_000,
                },
            )  # Multiple SSE frames in one receive may share a clock.
            self.assertEqual(
                report["completion_tokens"], 4
            )  # Not 2 requests * cap 8, nor chunk count.

    def test_native_millisecond_rounding_is_not_an_empirical_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = fixture(directory)
            for response in raw[0]["responses"]:
                response["perf_ns"] = raw[0]["start_perf_ns"] + 24_925_571
            # Pinned ns * (1000 / 1e9), not ns / 1e6 (one ULP different).
            for name in ("time_to_first_token", "request_latency"):
                ordinary[0]["metrics"][name]["value"] = 24.925570999999998
            export(directory, ordinary, raw, summary)
            self.assertTrue(validate(directory)["protocol_valid"])

    def test_rejects_corrupt_content_clocks_and_native_timings(self):
        for name, change in (
            ("missing clock", lambda o, r: r["responses"][0].pop("perf_ns")),
            ("boolean clock", lambda o, r: r["responses"][0].update(perf_ns=True)),
            ("fractional clock", lambda o, r: r["responses"][0].update(perf_ns=1.1e9)),
            ("pre-start clock", lambda o, r: r["responses"][0].update(perf_ns=0)),
            (
                "regressing clock",
                lambda o, r: r["responses"][1].update(perf_ns=1_050_000_000),
            ),
            (
                "TTFT mismatch",
                lambda o, r: o["metrics"]["time_to_first_token"].update(value=0),
            ),
            (
                "last-content mismatch",
                lambda o, r: o["metrics"]["request_latency"].update(value=200),
            ),
            (
                "timing unit",
                lambda o, r: o["metrics"]["request_latency"].update(unit="sec"),
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = fixture(directory)
                change(ordinary[0], raw[0])
                export(directory, ordinary, raw, summary)
                report = validate(directory)
                self.assertFalse(report["protocol_valid"])
                self.assertEqual(report["successful_requests"], 1)
                self.assertEqual(len(report["invalid_requests"]), 1)
                self.assertNotIn("0", report["request_timing_ns"])

    def test_missing_or_invalid_issued_start_cannot_produce_an_arrival_summary(self):
        for value in (None, True, -1, 1e9):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = fixture(directory)
                raw[0]["start_perf_ns"] = value
                export(directory, ordinary, raw, summary)
                with self.assertRaisesRegex(ValueError, "nanosecond timestamp"):
                    validate(directory)

    def test_rejects_broken_streams_and_usage_even_if_aiperf_claimed_success(self):
        variants = {}
        base = {"object": "text_completion", "choices": []}
        variants["missing done"] = lambda r: r["responses"].pop()
        variants["missing usage"] = lambda r: r["responses"].pop(1)
        variants["late JSON error"] = lambda r: r["responses"].insert(
            1, message({"error": {"message": "device fault"}})
        )
        variants["bad JSON"] = lambda r: r["responses"].insert(
            1, {"perf_ns": 1_100_000_000, "packets": [{"name": "data", "value": "{"}]}
        )
        variants["data after done"] = lambda r: r["responses"].append(message(base))
        variants["null choice"] = lambda r: r["responses"].insert(
            0, message({"object": "text_completion", "choices": [None]})
        )
        variants["duplicate finish"] = lambda r: r["responses"].insert(
            1, copy.deepcopy(r["responses"][0])
        )
        variants["EOS bypass"] = lambda r: r["payload"].update(ignore_eos=True)
        variants["forced minimum"] = lambda r: r["payload"].update(min_tokens=8)
        variants["boolean token cap"] = lambda r: r["payload"].update(max_tokens=True)
        variants["incoherent totals"] = lambda r: r["responses"].__setitem__(
            1,
            message(
                {
                    **base,
                    "usage": {
                        "prompt_tokens": 16,
                        "completion_tokens": 2,
                        "total_tokens": 99,
                    },
                }
            ),
        )
        variants["boolean token usage"] = lambda r: r["responses"].__setitem__(
            1,
            message(
                {
                    **base,
                    "usage": {
                        "prompt_tokens": 16,
                        "completion_tokens": True,
                        "total_tokens": 17,
                    },
                }
            ),
        )
        for name, mutate in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = fixture(directory)
                mutate(raw[0])
                export(directory, ordinary, raw, summary)
                report = validate(directory)
                self.assertFalse(report["protocol_valid"])
                self.assertEqual(report["successful_requests"], 1)
                self.assertEqual(len(report["invalid_requests"]), 1)

    def test_chat_label_is_accepted_only_for_terminal_usage_only_frames(self):
        # Stock MLX-LM 0.32.2 labels the usage-only frame this way even on
        # /v1/completions. It must not admit chat content or bypass accounting.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = fixture(directory)
            usage = json.loads(raw[0]["responses"][1]["packets"][0]["value"])
            usage["object"] = "chat.completion"
            raw[0]["responses"][1] = message(usage)
            export(directory, ordinary, raw, summary)
            self.assertTrue(validate(directory)["protocol_valid"])

            for variant in ("content", "missing usage", "bad totals", "early usage"):
                with self.subTest(variant=variant):
                    changed = copy.deepcopy(raw)
                    if variant == "content":
                        frame = json.loads(
                            changed[0]["responses"][0]["packets"][0]["value"]
                        )
                        frame["object"] = "chat.completion"
                        changed[0]["responses"][0] = message(frame)
                    elif variant == "missing usage":
                        changed[0]["responses"][1] = message(
                            {"object": "chat.completion", "choices": []}
                        )
                    elif variant == "bad totals":
                        frame = copy.deepcopy(usage)
                        frame["usage"]["total_tokens"] += 1
                        changed[0]["responses"][1] = message(frame)
                    else:
                        changed[0]["responses"][0:2] = changed[0]["responses"][1::-1]
                    export(directory, ordinary, changed, summary)
                    self.assertFalse(validate(directory)["protocol_valid"])

    def test_failures_remain_in_attempted_work_and_do_not_inflate_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = fixture(directory)
            error = {"message": "overloaded", "code": 429}
            ordinary[0]["error"] = error
            raw[0].update(status=429, responses=[], error=error)
            summary["request_count"]["avg"] = 1
            summary["error_request_count"] = {"avg": 1, "unit": "requests"}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                summary[f"total_usage_{key}"]["avg"] /= 2
            export(directory, ordinary, raw, summary)
            report = validate(directory)
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
            self.assertEqual(report["issued_load"]["issued_requests"], 2)
            self.assertEqual(report["issued_load"]["start_span_ns"], 2_000_000_000)
            self.assertEqual(set(report["request_timing_ns"]), {"1"})
            for clocks in (
                (None,),
                (True,),
                (-1,),
                (0,),
                (1.1e9,),
                (1_100_000_000, 1_050_000_000),
            ):
                with self.subTest(failed_response_clocks=clocks):
                    raw[0]["responses"] = [{"perf_ns": value} for value in clocks]
                    export(directory, ordinary, raw, summary)
                    changed = validate(directory)
                    self.assertFalse(changed["protocol_valid"])
                    self.assertEqual(changed["failed_requests"], 1)
                    self.assertEqual(changed["successful_requests"], 1)
                    self.assertEqual(changed["invalid_requests"][0]["request_id"], "0")

    def test_rejects_missing_duplicate_and_inconsistent_exports(self):
        for mutation in (
            "missing raw",
            "duplicate raw",
            "aggregate tokens",
            "record tokens",
            "cancelled",
            "version",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = fixture(directory)
                if mutation == "missing raw":
                    raw.pop()
                elif mutation == "duplicate raw":
                    raw.append(raw[0])
                elif mutation == "aggregate tokens":
                    summary["total_usage_completion_tokens"]["avg"] = 16
                elif mutation == "record tokens":
                    ordinary[0]["metrics"]["usage_completion_tokens"]["value"] = 8
                elif mutation == "cancelled":
                    summary["was_cancelled"] = True
                else:
                    summary["aiperf_version"] = "0.12.0"
                export(directory, ordinary, raw, summary)
                if mutation == "record tokens":
                    self.assertFalse(validate(directory)["protocol_valid"])
                else:
                    with self.assertRaises(ValueError):
                        validate(directory)

    def test_variable_input_lengths_are_validated_and_bound_to_client_config(self):
        mixed = MANIFEST.replace("input_tokens_stddev = 0", "input_tokens_stddev = 8")
        manifest = parse_manifest(mixed.encode())
        with tempfile.TemporaryDirectory() as tmp:
            _, _, summary = fixture(Path(tmp))
            actual = summary["input_config"]
            with self.assertRaisesRegex(ValueError, "effective datasets differs"):
                check_config(actual, manifest)
            actual["datasets"][0]["prompts"]["isl"]["stddev"] = 8
            check_config(actual, manifest)

        for field, values in (
            ("input_tokens_stddev", ("-1", "false", "0.5", '"8"')),
            ("output_tokens_stddev", ("1", "false")),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    invalid = MANIFEST.replace(f"{field} = 0", f"{field} = {value}")
                    with self.assertRaises(ValueError):
                        parse_manifest(invalid.encode())

    def test_rejects_effective_config_drift(self):
        for field in ("concurrency", "tokenizer", "SLO", "arrival"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                ordinary, raw, summary = fixture(directory)
                actual = summary["input_config"]
                if field == "concurrency":
                    actual["phases"][0]["concurrency"] = 8
                elif field == "tokenizer":
                    actual["tokenizer"]["revision"] = "different"
                elif field == "SLO":
                    actual["slos"] = {"request_latency": 9999}
                else:
                    actual["phases"][0].update(type="poisson", rate=10)
                export(directory, ordinary, raw, summary)
                with self.assertRaisesRegex(ValueError, "differs from manifest"):
                    validate(directory)

    def test_zero_successes_cannot_pass_and_keep_failure_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = fixture(directory)
            for record in ordinary:
                record["error"] = {"message": "overloaded", "code": 429}
            for record in raw:
                record.update(status=429, responses=[], error={"message": "overloaded"})
            summary["request_count"]["avg"] = 0
            summary["error_request_count"] = {"avg": 2, "unit": "requests"}
            export(directory, ordinary, raw, summary)
            report = validate(directory)
            self.assertFalse(report["protocol_valid"])
            self.assertEqual(report["failed_requests"], report["attempted_requests"])
            self.assertEqual(report["completion_tokens"], 0)

    def test_run_refuses_existing_or_raced_output_without_replacing_evidence(self):
        for raced in (False, True):
            with self.subTest(raced=raced), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                source = directory / "source"
                fixture(source)
                output = directory / "output"
                validation = output / "validation.json"
                if not raced:
                    output.mkdir()
                    validation.write_text("existing evidence")
                client = directory / "fake-aiperf"
                # A competing owner creates output during the version check,
                # before the runner's mkdir, without nondeterministic sleeps.
                client.write_text(
                    f"#!{sys.executable}\nfrom pathlib import Path\n"
                    + (
                        f"Path({str(output)!r}).mkdir()\nPath({str(validation)!r}).write_text('existing evidence')\n"
                        if raced
                        else ""
                    )
                    + "print('0.13.0')\n"
                )
                client.chmod(0o755)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "endpoint.py"),
                        "run",
                        str(source / "manifest.toml"),
                        "--output",
                        str(output),
                        "--aiperf",
                        str(client),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 1)
                self.assertEqual(validation.read_text(), "existing evidence")
                self.assertEqual((source / "manifest.toml").read_text(), MANIFEST)

    def test_cli_failure_invalidates_old_validation_without_touching_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ordinary, raw, summary = fixture(directory)
            export(directory, ordinary, raw, summary)
            (directory / "validation.json").write_text('{"protocol_valid":true}')
            (directory / "client-exit-code").write_text("23\n")
            result = subprocess.run(
                [sys.executable, str(ROOT / "endpoint.py"), "validate", str(directory)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(
                json.loads((directory / "validation.json").read_text())[
                    "protocol_valid"
                ]
            )
            self.assertEqual((directory / "client-exit-code").read_text(), "23\n")


if __name__ == "__main__":
    unittest.main()
