"""Frozen endpoint QA must preserve whole answers, failures and per-question losses."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import gemma4_endpoint_quality as quality
from test_endpoint import export, message
from test_endpoint_trace import TRACE_MANIFEST, trace_fixture


def endpoint_fixture(directory):
    ordinary, raw, summary = trace_fixture(directory)
    cases = [
        {
            "id": f"req-{i}",
            "prompt": f"fixture prompt {i}",
            "prompt_ids": [2] * 16,
            "answers": [{"text": "hello world"}],
        }
        for i in range(2)
    ]
    package = tomllib.loads(quality.MANIFEST.read_text())
    revision = package["checkpoint"]["revision"]
    manifest = (
        TRACE_MANIFEST.replace("output_tokens_mean = 8", "output_tokens_mean = 64")
        .replace('revision = "fixture"', f'revision = "{revision}"')
        .replace(
            '"' + "0" * 64 + '"',
            json.dumps(package["files"]["model.safetensors"]["sha256"]),
        )
        .replace('dtype = "none"', 'dtype = "bf16"')
    )
    summary["input_config"]["tokenizer"]["revision"] = revision
    trace = quality.trace_bytes(cases)
    (directory / "schedule.jsonl").write_bytes(trace)
    (directory / "manifest.toml").write_text(manifest)
    (directory / "invocation.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
                "trace_sha256": hashlib.sha256(trace).hexdigest(),
            }
        )
    )
    for record in raw:
        record["payload"]["max_tokens"] = 64
    export(directory, ordinary, raw, summary)
    return cases, ordinary, raw, summary


class EndpointQualityTests(unittest.TestCase):
    def test_complete_unordered_exports_and_fragmented_answers_are_scored_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cases, ordinary, raw, summary = endpoint_fixture(d)
            raw[0]["responses"][0]["packets"][0]["value"] = json.dumps(
                {
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": "hello ", "finish_reason": None}],
                }
            )
            fragment = message(
                {
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": "world", "finish_reason": "stop"}],
                }
            )
            raw[0]["responses"].insert(1, fragment)
            export(d, ordinary[::-1], raw[::-1], summary)
            result = quality.analyze_run(d, cases)
            self.assertTrue(result["complete"])
            self.assertEqual(result["scores"]["exact_matches"], 2)
            self.assertEqual(result["scores"]["mean_f1"], "1")
            self.assertEqual(result["rows"][0]["text"], "hello world")
            self.assertEqual(result["rows"][0]["usage"]["completion_tokens"], 2)
            self.assertIn("client/profile_export_raw.jsonl", result["source_sha256"])
            compared = quality.assess(result, result)
            self.assertTrue(compared["quality_sample_pass"])
            self.assertFalse(compared["input_token_ids_attested"])
            self.assertFalse(compared["device_execution_attested"])
            self.assertFalse(compared["population_quality_equivalence"])

    def test_unmanifested_custom_stops_cannot_improve_the_quality_screen(self):
        for key, value in (("stop", "world"), ("stop_token_ids", [42])):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                cases, ordinary, raw, summary = endpoint_fixture(directory)
                raw[0]["payload"][key] = value
                export(directory, ordinary, raw, summary)
                with self.assertRaisesRegex(ValueError, "custom generation stops"):
                    quality.analyze_run(directory, cases)
                raw[0]["payload"][key] = []
                export(directory, ordinary, raw, summary)
                self.assertTrue(quality.analyze_run(directory, cases)["complete"])

    def test_whole_capped_answer_is_not_extracted_or_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cases, ordinary, raw, summary = endpoint_fixture(d)
            packet = json.loads(raw[0]["responses"][0]["packets"][0]["value"])
            packet["choices"][0].update(
                text="hello world\nExplanation", finish_reason="length"
            )
            raw[0]["responses"][0]["packets"][0]["value"] = json.dumps(packet)
            usage = {"prompt_tokens": 16, "completion_tokens": 64, "total_tokens": 80}
            raw[0]["responses"][1]["packets"][0]["value"] = json.dumps(
                {"object": "text_completion", "choices": [], "usage": usage}
            )
            metrics = ordinary[0]["metrics"]
            for field, value in usage.items():
                metrics[f"usage_{field}"]["value"] = value
            metrics["output_sequence_length"]["value"] = 64
            for field in ("completion_tokens", "total_tokens"):
                summary[f"total_usage_{field}"]["avg"] += 62
            export(d, ordinary, raw, summary)
            row = quality.analyze_run(d, cases)["rows"][0]
            self.assertFalse(row["em"])
            self.assertEqual(row["f1"], "4/5")
            self.assertEqual(row["finish_reason"], "length")
            self.assertEqual(row["text"], "hello world\nExplanation")

    def test_failed_or_invalid_answer_cannot_become_a_zero_or_survivor_mean(self):
        for mode in ("transport", "empty-eos", "late-error", "prompt-count", "truncated"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                cases, ordinary, raw, summary = endpoint_fixture(d)
                if mode in ("transport", "empty-eos"):
                    if mode == "transport":
                        ordinary[0]["error"] = {"type": "http", "message": "overloaded"}
                        raw[0].update(
                            status=429, responses=[], error={"message": "raw failure"}
                        )
                    else:
                        # Native AIPerf 0.13 rejects an otherwise complete EOS-only
                        # reply. Retain its actual answer/usage, not a zero score.
                        error = {"type": "InvalidInferenceResultError"}
                        ordinary[0].update(error=error, metrics={})
                        raw[0]["error"] = error
                        raw[0]["responses"][0] = message(
                            {
                                "object": "text_completion",
                                "choices": [
                                    {"index": 0, "text": "", "finish_reason": "stop"}
                                ],
                            }
                        )
                        raw[0]["responses"][1] = message(
                            {
                                "object": "text_completion",
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": 16,
                                    "completion_tokens": 1,
                                    "total_tokens": 17,
                                },
                            }
                        )
                    summary["request_count"]["avg"] = 1
                    summary["error_request_count"] = {"avg": 1, "unit": "requests"}
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        summary[f"total_usage_{field}"]["avg"] //= 2
                elif mode == "late-error":
                    raw[0]["responses"].insert(
                        1, message({"error": {"message": "fault"}})
                    )
                elif mode == "prompt-count":
                    cases[0]["prompt_ids"].pop()
                else:
                    raw[0]["responses"].pop()
                export(d, ordinary, raw, summary)
                result = quality.analyze_run(d, cases)
                self.assertFalse(result["complete"])
                self.assertIsNone(result["scores"])
                self.assertEqual(len(result["rows"]), 2)
                self.assertNotIn("f1", result["rows"][0])
                observed = result["rows"][0]["observed"]
                if mode in ("late-error", "truncated"):
                    self.assertEqual(observed["text"], "hello world")
                    self.assertEqual(observed["finish_reasons"], ["stop"])
                    self.assertEqual(
                        observed["usage_frames"][0]["completion_tokens"], 2
                    )
                if mode == "empty-eos":
                    self.assertEqual(observed["text"], "")
                    self.assertEqual(observed["finish_reasons"], ["stop"])
                    self.assertEqual(observed["usage_frames"][0]["completion_tokens"], 1)
                    self.assertEqual(
                        result["rows"][0]["raw_error"]["type"],
                        "InvalidInferenceResultError",
                    )
                if mode == "late-error":
                    self.assertEqual(observed["errors"], [{"message": "fault"}])
                if mode == "transport":
                    self.assertEqual(
                        result["rows"][0]["raw_error"], {"message": "raw failure"}
                    )
                compared = quality.assess(result, result)
                self.assertFalse(compared["quality_sample_pass"])
                self.assertIsNone(compared["losses"])

    def test_malformed_normalized_frames_remain_unscored_evidence(self):
        for responses in (
            [None],
            [{"perf_ns": 1_100_000_000, "packets": None}],
            [message({"object": "text_completion", "choices": "broken"})],
            [message({"object": "text_completion", "choices": [None]})],
            [message({"object": "text_completion", "choices": [{"text": None}]})],
        ):
            with (
                self.subTest(responses=responses),
                tempfile.TemporaryDirectory() as tmp,
            ):
                directory = Path(tmp)
                cases, ordinary, raw, summary = endpoint_fixture(directory)
                raw[0]["responses"] = responses
                export(directory, ordinary, raw, summary)
                report = quality.analyze_run(directory, cases)
                self.assertFalse(report["complete"])
                self.assertNotIn("f1", report["rows"][0])
                self.assertTrue(report["rows"][0]["observed"]["malformed_data"])

    def test_changed_prompt_or_incomplete_coverage_rejects_before_scoring(self):
        for mode in ("text", "missing", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                cases, ordinary, raw, summary = endpoint_fixture(d)
                if mode == "text":
                    raw[0]["payload"]["prompt"] += " changed"
                elif mode == "missing":
                    raw.pop()
                    ordinary.pop()
                else:
                    raw[1]["metadata"]["conversation_id"] = "req-0"
                    ordinary[1]["metadata"] = copy.deepcopy(raw[1]["metadata"])
                export(d, ordinary, raw, summary)
                with self.assertRaises(ValueError):
                    quality.analyze_run(d, cases)

    def test_equal_aggregate_scores_do_not_hide_per_question_losses(self):
        def run(texts):
            rows = []
            for i, text in enumerate(texts):
                em, f1 = quality.answer_score(text, [{"text": "cat dog"}])
                rows.append({"id": str(i), "em": em, "f1": str(f1)})
            return {"complete": True, "rows": rows}

        before = run(["cat dog", "wrong", "cat dog bird"])
        after = run(["wrong", "cat dog", "cat dog bird extra"])
        report = quality.assess(before, after)
        self.assertTrue(report["complete"])
        self.assertFalse(report["quality_sample_pass"])
        self.assertEqual(report["losses"], ["0", "2"])
        self.assertEqual(report["gains"], ["1"])

    def test_rendered_prompt_pin_is_independent_of_the_frozen_token_ids(self):
        cases = [{"id": "x", "prompt": "original", "prompt_ids": [2]}]
        data = json.dumps([["x", "original"]], separators=(",", ":")).encode()
        with (
            patch.object(
                quality, "read_json", side_effect=[(b"dataset", {}), (b"cases", cases)]
            ),
            patch.object(quality, "DATASET_SHA256", quality.digest(b"dataset")),
            patch.object(quality, "validate_corpus") as validate,
            patch.object(quality, "RENDERED_PROMPTS_SHA256", quality.digest(data)),
        ):
            self.assertEqual(quality.corpus(Path("dataset"), Path("cases")), cases)
            validate.assert_called_once()
        cases[0]["prompt"] += " second template"
        with (
            patch.object(
                quality, "read_json", side_effect=[(b"dataset", {}), (b"cases", cases)]
            ),
            patch.object(quality, "DATASET_SHA256", quality.digest(b"dataset")),
            patch.object(quality, "validate_corpus"),
            patch.object(quality, "RENDERED_PROMPTS_SHA256", quality.digest(data)),
            self.assertRaisesRegex(ValueError, "rendered prompts changed"),
        ):
            quality.corpus(Path("dataset"), Path("cases"))

    def test_provenance_failure_after_preparation_cannot_report_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "quality",
                        "prepare",
                        "--dataset",
                        "dataset.json",
                        "--cases",
                        "cases.json",
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(quality, "corpus", return_value=[]),
                patch.object(quality, "file_hash", side_effect=OSError("unavailable")),
                patch("builtins.print"),
            ):
                self.assertEqual(quality.main(), 1)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["error"], "unavailable")

    def test_comparison_provenance_failure_clears_an_earlier_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            run = {"complete": True, "rows": [{"id": "x", "em": True, "f1": "1"}]}
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "quality",
                        "compare",
                        "--dataset",
                        "dataset.json",
                        "--cases",
                        "cases.json",
                        "--baseline",
                        "baseline",
                        "--candidate",
                        "candidate",
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(quality, "corpus", return_value=[]),
                patch.object(quality, "analyze_run", return_value=run),
                patch.object(quality, "file_hash", side_effect=OSError("unavailable")),
                patch("builtins.print"),
            ):
                self.assertEqual(quality.main(), 1)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["quality_sample_pass"])

    def test_cli_failure_is_incomplete_and_never_replaces_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            command = [
                sys.executable,
                str(Path(quality.__file__)),
                "prepare",
                "--dataset",
                str(Path(tmp) / "missing.json"),
                "--cases",
                str(Path(tmp) / "missing-cases.json"),
                "--output",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, check=False)
            self.assertEqual(result.returncode, 1)
            report = output / "report.json"
            old = report.read_bytes()
            self.assertFalse(json.loads(old)["complete"])
            result = subprocess.run(command, capture_output=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(report.read_bytes(), old)


if __name__ == "__main__":
    unittest.main()
