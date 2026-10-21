"""Protect common histories and failure evidence without claiming CUDA execution."""

import copy
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from gemma4_sglang import MANIFEST, VOCAB, reference_cases, sequence, timing_cases

SCRIPT = Path(__file__).with_name("gemma4_sglang.py")
MANIFEST_DATA = tomllib.loads(MANIFEST.read_text())


def corpus():
    cases = []
    for name, length in (("chat", 18), ("window_511", 511), ("window_513", 513)):
        prompt = [2, 7, 11] * (length // 3) + [2, 7, 11][: length % 3]
        history = []
        steps = []
        for index, inputs in enumerate((prompt, [31], [47], [59])):
            history.extend(inputs)
            steps.append(
                {
                    "index": index,
                    "input_ids": list(inputs),
                    "history_ids": list(history),
                    "position_ids": list(
                        range(len(history) - len(inputs), len(history))
                    ),
                    # Deliberately not the next input: history, not newly observed
                    # predictions, is authoritative for this adapter.
                    "chosen": 900 + index,
                }
            )
        cases.append({"name": name, "prompt_ids": prompt, "steps": steps})
    return {
        "complete": True,
        "precision": "bf16",
        "device": "cpu",
        "attention": "eager",
        "checkpoint": MANIFEST_DATA["checkpoint"],
        "cases": cases,
    }


class SglangInputTests(unittest.TestCase):
    def test_quality_and_timing_force_consumed_inputs_not_predictions(self):
        reference = corpus()
        original = copy.deepcopy(reference)
        cases = reference_cases(reference, MANIFEST_DATA)
        self.assertEqual(reference, original)
        for case in cases:
            self.assertEqual(case["inputs"][1:], [[31], [47], [59]])
        timing = timing_cases(cases)
        self.assertEqual(
            [len(case["inputs"][0]) for case in timing], [1, 8, 18, 32, 64, 65, 128]
        )
        prompt = reference["cases"][0]["prompt_ids"]
        for case in timing:
            self.assertEqual(
                case["inputs"][0],
                [prompt[i % 18] for i in range(len(case["inputs"][0]))],
            )
            self.assertEqual(case["inputs"][1:], [[31], [47], [59]])
        timing[-1]["inputs"][1][0] = 99
        self.assertEqual(cases[0]["inputs"][1], [31])

    def test_rejects_discontinuous_histories_and_invalid_tokens(self):
        for field, value in (
            ("input_ids", [True]),
            ("input_ids", [-1]),
            ("input_ids", [262144]),
            ("input_ids", [1.0]),
            ("input_ids", [31, 47]),
            ("history_ids", [31]),
            ("position_ids", [0]),
            ("index", 0),
        ):
            with self.subTest(field=field, value=value):
                reference = corpus()
                reference["cases"][1]["steps"][1][field] = value
                with self.assertRaises(ValueError):
                    reference_cases(reference, MANIFEST_DATA)
        for change in (
            {"complete": False},
            {"precision": "fp32"},
            {"checkpoint": {}},
            {"device": "cuda"},
            {"attention": "flash_attention_2"},
        ):
            reference = corpus() | change
            with self.subTest(change=change), self.assertRaises(ValueError):
                reference_cases(reference, MANIFEST_DATA)

    def test_later_forward_failure_keeps_completed_samples(self):
        # Replace only the CUDA forward/completion boundary. Real host arrays and
        # the production evidence writer exercise incremental sample retention.
        class HostArray(np.ndarray):
            def cpu(self):
                return self

            def numpy(self):
                return np.asarray(self)

        host = np.zeros((1, VOCAB), dtype=np.float32).view(HostArray)
        result = SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=host),
            can_run_graph=False,
        )
        batch = SimpleNamespace(seq_lens_cpu=np.array([18]))
        forward_batch = SimpleNamespace(positions=np.arange(18).view(HostArray))
        torch = SimpleNamespace(float32=np.dtype("float32"), isfinite=np.isfinite)
        wrapped = SimpleNamespace(
            synchronize=lambda: None,
            clear=lambda: None,
            torch_runner=None,
        )
        case = reference_cases(corpus(), MANIFEST_DATA)[0]
        report = {"complete": False, "runs": []}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "gemma4_sglang.forward",
                side_effect=[
                    (result, batch, forward_batch),
                    RuntimeError("device failure"),
                ],
            ),
        ):
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "device failure"):
                sequence(torch, None, wrapped, case, output, report)
            saved = json.loads((output / "report.json").read_text())
            self.assertFalse(saved["complete"])
            self.assertEqual(len(saved["runs"]), 1)
            run = saved["runs"][0]
            self.assertFalse(run["complete"])
            self.assertEqual(len(run["steps"]), 1)
            self.assertEqual(run["steps"][0]["input_ids"], case["inputs"][0])
            self.assertEqual((output / "chat.0.f32le").stat().st_size, VOCAB * 4)

    def invoke(self, root, output, reference):
        path = root / "reference.json"
        path.write_text(json.dumps(reference))
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--snapshot",
                str(root / "unused-package"),
                "--reference",
                str(path),
                "--output",
                str(output),
                "--validate-only",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_real_cli_validation_is_not_marked_as_completed_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "plan"
            result = self.invoke(root, output, corpus())
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["validated_inputs"])
            self.assertFalse(report["complete"])
            self.assertEqual(report["runs"], [])
            self.assertEqual(
                (report["plan"]["trials"], report["plan"]["warmup_trial"]), (4, 0)
            )
            self.assertEqual(list(output.iterdir()), [output / "report.json"])

    def test_failure_is_retained_and_retry_cannot_overwrite_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "failed"
            reference = corpus()
            reference["cases"][2]["steps"][3]["history_ids"][-1] = 100
            result = self.invoke(root, output, reference)
            self.assertNotEqual(result.returncode, 0)
            saved = (output / "report.json").read_bytes()
            report = json.loads(saved)
            self.assertFalse(report["complete"])
            self.assertIn("discontinuous reference history", report["error"])
            retry = self.invoke(root, output, corpus())
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual((output / "report.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
