"""Analyzer data/numerical contracts only; synthetic captures are not GPU evidence."""

import copy
import json
import math
import struct
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np
from gemma4_compare import compare, metrics
from gemma4_reference import MANIFEST, sha256
from gemma4_sglang import reference_cases, timing_cases
from test_gemma4_sglang import corpus


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.manifest = tomllib.loads(MANIFEST.read_text())
        cls.source = cls.directory / "capture.py"
        cls.source.write_text("# synthetic data fixture, no device execution\n")
        cls.reference_path = cls.directory / "reference.json"
        cls.report_path = cls.directory / "report.json"
        cls.tensors = cls.directory / "tensors.safetensors"
        cls.reference = corpus() | {
            "manifest_sha256": sha256(MANIFEST),
            "packages": {
                key: cls.manifest["reference"][key]
                for key in ("torch", "transformers", "safetensors", "tokenizers")
            },
            "implementation": {
                "declared_upstream_commit": cls.manifest["reference"][
                    "transformers_commit"
                ],
                "verified_file_sha256": cls.manifest["reference"]["source_sha256"],
            },
            "files": cls.manifest["files"],
        }
        header, data = {}, bytearray()
        for case in cls.reference["cases"]:
            for step in case["steps"]:
                row = np.zeros(262144, dtype="<u2")
                row[step["chosen"]] = 0x3F80
                name = f"{case['name']}.{step['index']}"
                header[name + ".logits"] = {
                    "dtype": "BF16",
                    "shape": [262144],
                    "data_offsets": [len(data), len(data) + row.nbytes],
                }
                data.extend(row.tobytes())
                values = (row.astype(np.uint32) << 16).view(np.float32)
                if name == "chat.0":
                    values[17] = np.float32(0.00123)  # Not representable in BF16.
                (cls.directory / (name + ".f32le")).write_bytes(values.tobytes())
        cls.header = header
        cls.data = data
        cls.write_tensors(header)
        cls.reference_path.write_text(json.dumps(cls.reference))
        cases = reference_cases(cls.reference, cls.manifest)
        timing = timing_cases(cases)
        cls.report = {
            "complete": True,
            "validation_only": False,
            "checkpoint": cls.manifest["checkpoint"],
            "files": cls.manifest["files"],
            "script_sha256": sha256(cls.source),
            "manifest_sha256": sha256(MANIFEST),
            "reference_sha256": sha256(cls.reference_path),
            "clock": "input preparation through completed full FP32 host logits; no sampling",
            "plan": {
                "quality": cases,
                "timing": timing,
                "trials": 4,
                "warmup_trial": 0,
            },
            "runs": [],
        }
        schedule = [(case, None) for case in cases] + [
            (case, trial) for case in timing for trial in range(4)
        ]
        for case, trial in schedule:
            run = {
                "name": case["name"],
                "trial": trial,
                "warmup": trial == 0,
                "complete": True,
                "steps": [],
            }
            position = 0
            for index, ids in enumerate(case["inputs"]):
                before = position
                position += len(ids)
                record = {
                    "step": index,
                    "input_ids": ids,
                    "history_length": position,
                    "position_ids": list(range(before, position)),
                    "shape": [262144],
                    "dtype": "float32",
                    "finite": True,
                    "chosen": 900 + index,
                    "elapsed_ns": 100
                    if trial is None
                    else (1000 if trial == 0 else trial + 1) * 1000000,
                }
                if trial is None:
                    path = cls.directory / f"{case['name']}.{index}.f32le"
                    record["logits"] = {
                        "path": path.name,
                        "sha256": sha256(path),
                        "elements": 262144,
                        "dtype": "float32-le",
                    }
                run["steps"].append(record)
            cls.report["runs"].append(run)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def write_tensors(cls, header):
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        cls.tensors.write_bytes(struct.pack("<Q", len(encoded)) + encoded + cls.data)
        cls.reference["tensor_file"] = {
            "size_bytes": cls.tensors.stat().st_size,
            "sha256": sha256(cls.tensors),
        }

    def run_comparison(self, report=None):
        self.report_path.write_text(
            json.dumps(self.report if report is None else report)
        )
        return compare(self.report_path, self.source, self.reference_path, self.tensors)

    def test_real_cli_preserves_fp32_diagnostics_and_excludes_retained_warmups(self):
        self.run_comparison()
        output = self.directory / "comparison.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("gemma4_compare.py")),
            "--report",
            str(self.report_path),
            "--source",
            str(self.source),
            "--reference",
            str(self.reference_path),
            "--tensors",
            str(self.tensors),
            "--output",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(output.read_text())
        self.assertEqual(actual["summary"]["rows"], 12)
        self.assertEqual(actual["summary"]["fp32_bits_equal"], 11)
        self.assertEqual(actual["summary"]["winners_equal"], 12)
        self.assertEqual(
            actual["summary"]["max_over_rows"]["max_abs"], float(np.float32(0.00123))
        )
        for timing in actual["timing"].values():
            self.assertEqual(timing["prefill_ns"], {"samples": 3, "median_ms": 3.0})
            self.assertEqual(timing["decode_ns"], {"samples": 9, "median_ms": 3.0})
        again = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(json.loads(output.read_text()), actual)

    def test_metadata_rejects_failed_wrong_identity_and_noncommon_histories(self):
        changes = [
            lambda r: r.update(complete=False),
            lambda r: r.update(validation_only=True),
            lambda r: r.update(error="late failure"),
            lambda r: r.update(checkpoint={}),
            lambda r: r.update(script_sha256="0" * 64),
            lambda r: r.update(reference_sha256="0" * 64),
            lambda r: r.update(manifest_sha256="0" * 64),
            lambda r: r.update(clock="BF16 host row"),
            lambda r: r["runs"].pop(),
            lambda r: r["runs"][0].update(complete=False),
            lambda r: r["runs"][0]["steps"][1].update(input_ids=[99]),
            lambda r: r["runs"][0]["steps"][1].update(position_ids=[0]),
            lambda r: r["runs"][0]["steps"][1].update(dtype="torch.bfloat16"),
            lambda r: r["runs"][0]["steps"][1].update(elapsed_ns=True),
            lambda r: r["runs"][3].update(trial=False),
            lambda r: r["runs"][3].update(warmup=False),
        ]
        for change in changes:
            report = copy.deepcopy(self.report)
            change(report)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_comparison(report)

    def test_row_integrity_encoding_finiteness_and_directory_confinement(self):
        record = self.report["runs"][0]["steps"][0]
        path = self.directory / record["logits"]["path"]
        original = path.read_bytes()
        try:
            path.write_bytes(original[:-4])
            with self.assertRaisesRegex(ValueError, "byte count"):
                self.run_comparison()
            path.write_bytes(b"\x00\x00\x80\x7f" + original[4:])
            with self.assertRaisesRegex(ValueError, "hash"):
                self.run_comparison()
            report = copy.deepcopy(self.report)
            report["runs"][0]["steps"][0]["logits"]["sha256"] = sha256(path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.run_comparison(report)
            path.write_bytes(original)
            report = copy.deepcopy(self.report)
            report["runs"][0]["steps"][0]["chosen"] = 42
            with self.assertRaisesRegex(ValueError, "winner"):
                self.run_comparison(report)
            report["runs"][0]["steps"][0]["logits"]["path"] = "../escape.f32le"
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.run_comparison(report)
        finally:
            path.write_bytes(original)

    def test_authenticated_reference_header_must_still_have_valid_bf16_ranges(self):
        for change in (
            lambda h: h["chat.0.logits"].update(dtype="F32"),
            lambda h: h["chat.0.logits"].update(shape=[1, 262144]),
            lambda h: h["chat.0.logits"].update(data_offsets=[0, 2]),
            lambda h: h["window_511.0.logits"].update(data_offsets=[0, 262144 * 2]),
        ):
            try:
                header = copy.deepcopy(self.header)
                change(header)
                self.write_tensors(header)
                self.reference_path.write_text(json.dumps(self.reference))
                report = copy.deepcopy(self.report)
                report["reference_sha256"] = sha256(self.reference_path)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.run_comparison(report)
            finally:
                self.write_tensors(self.header)
                self.reference_path.write_text(json.dumps(self.reference))


class MetricTests(unittest.TestCase):
    def test_independent_two_logit_distribution_and_signed_zero(self):
        result = metrics(
            np.array([0, 2], dtype=np.float32), np.zeros(2, dtype=np.float32)
        )
        self.assertAlmostEqual(
            result["kl_reference_to_observed"], math.log(math.cosh(1))
        )
        self.assertAlmostEqual(result["tv"], 0.5 - 1 / (1 + math.exp(2)))
        self.assertIsNone(result["relative_l2"])
        zero = np.array([-0.0, 0.0], dtype=np.float32)
        result = metrics(zero, np.zeros(2, dtype=np.float32))
        self.assertFalse(result["fp32_bits_equal"])
        self.assertEqual(result["max_abs"], 0)
        self.assertEqual(result["tv"], 0)
        for row in (
            np.array([np.nan, 0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                metrics(row, zero)


if __name__ == "__main__":
    unittest.main()
