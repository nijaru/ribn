"""Protect reference provenance/history alignment without a model runtime dependency."""

import copy
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from gemma4_reference import MANIFEST, sha256
from gemma4_reference_components import validated_rows


class ComponentReferenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = tomllib.loads(MANIFEST.read_text())
        steps = [
            {
                "index": 0,
                "history_ids": [2, 30],
                "input_ids": [2, 30],
                "position_ids": [0, 1],
                "chosen": 31,
            },
            {
                "index": 1,
                "history_ids": [2, 30, 31],
                "input_ids": [31],
                "position_ids": [2],
                "chosen": 32,
            },
        ]
        greedy = {
            "name": "greedy",
            "policy": "greedy",
            "prompt_ids": [2, 30],
            "steps": steps,
        }
        forced = copy.deepcopy(greedy)
        forced.update(name="forced", policy="forced", forced_ids=[31])
        forced["steps"][0]["chosen"] = (
            99  # A forced history must not follow this winner.
        )
        self.source = {
            "schema_version": 1,
            "complete": True,
            "device": "cpu",
            "attention": "eager",
            "precision": "bf16",
            "checkpoint": self.manifest["checkpoint"],
            "manifest_sha256": sha256(MANIFEST),
            "files": self.manifest["files"],
            "packages": {
                name: self.manifest["reference"][name]
                for name in ("torch", "transformers", "safetensors", "tokenizers")
            },
            "implementation": {
                "verified_file_sha256": self.manifest["reference"]["source_sha256"]
            },
            "cases": [greedy, forced],
        }

    def test_greedy_and_forced_inputs_keep_their_actual_common_histories(self):
        rows = validated_rows(self.source, self.manifest, "bf16")
        self.assertEqual(
            [
                (row["name"], row["position"], row["original_input_tokens"])
                for row in rows
            ],
            [
                ("greedy.0", 1, 2),
                ("greedy.1", 2, 1),
                ("forced.0", 1, 2),
                ("forced.1", 2, 1),
            ],
        )
        self.assertEqual(rows[-1]["history_ids"], [2, 30, 31])

    def test_chunked_prefill_rows_keep_their_prefixes_and_prediction_indices(self):
        source = copy.deepcopy(self.source)
        source["prefill_chunk_size"] = 1
        for case in source["cases"]:
            case["prefill_intermediates"] = [
                {
                    "index": 0,
                    "history_ids": [2],
                    "input_ids": [2],
                    "position_ids": [0],
                    "chosen": 99,  # Not the next prompt token or a generation choice.
                }
            ]
            case["steps"][0].update(input_ids=[30], position_ids=[1])
        rows = validated_rows(source, self.manifest, "bf16")
        self.assertEqual(
            [
                (row["name"], row["position"], row["original_input_tokens"])
                for row in rows
            ],
            [
                ("greedy.prefill.0", 0, 1),
                ("greedy.0", 1, 1),
                ("greedy.1", 2, 1),
                ("forced.prefill.0", 0, 1),
                ("forced.0", 1, 1),
                ("forced.1", 2, 1),
            ],
        )
        missing = copy.deepcopy(source)
        missing["cases"][0]["prefill_intermediates"] = []
        wrong_prefix = copy.deepcopy(source)
        wrong_prefix["cases"][0]["prefill_intermediates"][0].update(
            history_ids=[4], input_ids=[4]
        )
        wrong_size = copy.deepcopy(source)
        wrong_size["prefill_chunk_size"] = 2
        for bad in (missing, wrong_prefix, wrong_size):
            with self.subTest(source=bad), self.assertRaises(ValueError):
                validated_rows(bad, self.manifest, "bf16")

    def test_incompatible_provenance_and_mixed_histories_reject(self):
        bad_history = copy.deepcopy(self.source)
        step = bad_history["cases"][0]["steps"][1]
        step.update(history_ids=[2, 30, 33], input_ids=[33])
        bad_positions = copy.deepcopy(self.source)
        bad_positions["cases"][0]["steps"][1]["position_ids"] = [1]
        bad_provenance = copy.deepcopy(self.source)
        bad_provenance["implementation"]["verified_file_sha256"] = {}
        incomplete = copy.deepcopy(self.source)
        incomplete["complete"] = False
        wrong_precision = copy.deepcopy(self.source)
        wrong_precision["precision"] = "fp32"
        for source in (
            bad_history,
            bad_positions,
            bad_provenance,
            incomplete,
            wrong_precision,
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                validated_rows(source, self.manifest, "bf16")

    def test_cli_retains_incomplete_input_failure_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, corpus, output = (
                root / "snapshot",
                root / "input",
                root / "output",
            )
            snapshot.mkdir()
            corpus.mkdir()
            (corpus / "reference.json").write_text(
                json.dumps({"schema_version": 1, "complete": False})
            )
            command = [
                sys.executable,
                str(MANIFEST.parent.parent / "gemma4_reference.py"),
                "--snapshot",
                str(snapshot),
                "--components-from",
                str(corpus),
                "--output",
                str(output),
                "--precision",
                "bf16",
            ]
            invalid = subprocess.run(
                command + ["--threads", "0"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(invalid.returncode, 1)
            self.assertIn("thread count must be positive", invalid.stderr)
            self.assertFalse(output.exists())
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=10
            )
            self.assertEqual(result.returncode, 1)
            report = json.loads((output / "reference.json").read_text())
            self.assertFalse(report["complete"])
            self.assertIn("incomplete/unknown corpus", report["error"])
            self.assertNotIn("packages", report)
            self.assertFalse((output / "tensors.safetensors").exists())
            # The same CLI cannot replace even a failed collection.
            before = (output / "reference.json").read_bytes()
            again = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=10
            )
            self.assertEqual(again.returncode, 1)
            self.assertEqual((output / "reference.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
