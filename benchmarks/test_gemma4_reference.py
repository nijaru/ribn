"""Protect reference provenance and failure evidence without loading model weights."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gemma4_reference import prompt_chunks, verify_files, verify_reference_sources

SCRIPT = Path(__file__).with_name("gemma4_reference.py")
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


class ReferenceEvidenceTests(unittest.TestCase):
    def test_prompt_chunking_preserves_order_and_rejects_nonpositive_sizes(self):
        prompt = [1, 5, 3, 2, 8, 4, 6, 7, 9]
        self.assertEqual(prompt_chunks(prompt, None), [prompt])
        self.assertEqual(prompt_chunks(prompt, 4), [[1, 5, 3, 2], [8, 4, 6, 7], [9]])
        self.assertEqual(prompt_chunks(prompt, 20), [prompt])
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                prompt_chunks(prompt, size)
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            prompt_chunks([], 4)

    def test_package_verification_rejects_same_size_corruption_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "model.safetensors"
            # Opaque bytes are deliberate: this tests provenance, not the format parser.
            path.write_bytes(b"abc")
            expected = {path.name: {"sha256": ABC_SHA256, "size_bytes": 3}}
            self.assertEqual(verify_files(root, expected), expected)
            path.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "wrong SHA256"):
                verify_files(root, expected)
            path.write_bytes(b"ab")
            with self.assertRaisesRegex(ValueError, "wrong size"):
                verify_files(root, expected)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing=.*model.safetensors"):
                verify_files(root, expected)

    def test_unverified_loader_sidecars_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"abc")
            expected = {"model.safetensors": {"sha256": ABC_SHA256, "size_bytes": 3}}
            templates = root / "chat_templates"
            templates.mkdir()
            (templates / "default.jinja").write_text("unverified framing")
            with self.assertRaisesRegex(ValueError, "extra=.*chat_templates"):
                verify_files(root, expected)
            (templates / "default.jinja").unlink()
            templates.rmdir()
            (root / "model.safetensors.index.json").write_text("{}")
            with self.assertRaisesRegex(
                ValueError, "extra=.*model.safetensors.index.json"
            ):
                verify_files(root, expected)

    def test_reference_source_hashes_reject_same_version_local_modifications(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "modeling.py"
            source.write_bytes(b"abc")
            expected = {source.name: ABC_SHA256}
            self.assertEqual(verify_reference_sources(root, expected), expected)
            source.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "differs from the upstream pin"):
                verify_reference_sources(root, expected)

    def test_cli_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, output = root / "snapshot", root / "evidence"
            snapshot.mkdir()
            output.mkdir()
            sentinel = output / "reference.json"
            sentinel.write_bytes(b"earlier evidence")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(sentinel.read_bytes(), b"earlier evidence")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_rotary_cli_refuses_existing_output_before_importing_model_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reference.json"
            output.write_bytes(b"earlier evidence")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT.with_name("gemma4_rotary_reference.py")),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("output already exists", result.stderr)
            self.assertEqual(output.read_bytes(), b"earlier evidence")

    def test_chunk_size_is_not_silently_ignored_by_component_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "evidence"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--snapshot",
                    str(root / "snapshot"),
                    "--output",
                    str(output),
                    "--components-from",
                    str(root / "corpus"),
                    "--prefill-chunk-size",
                    "4",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("not --components-from", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_records_incomplete_source_failure_before_importing_model_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, output = root / "snapshot", root / "evidence"
            snapshot.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            report = json.loads((output / "reference.json").read_text())
            self.assertFalse(report["complete"])
            self.assertIn("config.json", report["error"])
            self.assertNotIn("packages", report)
            self.assertFalse((output / "tensors.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
