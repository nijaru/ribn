"""Small and CUDA reference collection must never replace earlier evidence."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RotaryEvidenceTests(unittest.TestCase):
    def test_both_modes_refuse_existing_output_before_runtime_import(self):
        script = Path(__file__).with_name("gemma4_rotary_reference.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            output.write_bytes(b"earlier evidence\n")
            for extra in ([], ["--apply-cuda"]):
                result = subprocess.run(
                    [sys.executable, str(script), "--output", str(output), *extra],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("output already exists", result.stderr)
                self.assertEqual(output.read_bytes(), b"earlier evidence\n")


if __name__ == "__main__":
    unittest.main()
