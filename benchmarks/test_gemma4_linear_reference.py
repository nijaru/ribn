"""Protection for projection orientation, independent accumulation and failure evidence."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gemma4_linear_reference import fsum_linear, validate_tokens


class LinearReferenceTests(unittest.TestCase):
    def test_exact_bf16_products_keep_cancellation_residual(self):
        # Each operand is BF16-exact; a sequential binary32 dot loses the unit.
        actual, absolute = fsum_linear(
            [[float(2**24), 1.0, -float(2**24)]], iter([[1.0, 1.0, 1.0]])
        )
        self.assertEqual(actual, [[1]])
        self.assertEqual(absolute, [[2**25 + 1]])

    def test_row_major_non_square_projection(self):
        actual, absolute = fsum_linear(
            [[2, -3], [4, 5]], iter([[1, 2], [-2, 0], [0, 4]])
        )
        self.assertEqual(actual, [[-4, -4, -12], [14, -8, 20]])
        self.assertEqual(absolute, [[8, 4, 12], [14, 8, 20]])

    def test_reject_shapes_that_would_silently_truncate(self):
        for inputs, weights in [
            ([], [[1]]),
            ([[]], [[]]),
            ([[1], [2, 3]], [[1]]),
            ([[1]], []),
            ([[1]], [[1, 2]]),
        ]:
            with (
                self.subTest(inputs=inputs, weights=weights),
                self.assertRaises(ValueError),
            ):
                fsum_linear(inputs, iter(weights))

    def test_token_boundaries(self):
        validate_tokens([0, 7, 7, 15], vocab=16)
        for tokens in ([], [True], [-1], [16], [0] * 5):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validate_tokens(tokens, vocab=16)

    def test_cli_failure_is_retained_without_loading_runtime_or_overwriting(self):
        script = Path(__file__).with_name("gemma4_linear_reference.py")
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output = root / "output"
            # Nonexistent snapshot and -S: importing any external runtime would
            # fail first. Invalid IDs must instead leave their own failure report.
            command = [
                sys.executable,
                "-S",
                str(script),
                "--snapshot",
                str(root / "missing"),
                "--output",
                str(output),
                "--token-ids",
                "-1",
            ]
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=15
            )
            self.assertNotEqual(result.returncode, 0)
            path = output / "report.json"
            original = path.read_bytes()
            report = json.loads(original)
            self.assertFalse(report["complete"])
            self.assertIn("nonnegative integers", report["error"])
            self.assertNotIn("packages", report)
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=15
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
