"""Independent dyadic checks for the encoded-operand component certificate."""

import copy
import json
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import gemma4_linear_bound as bound
import numpy as np


def exact(bits):
    exponent = (int(bits) >> 7) & 255
    fraction = int(bits) & 127
    magnitude = (
        Fraction(128 + fraction) * Fraction(2) ** (exponent - 134)
        if exponent
        else Fraction(fraction) * Fraction(2) ** -133
    )
    return -magnitude if int(bits) & 0x8000 else magnitude


def vector(bits):
    return np.array(bits, dtype=np.uint16)


class EnclosureTests(unittest.TestCase):
    def test_certificates_contain_exact_dyadic_sums_across_the_finite_bf16_range(self):
        rng = random.Random(42317)
        special = [0, 0x8000, 1, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80]
        for k in (1, 2, 17, 64):
            for _ in range(12):
                x = vector(
                    [
                        rng.choice(special)
                        if i % 3 == 0
                        else rng.randrange(0x7F80) | (rng.randrange(2) << 15)
                        for i in range(k)
                    ]
                )
                w = vector(
                    [
                        [
                            rng.randrange(0x7F80) | (rng.randrange(2) << 15)
                            for _ in range(k)
                        ]
                        for _ in range(3)
                    ]
                )
                lo, hi, slo, shi, _, _ = bound.dot_enclosure(x, w)
                for row in range(3):
                    products = [
                        exact(a) * exact(b) for a, b in zip(x, w[row], strict=True)
                    ]
                    dot, magnitude = sum(products), sum(map(abs, products))
                    self.assertLessEqual(Fraction.from_float(float(lo[row])), dot)
                    self.assertGreaterEqual(Fraction.from_float(float(hi[row])), dot)
                    self.assertLessEqual(
                        Fraction.from_float(float(slo[row])), magnitude
                    )
                    self.assertGreaterEqual(
                        Fraction.from_float(float(shi[row])), magnitude
                    )

    def test_normal_products_cancellation_and_exact_zeros_certify(self):
        x = vector([[0x3F80, 0x3F80]])
        weight = vector([[0x3F80, 0x3F80], [0x3F80, 0xBF80], [0, 0], [0x3F00, 0x3F00]])
        output = vector([[0x4000, 0, 0x8000, 0x3F80]])
        report = bound.certify(x, weight, output)
        self.assertTrue(report["component_bound_acceptance"])
        self.assertEqual(
            report["counts"],
            {"accepted": 4, "violated": 0, "indeterminate": 0, "unsupported": 0},
        )
        output[0, 0] = 0x4080  # 4, not the exact dot 2.
        report = bound.certify(x, weight, output)
        self.assertEqual(report["counts"]["violated"], 1)
        self.assertFalse(report["component_bound_acceptance"])

    def test_reference_uncertainty_cannot_grant_an_optimistic_boundary_pass(self):
        # K=512 -> gamma32=1/(129*127). d=1, S=127^2 -> bound=1 exactly.
        # The true error of y=0 is on the boundary. A nonzero reference enclosure
        # cannot prove this pass: using its center or the upper allowed bound would.
        k = 512
        x = np.full(k, 0x3F80, dtype=np.uint16)
        weight = np.zeros((1, k), dtype=np.uint16)
        weight[0, :3] = vector([0x45FC, 0x3F80, 0xC5FC])  # 8064, 1, -8064
        products = [exact(a) * exact(b) for a, b in zip(x, weight[0], strict=True)]
        dot, magnitude = sum(products), sum(map(abs, products))
        error32 = Fraction(k, 2**23 - k) * magnitude
        self.assertEqual((dot, magnitude), (1, 16129))
        self.assertEqual(error32 + Fraction(1, 128) * (abs(dot) + error32), 1)
        report = bound.certify(x[None, :], weight, vector([[0]]))
        self.assertEqual(report["counts"]["indeterminate"], 1)
        self.assertEqual(report["counts"]["violated"], 0)
        self.assertFalse(report["component_bound_acceptance"])

    def test_passes_and_violations_respect_the_exact_bound_not_just_the_enclosure(self):
        rng = random.Random(853)
        for _ in range(40):
            x = vector(
                [
                    (rng.randrange(117, 138) << 7)
                    | rng.randrange(128)
                    | (rng.randrange(2) << 15)
                    for _ in range(32)
                ]
            )
            weight = vector(
                [
                    [
                        (rng.randrange(117, 138) << 7)
                        | rng.randrange(128)
                        | (rng.randrange(2) << 15)
                        for _ in range(32)
                    ]
                    for _ in range(3)
                ]
            )
            dots, magnitudes, outputs = [], [], []
            for row in weight:
                products = [exact(a) * exact(b) for a, b in zip(x, row, strict=True)]
                dot, magnitude = sum(products), sum(map(abs, products))
                dots.append(dot)
                magnitudes.append(magnitude)
                # Construct a result near the exact dot; this is not the oracle.
                bits = int.from_bytes(struct.pack("<f", float(dot)), "little")
                outputs.append(((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF)
            output = vector(outputs)
            certificate = bound.classify(x, weight, output)
            for column in range(3):
                accumulated = Fraction(32, 2**23 - 32) * magnitudes[column]
                limit = accumulated + Fraction(1, 128) * (
                    abs(dots[column]) + accumulated
                )
                error = abs(exact(output[column]) - dots[column])
                self.assertLessEqual(
                    Fraction.from_float(float(certificate["bound_lower"][column])),
                    limit,
                )
                self.assertGreaterEqual(
                    Fraction.from_float(float(certificate["bound_upper"][column])),
                    limit,
                )
                self.assertTrue(certificate["accepted"][column])
                self.assertLessEqual(error, limit)
            # A large representable perturbation must be a definite violation.
            output[0] ^= 0x8000
            certificate = bound.classify(x, weight, output)
            self.assertTrue(certificate["violated"][0])
            accumulated = Fraction(32, 2**23 - 32) * magnitudes[0]
            limit = accumulated + Fraction(1, 128) * (abs(dots[0]) + accumulated)
            self.assertGreater(abs(exact(output[0]) - dots[0]), limit)

    def test_subnormal_or_overflow_proof_gaps_reject_even_plausible_results(self):
        for x, weight, output in [
            ([0x0080], [[0x3F00]], [0x0040]),  # product quantum below FP32 normal
            ([1], [[0x7180]], [0x2F00]),  # BF16 subnormal operand, product 2^-33
            ([0x7F7F, 0x7F7F], [[0x3F80, 0xBF80]], [0]),  # S too large
        ]:
            with self.subTest(x=x):
                report = bound.certify(vector([x]), vector(weight), vector([output]))
                self.assertEqual(report["counts"]["unsupported"], 1)
                self.assertFalse(report["component_bound_acceptance"])
                self.assertEqual(report["counts"]["violated"], 0)

    def test_nonzero_subnormal_products_never_become_zero_certificates(self):
        # Both encodings are finite; their exact product is normal binary32.
        # The unsupported BF16 input must survive decoding. In particular,
        # a host FP32 flush mode must not turn this into a zero-S acceptance.
        x, weight, output = vector([[1]]), vector([[0x7180]]), vector([[0]])
        self.assertEqual(
            Fraction.from_float(float(bound.values(x)[0, 0])), Fraction(2) ** -133
        )
        report = bound.certify(x, weight, output)
        self.assertEqual(report["counts"]["unsupported"], 1)
        self.assertFalse(report["component_bound_acceptance"])

    def test_zero_product_sum_has_an_exact_zero_bound(self):
        x, weight = vector([[0]]), vector([[0x3F80], [0x8000]])
        good = bound.certify(x, weight, vector([[0, 0x8000]]))
        bad = bound.certify(x, weight, vector([[1, 0x8000]]))
        self.assertTrue(good["component_bound_acceptance"])
        self.assertEqual(bad["counts"]["violated"], 1)
        self.assertEqual(bad["exceptions"][0]["bound_upper"], 0)

    def test_chunking_preserves_all_rows_columns_and_verdicts(self):
        x = np.full((4, 17), 0x3F80, dtype=np.uint16)
        weight = np.zeros((137, 17), dtype=np.uint16)
        weight[:, 0] = 0x3F80
        output = np.full((4, 137), 0x3F80, dtype=np.uint16)
        output[3, 136] = 0x4000
        expected = bound.certify(x, weight, output, chunk_columns=1)
        for chunk in (3, 64):
            self.assertEqual(
                bound.certify(x, weight, output, chunk_columns=chunk), expected
            )
        self.assertEqual(expected["counts"]["accepted"], 547)
        self.assertEqual(expected["exceptions"][0]["row"], 3)
        self.assertEqual(expected["exceptions"][0]["column"], 136)

    def test_nonfinite_wrong_dtype_shapes_and_unbounded_chunks_reject(self):
        for change in [
            lambda d: d.update(x=vector([[0x7F80]])),
            lambda d: d.update(weight=vector([[0x7FC1]])),
            lambda d: d.update(output=vector([[0xFF80]])),
            lambda d: d.update(x=np.ones((1, 1), dtype=np.float32)),
            lambda d: d.update(output=np.zeros((1, 2), dtype=np.uint16)),
            lambda d: d.update(x=np.ones((2, 1), dtype=np.uint16)),
            lambda d: d.update(chunk_columns=True),
            lambda d: d.update(chunk_columns=65),
        ]:
            data = {
                "x": vector([[0x3F80]]),
                "weight": vector([[0x3F80]]),
                "output": vector([[0x3F80]]),
            }
            change(data)
            with self.assertRaises(ValueError):
                bound.certify(**data)


class CaptureTests(unittest.TestCase):
    @staticmethod
    def capture(root):
        shape = {
            "input_shape": [1, 256],
            "weight_shape": [2560, 256],
            "output_shape": [1, 2560],
        }
        metadata = {
            "index": 0,
            "role": "ple",
            "dtype": "BF16",
            "acknowledged_native_return": True,
            **shape,
        }
        for name, shape_key in (
            ("input", "input_shape"),
            ("weight", "weight_shape"),
            ("output", "output_shape"),
        ):
            filename = "weight-ple.bf16" if name == "weight" else f"00-ple-{name}.bf16"
            np.zeros(metadata[shape_key], dtype="<u2").tofile(root / filename)
        path = root / "00-ple.json"
        path.write_text(json.dumps(metadata))
        return path, metadata

    def test_cli_accepts_only_the_component_and_never_overwrites_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.capture(root)
            output = root / "score"
            command = [
                sys.executable,
                str(Path(bound.__file__).resolve()),
                "--capture",
                str(path),
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=20
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report_path = output / "report.json"
            original = report_path.read_bytes()
            report = json.loads(original)
            self.assertTrue(report["complete"])
            self.assertTrue(report["component_bound_acceptance"])
            self.assertFalse(report["full_model_acceptance"])
            self.assertFalse(report["device_execution_attested"])
            self.assertEqual(report["counts"]["accepted"], 2560)
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=20
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(report_path.read_bytes(), original)

    def test_corrupt_or_incomplete_metadata_and_wrong_byte_counts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, metadata = self.capture(root)
            for index, change in enumerate(
                [
                    lambda m: m.update(dtype="F32"),
                    lambda m: m.update(index=True),
                    lambda m: m.update(index=1),
                    lambda m: m.update(role="../weight"),
                    lambda m: m.update(input_shape=[True, 256]),
                    lambda m: m.update(weight_shape=[2560, 128]),
                    lambda m: m.update(acknowledged_native_return=False),
                ]
            ):
                mutated = copy.deepcopy(metadata)
                change(mutated)
                path.write_text(json.dumps(mutated))
                with self.assertRaises(ValueError):
                    bound.verify(path, root / f"invalid-{index}")
                report = json.loads((root / f"invalid-{index}/report.json").read_text())
                self.assertFalse(report["complete"])
                self.assertFalse(report["component_bound_acceptance"])
            path.write_text(json.dumps(metadata))
            (root / "00-ple-input.bf16").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "byte count"):
                bound.read_capture(path)
            path.write_text('{"index":0,"index":0}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                bound.read_capture(path)


if __name__ == "__main__":
    unittest.main()
