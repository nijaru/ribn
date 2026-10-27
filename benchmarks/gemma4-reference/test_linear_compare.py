"""Explicit CPU tensor diagnostics; synthetic files are never device evidence."""

import copy
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemma4_linear_compare as comparator
import gemma4_linear_reference as linear
import gemma4_reference as reference
import torch
from gemma4_linear_compare import compare, projection_errors
from gemma4_reference import sha256
from safetensors.torch import save_file


def operands():
    x = torch.tensor([[2, -3], [4, 5]], dtype=torch.bfloat16)
    w = torch.tensor([[1, 2], [-2, 0], [0, 4]], dtype=torch.bfloat16)
    values, absolute = linear.fsum_linear(
        x.float().tolist(), (row.float().tolist() for row in w)
    )
    expected = {
        "embedding_rows": x.clone(),
        "input": x,
        "weight": w,
        "reference": torch.tensor(values, dtype=torch.float64),
        "sum_abs_products": torch.tensor(absolute, dtype=torch.float64),
    }
    captured = {
        "input": x.clone(),
        "weight": w.clone(),
        "output": expected["reference"].float(),
    }
    return expected, captured


def file_fixture(d):
    expected, captured = operands()
    rp = d / "reference.json"
    cp = d / "capture.safetensors"
    tp = d / "projection.safetensors"
    save_file(expected, tp)
    save_file(captured, cp)
    manifest = tomllib.loads(reference.MANIFEST.read_text())
    frozen = {
        "schema_version": 1,
        "complete": True,
        "checkpoint": manifest["checkpoint"],
        "package_before": manifest["files"],
        "package_after": manifest["files"],
        "source_parameters": {
            "embedding_rows": linear.EMBEDDING,
            "weight": linear.PROJECTION,
        },
        "generator_sha256": {
            p.name: sha256(p)
            for p in [
                Path(linear.__file__),
                Path(reference.__file__),
                reference.MANIFEST,
            ]
        },
        "packages": {
            k: manifest["reference"][k]
            for k in ["torch", "transformers", "safetensors", "tokenizers"]
        },
        "artifact": {
            "file": tp.name,
            "size_bytes": tp.stat().st_size,
            "sha256": sha256(tp),
        },
        "token_ids": [0, 1],
        "tensors": {
            n: {"shape": list(t.shape), "dtype": str(t.dtype)}
            for n, t in expected.items()
        },
    }
    rp.write_text(json.dumps(frozen))
    return rp, cp, tp, frozen


class ComparisonTests(unittest.TestCase):
    def test_non_square_orientation_and_full_output_errors(self):
        expected, captured = operands()
        captured["output"][0, 0] += 1
        captured["output"][1, 2] -= 2
        errors = projection_errors(expected, captured, torch)
        self.assertEqual(errors["max_abs"], 2)
        self.assertAlmostEqual(errors["rmse"], (5 / 6) ** 0.5)
        self.assertAlmostEqual(errors["relative_l2"], (5 / 836) ** 0.5)

    def test_binary64_reference_is_not_rounded_to_captured_dtype(self):
        x = torch.tensor([[2**24, 1]], dtype=torch.bfloat16)
        w = torch.ones((1, 2), dtype=torch.bfloat16)
        target = torch.tensor([[2**24 + 1]], dtype=torch.float64)
        expected = {
            "embedding_rows": x,
            "input": x,
            "weight": w,
            "reference": target,
            "sum_abs_products": target,
        }
        captured = {"input": x, "weight": w, "output": target.float()}
        result = projection_errors(expected, captured, torch)
        self.assertEqual(result["max_abs"], 1)
        self.assertEqual(result["relative_l2"], 1 / (2**24 + 1))

    def test_zero_reference_norm_stays_undefined(self):
        expected, captured = operands()
        expected["reference"].zero_()
        captured["output"].fill_(1)
        result = projection_errors(expected, captured, torch)
        self.assertIsNone(result["relative_l2"])
        self.assertEqual(result["max_abs"], 1)
        self.assertEqual(result["rmse"], 1)

    def test_rejects_operand_bits_shapes_dtypes_and_nonfinite_outputs(self):
        for name in (
            "input-bit",
            "weight-zero-sign",
            "transpose",
            "dtype",
            "nan",
            "ref-inf",
            "negative-scale",
        ):
            with self.subTest(name=name):
                expected, captured = operands()
                if name == "input-bit":
                    captured["input"].view(torch.int16)[0, 0] ^= 1
                elif name == "weight-zero-sign":
                    captured["weight"][1, 1] = -0.0
                elif name == "transpose":
                    captured["output"] = captured["output"].T
                elif name == "dtype":
                    captured["input"] = captured["input"].float()
                elif name == "nan":
                    captured["output"][0, 0] = float("nan")
                elif name == "ref-inf":
                    expected["reference"][0, 0] = float("inf")
                else:
                    expected["sum_abs_products"][0, 0] = -1
                with self.assertRaises(ValueError):
                    projection_errors(expected, captured, torch)

    def test_file_identity_failure_reports_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rp, cp, _, frozen = file_fixture(d)
            output = d / "comparison"
            result = compare(rp, cp, output)
            self.assertTrue(result["complete"])
            self.assertTrue(result["stored_operand_bits_equal"])
            self.assertFalse(result["actual_device_operand_capture_verified"])
            self.assertFalse(result["native_execution_verified"])
            self.assertEqual(result["numerical_acceptance"], "none")
            original = (output / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                compare(rp, cp, output)
            self.assertEqual((output / "report.json").read_bytes(), original)
            for name, change in [
                ("hash", lambda f: f["artifact"].update(sha256="0" * 64)),
                (
                    "escape",
                    lambda f: f["artifact"].update(file="../projection.safetensors"),
                ),
                ("incomplete", lambda f: f.update(complete=False)),
                (
                    "source",
                    lambda f: f["generator_sha256"].update({"model.toml": "0" * 64}),
                ),
            ]:
                with self.subTest(name=name):
                    bad = copy.deepcopy(frozen)
                    change(bad)
                    rp.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        compare(rp, cp, d / name)
                    report = json.loads((d / name / "report.json").read_text())
                    self.assertFalse(report["complete"])
                    self.assertIn("error", report)

    def test_replacements_cannot_rebase_validated_identity(self):
        for target in ("report", "tensors", "source"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                rp, cp, tp, _ = file_fixture(d)
                local_source = d / "comparator.py"
                local_source.write_text("original source\n")
                mutated = False

                def digest(
                    path, *, tp=tp, rp=rp, target=target, local_source=local_source
                ):
                    nonlocal mutated
                    value = sha256(path)
                    if Path(path) == tp and not mutated:
                        mutated = True
                        if target == "report":
                            report = json.loads(rp.read_text())
                            report["token_ids"] = [2, 3]
                            rp.write_text(json.dumps(report))
                        elif target == "tensors":
                            expected, _ = operands()
                            expected["reference"][0, 0] += 1
                            save_file(expected, tp)
                        else:
                            local_source.write_text("changed source\n")
                    return value

                # Only the identity boundary is redirected; real files/decoding
                # perform the replacement. Project source stays untouched.
                with (
                    patch.object(comparator, "__file__", str(local_source)),
                    patch.object(comparator, "sha256", digest),
                    self.assertRaisesRegex(ValueError, "changed"),
                ):
                    compare(rp, cp, d / "output")
                self.assertTrue(mutated)
                self.assertFalse(
                    json.loads((d / "output/report.json").read_text())["complete"]
                )

    def test_cli_bad_reference_retains_failure_before_importing_torch(self):
        script = Path(__file__).resolve().parents[1] / "gemma4_linear_compare.py"
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(script),
                    "--reference-report",
                    str(d / "missing"),
                    "--capture",
                    str(d / "missing-capture"),
                    "--output",
                    str(d / "output"),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "FileNotFoundError",
                json.loads((d / "output/report.json").read_text())["error"],
            )


if __name__ == "__main__":
    unittest.main()
