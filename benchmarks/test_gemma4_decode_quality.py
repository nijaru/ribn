"""The fixed QA scorer's data/score contracts; no synthetic record attests CUDA."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import gemma4_decode_quality as quality


def fixtures(lengths=(3, 3, 3, 3)):
    cases, control, candidate = [], [], []
    for i, length in enumerate(lengths):
        case = {"id": str(i), "prompt_ids": [2, 3 + i], "answers": [{"text": "cat"}]}
        row = {
            "id": case["id"],
            "prompt_ids": case["prompt_ids"],
            "text": "cat",
            "token_ids": [5] * (length - 1) + [1],
            "finish_reason": "stop",
            "prediction_widths": [0] + [1] * (length - 1),
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": length,
                "total_tokens": 2 + length,
            },
        }
        cases.append(case)
        control.append(row)
        tested = copy.deepcopy(row)
        tested["prediction_widths"] = [0] + [
            4 if sum(n > step for n in lengths) == 4 else 1 for step in range(1, length)
        ]
        candidate.append(tested)
    header = {"schema_version": 1, "complete": True}
    return cases, {**header, "results": control}, {**header, "results": candidate}


class QualityTests(unittest.TestCase):
    def test_selection_is_id_only_and_excludes_the_original_sample(self):
        identifiers = [f"q{i}" for i in range(180)]
        old = sorted(
            identifiers,
            key=lambda q: (quality.digest((quality.OLD_SEED + "\0" + q).encode()), q),
        )[:32]
        result = quality.selected_ids(identifiers)
        self.assertEqual(len(result), 128)
        self.assertFalse(set(result).intersection(old))
        self.assertEqual(result, quality.selected_ids(list(reversed(identifiers))))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            quality.selected_ids(identifiers + [identifiers[0]])
        with self.assertRaisesRegex(ValueError, "insufficient"):
            quality.selected_ids(identifiers[:100])

    def test_whole_answer_aliases_empty_text_and_repeated_tokens(self):
        for text, aliases, expected in [
            ("THE, Cat!", ["cat"], (True, Fraction(1))),
            ("cat cat dog", ["cat dog dog"], (False, Fraction(2, 3))),
            ("answer\nExplanation", ["answer"], (False, Fraction(2, 3))),
            ("wrong", ["right", "wrong"], (True, Fraction(1))),
            ("a—cat", ["cat"], (False, Fraction(0))),
            ("", [""], (True, Fraction(0))),
            ("", ["answer"], (False, Fraction(0))),
        ]:
            with self.subTest(text=text):
                self.assertEqual(
                    quality.answer_score(text, [{"text": a} for a in aliases]), expected
                )

    def test_normalized_answer_changes_are_allowed_but_do_not_attest_execution(self):
        cases, control, candidate = fixtures()
        candidate["results"][0]["text"] = "The cat!"
        result = quality.assess(cases, control, candidate)
        self.assertTrue(result["quality_sample_pass"])
        self.assertFalse(result["rows"][0]["text_equal"])
        self.assertFalse(result["device_execution_attested"])
        self.assertFalse(result["numerical_or_lifecycle_acceptance"])
        self.assertFalse(result["population_quality_equivalence"])

    def test_equal_aggregate_scores_cannot_offset_an_individual_loss(self):
        cases, control, candidate = fixtures()
        control["results"][1]["text"] = "dog"
        candidate["results"][0]["text"] = "dog"
        result = quality.assess(cases, control, candidate)
        self.assertTrue(result["complete"])
        self.assertFalse(result["quality_sample_pass"])
        self.assertEqual(result["losses"], ["0"])
        self.assertEqual(
            sum(r["baseline"]["em"] for r in result["rows"]),
            sum(r["candidate"]["em"] for r in result["rows"]),
        )

    def test_f1_loss_is_rejected_even_when_both_exact_matches_fail(self):
        cases, control, candidate = fixtures()
        control["results"][0]["text"] = "cat dog"
        candidate["results"][0]["text"] = "cat dog bird"
        result = quality.assess(cases, control, candidate)
        self.assertEqual(result["losses"], ["0"])
        self.assertEqual(result["rows"][0]["baseline"], {"em": False, "f1": "2/3"})
        self.assertEqual(result["rows"][0]["candidate"], {"em": False, "f1": "1/2"})

    def test_incomplete_failed_missing_duplicate_and_foreign_results_reject(self):
        for change in [
            lambda r: r.update(complete=False),
            lambda r: r.update(schema_version=True),
            lambda r: r.update(error="timeout"),
            lambda r: r["results"].pop(),
            lambda r: r["results"].__setitem__(1, r["results"][0]),
            lambda r: r["results"][0].update(id="foreign"),
            lambda r: r["results"][0].update(error="execution failed"),
        ]:
            cases, control, candidate = fixtures()
            change(candidate)
            with self.assertRaises(ValueError):
                quality.assess(cases, control, candidate)

    def test_usage_tokens_terminals_and_widths_are_validated_not_trusted(self):
        for change in [
            lambda r: r.update(prompt_ids=[True, 3]),
            lambda r: r.update(token_ids=[]),
            lambda r: r.update(token_ids=[True, 5, 1]),
            lambda r: r.update(token_ids=[5, quality.VOCAB_SIZE, 1]),
            lambda r: r.update(finish_reason="length"),
            lambda r: r.update(finish_reason="error"),
            lambda r: r["usage"].update(completion_tokens=True),
            lambda r: r["usage"].update(total_tokens=9),
            lambda r: r.update(prediction_widths=[0, True, 4]),
            lambda r: r.update(prediction_widths=[4, 4, 4]),
            lambda r: r.update(prediction_widths=[0, 1, 1]),
            lambda r: r.update(text=None),
        ]:
            cases, control, candidate = fixtures()
            change(candidate["results"][0])
            with self.assertRaises(ValueError):
                quality.assess(cases, control, candidate)
        cases, control, candidate = fixtures()
        control["results"][0]["prediction_widths"][1] = 4
        with self.assertRaisesRegex(ValueError, "control is not serial"):
            quality.assess(cases, control, candidate)

    def test_eos_shrink_uses_scalar_fallback_without_refilling_and_order_is_by_id(self):
        cases, control, candidate = fixtures((3, 4, 2, 3))
        candidate["results"].reverse()
        self.assertTrue(
            quality.assess(cases, control, candidate)["quality_sample_pass"]
        )
        # At prediction index two one peer has finished, so only three are active.
        candidate["results"][0]["prediction_widths"][2] = 4
        with self.assertRaisesRegex(ValueError, "fixed-group"):
            quality.assess(cases, control, candidate)
        cases, control, candidate = fixtures((1, 3, 3, 3))
        with self.assertRaisesRegex(ValueError, "no four-row"):
            quality.assess(cases, control, candidate)

    def test_length_cap_is_scored_whole_not_dropped(self):
        cases, control, candidate = fixtures((64, 64, 64, 64))
        for record in (*control["results"], *candidate["results"]):
            record["finish_reason"] = "length"
        candidate["results"][0]["text"] = "cat plus explanation"
        self.assertEqual(quality.assess(cases, control, candidate)["losses"], ["0"])

    def test_corpus_pins_selected_ids_source_aliases_and_encoded_prompts(self):
        questions = [
            {
                "id": f"q{i}",
                "question": "Which animal?",
                "answers": [{"text": "cat", "answer_start": 0}],
            }
            for i in range(180)
        ]
        dataset = {
            "version": "1.1",
            "data": [{"paragraphs": [{"context": "cat dog", "qas": questions}]}],
        }
        source = {q["id"]: q for q in questions}
        cases = [
            {**source[qid], "context": "cat dog", "prompt_ids": [2, i + 3]}
            for i, qid in enumerate(quality.selected_ids(list(source)))
        ]
        pairs = [[c["id"], c["prompt_ids"]] for c in cases]
        # Only the external corpus identity is substituted; real selection/content validation runs.
        digest = quality.digest(
            json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()
        )
        with patch.object(quality, "PROMPTS_SHA256", digest):
            quality.validate_corpus(dataset, cases)
            for change in [
                lambda c: c.reverse(),
                lambda c: c[0].update(question="changed"),
                lambda c: c[0].update(answers=[{"text": "dog", "answer_start": 4}]),
                lambda c: c[0].update(prompt_ids=[2, 4]),
                lambda c: c[0].update(prompt_ids=[True, 3]),
            ]:
                mutated = copy.deepcopy(cases)
                change(mutated)
                with self.assertRaises(ValueError):
                    quality.validate_corpus(dataset, mutated)

    def test_json_rejects_duplicate_keys_nonfinite_numbers_and_oversize_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            for payload in (
                '{"complete":false,"complete":true}',
                "[NaN]",
                "[1e999]",
                "[Infinity]",
            ):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    quality.read_json(path)
            path.write_bytes(b" " * (quality.MAX_FILE_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                quality.read_json(path)

    def test_cli_failure_retains_error_and_cannot_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            command = [sys.executable, "-S", str(Path(quality.__file__).resolve())]
            for name in ("dataset", "cases", "baseline", "candidate"):
                command.extend([f"--{name}", str(root / "missing")])
            command.extend(["--output", str(output)])
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["complete"])
            self.assertIn("FileNotFoundError", report["error"])
            original = (output / "report.json").read_bytes()
            result = subprocess.run(
                command, capture_output=True, check=False, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output / "report.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
