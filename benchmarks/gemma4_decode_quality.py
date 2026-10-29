#!/usr/bin/env python3
"""Score the frozen Gemma CUDA reordered-decode QA screen, not device acceptance.

Stdlib only. Model output is string data, never executable code. Inputs must already
exist locally. See gemma4-cuda-baseline.md for the corpus and result-record contract.
"""

import argparse
import hashlib
import json
import math
import re
import string
from collections import Counter
from fractions import Fraction
from pathlib import Path

from gemma4_reference import require

DATASET_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
PROMPTS_SHA256 = "d7048e2e7172913b0246031f59544b0920e5213eecabd73ba7f33a9a2f186ad5"
SEED = "ribn-gemma4-cuda-reordered-decode-qa-v1"
OLD_SEED = "ribn-gemma4-generated-qa-v1"
MAX_FILE_BYTES = 8 << 20
MAX_NEW_TOKENS = 64
VOCAB_SIZE = 262144
DELETE_PUNCTUATION = str.maketrans("", "", string.punctuation)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def finite_float(value):
    parsed = float(value)
    require(math.isfinite(parsed), "nonfinite JSON number")
    return parsed


def read_json(path):
    with path.open("rb") as stream:
        payload = stream.read(MAX_FILE_BYTES + 1)
    require(len(payload) <= MAX_FILE_BYTES, f"input exceeds 8 MiB: {path}")
    return payload, json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def selected_ids(identifiers):
    """Selection uses IDs only, never labels, lengths or model results."""
    require(len(set(identifiers)) == len(identifiers), "duplicate source question ID")

    def rank(seed, identifier):
        return digest((seed + "\0" + identifier).encode()), identifier

    old = set(sorted(identifiers, key=lambda q: rank(OLD_SEED, q))[:32])
    remaining = sorted(set(identifiers) - old, key=lambda q: rank(SEED, q))
    require(len(remaining) >= 128, "insufficient source questions")
    return remaining[:128]


def validate_corpus(dataset, cases):
    require(dataset["version"] == "1.1", "wrong dataset version")
    source = {}
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                identifier = question["id"]
                require(identifier not in source, "duplicate source question ID")
                source[identifier] = {
                    "context": paragraph["context"],
                    "question": question["question"],
                    "answers": question["answers"],
                }
    require(
        isinstance(cases, list)
        and [c["id"] for c in cases] == selected_ids(list(source)),
        "wrong selected question IDs or order",
    )
    for case in cases:
        require(
            all(case[key] == value for key, value in source[case["id"]].items()),
            "question content or human aliases changed",
        )
        require(
            case["answers"]
            and all(
                case["context"][a["answer_start"] : a["answer_start"] + len(a["text"])]
                == a["text"]
                for a in case["answers"]
            ),
            "invalid human answer span",
        )
        require(
            isinstance(case["prompt_ids"], list)
            and 0 < len(case["prompt_ids"]) <= 4096 - MAX_NEW_TOKENS
            and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in case["prompt_ids"]),
            "invalid frozen prompt IDs",
        )
    pairs = [[c["id"], c["prompt_ids"]] for c in cases]
    encoded = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()
    require(digest(encoded) == PROMPTS_SHA256, "wrong frozen prompt digest")


def normalize(text):
    without_articles = re.sub(
        r"\b(?:a|an|the)\b", " ", text.lower().translate(DELETE_PUNCTUATION)
    )
    return " ".join(without_articles.split())


def answer_score(text, aliases):
    """Pinned SQuADv1.1 semantics, with exact rational token-overlap F1."""
    prediction = normalize(text)
    tokens = prediction.split()
    counts = Counter(tokens)
    em, f1 = False, Fraction(0)
    for alias in aliases:
        gold = normalize(alias["text"])
        target = gold.split()
        overlap = sum((counts & Counter(target)).values())
        score = (
            Fraction(2 * overlap, len(tokens) + len(target)) if overlap else Fraction(0)
        )
        em = em or prediction == gold
        f1 = max(f1, score)
    return em, f1


def result_rows(result, cases, role):
    require(
        type(result.get("schema_version")) is int
        and result["schema_version"] == 1
        and result.get("complete") is True
        and "error" not in result,
        f"incomplete {role} run",
    )
    rows = result["results"]
    require(
        isinstance(rows, list) and len(rows) == len(cases), f"missing {role} results"
    )
    by_id = {}
    for row in rows:
        require(
            isinstance(row["id"], str) and row["id"] not in by_id, "duplicate result ID"
        )
        by_id[row["id"]] = row
    require(set(by_id) == {c["id"] for c in cases}, f"wrong {role} question IDs")
    for case in cases:
        row = by_id[case["id"]]
        require("error" not in row and isinstance(row["text"], str), "failed result")
        require(
            row["prompt_ids"] == case["prompt_ids"]
            and all(type(t) is int for t in row["prompt_ids"]),
            "result prompt IDs differ",
        )
        tokens, widths, usage = row["token_ids"], row["prediction_widths"], row["usage"]
        require(
            isinstance(tokens, list)
            and 0 < len(tokens) <= MAX_NEW_TOKENS
            and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in tokens),
            "invalid generated token IDs",
        )
        require(
            row["finish_reason"] in ("stop", "length")
            and (row["finish_reason"] != "length" or len(tokens) == MAX_NEW_TOKENS),
            "invalid terminal outcome",
        )
        require(
            all(
                type(usage[k]) is int
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            and usage["prompt_tokens"] == len(case["prompt_ids"])
            and usage["completion_tokens"] == len(tokens)
            and usage["total_tokens"] == len(case["prompt_ids"]) + len(tokens),
            "usage does not match retained token IDs",
        )
        require(
            isinstance(widths, list)
            and len(widths) == len(tokens)
            and all(type(w) is int for w in widths)
            and widths[0] == 0
            and all(w in (1, 4) for w in widths[1:]),
            "invalid prediction widths",
        )
        if role == "baseline":
            require(all(w == 1 for w in widths[1:]), "control is not serial decode")
    if role == "candidate":
        for offset in range(0, len(cases), 4):
            group = [by_id[c["id"]] for c in cases[offset : offset + 4]]
            for row in group:
                widths = row["prediction_widths"]
                require(4 in widths, "question has no four-row decode exposure")
                for step, width in enumerate(widths[1:], 1):
                    active = sum(len(peer["token_ids"]) > step for peer in group)
                    require(
                        width == (4 if active == 4 else 1),
                        "wrong fixed-group decode width",
                    )
    return by_id


def assess(cases, baseline, candidate):
    control = result_rows(baseline, cases, "baseline")
    tested = result_rows(candidate, cases, "candidate")
    rows = []
    for case in cases:
        identifier = case["id"]
        control_score = answer_score(control[identifier]["text"], case["answers"])
        candidate_score = answer_score(tested[identifier]["text"], case["answers"])
        rows.append(
            {
                "id": identifier,
                "baseline": {"em": control_score[0], "f1": str(control_score[1])},
                "candidate": {"em": candidate_score[0], "f1": str(candidate_score[1])},
                "loss": candidate_score[0] < control_score[0]
                or candidate_score[1] < control_score[1],
                "text_equal": control[identifier]["text"] == tested[identifier]["text"],
            }
        )
    return {
        "complete": True,
        "quality_sample_pass": not any(row["loss"] for row in rows),
        "questions": len(rows),
        "losses": [row["id"] for row in rows if row["loss"]],
        "rows": rows,
        "device_execution_attested": False,
        "numerical_or_lifecycle_acceptance": False,
        "population_quality_equivalence": False,
    }


def compare(dataset, cases, baseline, candidate, output):
    output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "complete": False,
        "quality_sample_pass": False,
        "inputs": {},
    }
    try:
        source_before = digest(Path(__file__).read_bytes())
        inputs = {}
        for name, path in (
            ("dataset", dataset),
            ("cases", cases),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            payload, inputs[name] = read_json(path)
            report["inputs"][name] = {
                "path": str(path.resolve()),
                "sha256": digest(payload),
            }
        require(
            report["inputs"]["dataset"]["sha256"] == DATASET_SHA256,
            "wrong dataset SHA256",
        )
        validate_corpus(inputs["dataset"], inputs["cases"])
        verdict = assess(inputs["cases"], inputs["baseline"], inputs["candidate"])
        require(
            digest(Path(__file__).read_bytes()) == source_before,
            "scorer source changed",
        )
        report["scorer_sha256"] = source_before
        report.update(verdict)
    except Exception as error:
        report.update(complete=False, quality_sample_pass=False)
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "cases", "baseline", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.dataset, args.cases, args.baseline, args.candidate, args.output
    )
    raise SystemExit(0 if result["quality_sample_pass"] else 1)


if __name__ == "__main__":
    main()
