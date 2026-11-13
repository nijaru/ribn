#!/usr/bin/env python3
"""Prepare and score frozen Gemma QA through audited AIPerf endpoint exports.

No load generator, model loading or execution of generated text. This finite-corpus
screen does not attest input token IDs, device retirement or broad quality equivalence.
"""

import argparse
import hashlib
import json
import sys
import tomllib
from fractions import Fraction
from pathlib import Path

from aiperf_results import records, validate
from aiperf_stream import completion_stream
from endpoint_config import parse_manifest, require
from gemma4_decode_quality import (
    DATASET_SHA256,
    MAX_NEW_TOKENS,
    answer_score,
    digest,
    read_json,
    validate_corpus,
)
from gemma4_reference import MANIFEST

# Ordered [question_id, rendered_prompt] pairs, UTF-8 compact JSON without newline.
# These publisher-template strings were frozen with the existing prompt-ID corpus.
RENDERED_PROMPTS_SHA256 = (
    "cad0f61be3442dfa13e28da464254680aba1166c56ac7b83db706e3f430439fa"
)


def corpus(dataset_path, cases_path):
    payload, dataset = read_json(dataset_path)
    require(digest(payload) == DATASET_SHA256, "wrong source dataset hash")
    _, cases = read_json(cases_path)
    validate_corpus(dataset, cases)
    require(
        all(isinstance(c.get("prompt"), str) and c["prompt"] for c in cases),
        "missing rendered prompt",
    )
    rendered = json.dumps(
        [[c["id"], c["prompt"]] for c in cases],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    require(digest(rendered) == RENDERED_PROMPTS_SHA256, "rendered prompts changed")
    return cases


def trace_bytes(cases):
    return "".join(
        json.dumps(
            {
                "session_id": c["id"],
                "timestamp": 0,
                "text_input": c["prompt"],
                "output_length": MAX_NEW_TOKENS,
            },
            ensure_ascii=False,
        )
        + "\n"
        for c in cases
    ).encode("utf-8")


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checkpoint_policy(manifest):
    package = tomllib.loads(MANIFEST.read_text())
    model = manifest["model"]
    require(
        model["revision"] == package["checkpoint"]["revision"]
        and model["artifact_sha256"] == package["files"]["model.safetensors"]["sha256"]
        and model["tokenizer_revision"] == package["checkpoint"]["revision"]
        and model["dtype"] == "bf16"
        and model["quantization"] == "none",
        "manifest does not declare the pinned BF16 checkpoint/tokenizer policy",
    )


def observed_answer(raw):
    """Retain partial/invalid stream evidence, never certify it for scoring."""
    pieces, finishes, usages, errors, malformed = [], [], [], [], []
    responses = raw.get("responses")
    if not isinstance(responses, list):
        malformed.append({"responses": responses})
        responses = []
    for response in responses:
        if not isinstance(response, dict):
            malformed.append(response)
            continue
        packets = response.get("packets")
        if not isinstance(packets, list):
            malformed.append({"packets": packets})
            continue
        for packet in packets:
            if not isinstance(packet, dict):
                malformed.append(packet)
                continue
            if packet.get("name") == "event" and packet.get("value") == "error":
                errors.append(packet)
            if packet.get("name") != "data" or packet.get("value") == "[DONE]":
                continue
            try:
                event = json.loads(packet["value"])
            except (ValueError, KeyError, TypeError):
                malformed.append(packet)
                continue
            if not isinstance(event, dict):
                malformed.append(packet)
                continue
            if "error" in event:
                errors.append(event["error"])
            if event.get("usage") is not None:
                usages.append(event["usage"])
            choices = event.get("choices", [])
            if not isinstance(choices, list):
                malformed.append(packet)
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    malformed.append(packet)
                    continue
                if isinstance(choice.get("text"), str):
                    pieces.append(choice["text"])
                else:
                    malformed.append(packet)
                if choice.get("finish_reason") is not None:
                    finishes.append(choice["finish_reason"])
    return {
        "text": "".join(pieces),
        "finish_reasons": finishes,
        "usage_frames": usages,
        "errors": errors,
        "malformed_data": malformed,
    }


def analyze_run(directory, cases):
    manifest = parse_manifest((directory / "manifest.toml").read_bytes())
    checkpoint_policy(manifest)
    workload = manifest["workload"]
    require(
        workload["kind"] == "fixed_schedule"
        and workload["request_count"] == len(cases)
        and workload["output_tokens_mean"] == MAX_NEW_TOKENS
        and workload["warmup_requests"] == 0,
        "QA requires the complete fixed-schedule cohort, cap 64 and no warmup",
    )
    trace = directory / workload["trace_file"]
    require(
        trace.read_bytes() == trace_bytes(cases), "run trace differs from frozen QA"
    )
    audit = validate(directory)  # Reaudit exports; never trust validation.json.
    raw = records(directory / "client" / "profile_export_raw.jsonl")
    ordinary = records(directory / "client" / "profile_export.jsonl")
    by_id = {r["metadata"]["conversation_id"]: key for key, r in raw.items()}
    require(len(by_id) == len(cases), "incomplete or duplicate QA coverage")
    rows = []
    for case in cases:
        key = by_id[case["id"]]
        request, result = raw[key], ordinary[key]
        row = {
            "id": case["id"],
            "request_id": key,
            "http_status": request.get("status"),
            "error": result.get("error"),
            "raw_error": request.get("error"),
            "observed": observed_answer(request),
        }
        try:
            require(row["error"] is None and row["raw_error"] is None, "failed request")
            require(request.get("status") == 200, "request did not return HTTP 200")
            require(not request["metadata"].get("was_cancelled"), "cancelled request")
            usage, _ = completion_stream(request, manifest)
            text = row["observed"]["text"]
            row.update(
                text=text,
                finish_reason=row["observed"]["finish_reasons"][-1],
                usage=usage,
            )
            require(
                usage["prompt_tokens"] == len(case["prompt_ids"]),
                "prompt token count differs from frozen IDs",
            )
            em, f1 = answer_score(text, case["answers"])
            row.update(em=em, f1=str(f1))
        except (ValueError, KeyError, TypeError) as error:
            row["validation_error"] = str(error)
        rows.append(row)
    complete = (
        audit["protocol_valid"]
        and audit["failed_requests"] == 0
        and all("f1" in row for row in rows)
    )
    paths = [
        "manifest.toml",
        "invocation.json",
        "client-exit-code",
        workload["trace_file"],
        "client/profile_export.jsonl",
        "client/profile_export_raw.jsonl",
        "client/profile_export_aiperf.json",
    ]
    return {
        "directory": str(directory.resolve()),
        "complete": complete,
        "engine": manifest["run"],
        "max_concurrency": workload["max_concurrency"],
        "source_sha256": {p: file_hash(directory / p) for p in paths},
        "endpoint_audit": audit,
        "rows": rows,
        # Never turn missing/failed answers into zero scores or survivor means.
        "scores": {
            "questions": len(rows),
            "exact_matches": sum(row["em"] for row in rows),
            "mean_f1": str(
                sum((Fraction(row["f1"]) for row in rows), Fraction()) / len(rows)
            ),
        }
        if complete
        else None,
    }


def assess(baseline, candidate):
    require(
        [r["id"] for r in baseline["rows"]] == [r["id"] for r in candidate["rows"]],
        "question coverage/order differs",
    )
    complete = baseline["complete"] and candidate["complete"]
    losses, gains = [], []
    if complete:
        for before, after in zip(baseline["rows"], candidate["rows"], strict=True):
            old = (before["em"], Fraction(before["f1"]))
            new = (after["em"], Fraction(after["f1"]))
            if new[0] < old[0] or new[1] < old[1]:
                losses.append(before["id"])
            if new[0] > old[0] or new[1] > old[1]:
                gains.append(before["id"])
    return {
        "schema_version": 1,
        "complete": complete,
        "quality_sample_pass": complete and not losses,
        "criterion": "candidate has no per-question EM or token-F1 loss against baseline",
        "losses": losses if complete else None,
        "gains": gains if complete else None,
        "baseline": baseline,
        "candidate": candidate,
        "input_token_ids_attested": False,
        "device_execution_attested": False,
        "numerical_or_lifecycle_acceptance": False,
        "population_quality_equivalence": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compare"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "compare" and (args.baseline is None or args.candidate is None):
        parser.error("compare requires --baseline and --candidate endpoint directories")
    try:
        args.output.mkdir(parents=True)  # Never replace earlier evidence.
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 1
    report = {"schema_version": 1, "complete": False}
    try:
        cases = corpus(args.dataset, args.cases)
        if args.action == "prepare":
            trace = trace_bytes(cases)
            (args.output / "schedule.jsonl").write_bytes(trace)
            report.update(
                complete=True, questions=len(cases), trace_sha256=digest(trace)
            )
        else:
            report = assess(
                analyze_run(args.baseline, cases), analyze_run(args.candidate, cases)
            )
        report["tool_sha256"] = {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "gemma4_endpoint_quality.py",
                "gemma4_decode_quality.py",
                "aiperf_results.py",
                "aiperf_stream.py",
                "aiperf_clock.py",
                "endpoint_config.py",
                "endpoint_trace.py",
                "gemma4_reference.py",
            )
        }
        report["corpus_sha256"] = {
            "dataset": file_hash(args.dataset),
            "cases": file_hash(args.cases),
            "rendered_prompts": RENDERED_PROMPTS_SHA256,
        }
    except (ValueError, OSError, KeyError, TypeError) as error:
        report.update(complete=False, quality_sample_pass=False, error=str(error))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("baseline", "candidate")},
            indent=2,
        )
    )
    return 0 if report["complete"] and report.get("quality_sample_pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
