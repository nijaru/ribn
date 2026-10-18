#!/usr/bin/env python3
"""Compare raw top-five margins on Ribn's exact token histories, not free-running text.

Uses an already-running, caller-qualified loopback llama-server. Never starts a
model process. Stores exact requests/responses and partial failures in a new output
directory. This is a numerical diagnostic, not a timing or quality-acceptance gate.
Native API contract: llama.cpp 4e416ee73, tools/server/README.md and server-task.cpp.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from itertools import pairwise
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "crates/qwen/tests/fixtures/qwen38-code-fill4096-257.tokens"
)
ARTIFACT = re.compile(r"gguf:sha256:[0-9a-f]{64}")
CONTEXT = re.compile(r"margins-context artifact=(\S+) prompt=([0-9,]+)")
ROW = re.compile(r"margins\[(\d+)\] chosen=(\d+) gap=(\S+) (.+)")
MAX_CONTEXT = 512  # qwen_decode_bench's actual bound, not the model's context limit.
MAX_RESPONSE_BYTES = 1 << 20
POLICY = {
    "n_predict": 1,
    # Negative temperature selects greedy while n_probs uses raw-logit softmax.
    # Zero plus top_k=1 can instead report a transformed distribution.
    "temperature": -1.0,
    "samplers": ["temperature"],
    "n_probs": 5,
    "post_sampling_probs": False,
    "ignore_eos": False,
    "logit_bias": [],
    "cache_prompt": False,
    "return_tokens": True,
    "add_special": False,
    "stream": False,
    "stop": [],
    "seed": 1,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token(value):
    require(type(value) is int and 0 <= value <= 0xFFFFFFFF, "invalid token ID")
    return value


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def top_five(entries, chosen):
    require(
        isinstance(entries, list) and len(entries) == 5,
        "expected five raw top logprobs",
    )
    ids = []
    values = []
    for entry in entries:
        require(isinstance(entry, dict), "invalid top-logprob entry")
        ids.append(token(entry.get("id")))
        value = entry.get("logprob")
        require(finite(value) and value <= 0, "logprob must be finite and nonpositive")
        values.append(value)
    require(len(set(ids)) == len(ids), "duplicate top-logprob token IDs")
    require(all(a >= b for a, b in pairwise(values)), "unordered top logprobs")
    require(
        chosen in ids and values[ids.index(chosen)] == values[0],
        "chosen token is not a greedy winner",
    )
    return [{"id": key, "logprob": value} for key, value in zip(ids, values)]


def parse_inputs(fixture_bytes, ribn_bytes):
    lines = fixture_bytes.decode().splitlines()
    require(
        len(lines) == 3 and ARTIFACT.fullmatch(lines[0]),
        "expected a three-line GGUF token fixture",
    )
    prompt = [token(int(value)) for value in lines[1].split()]
    require(prompt and len(prompt) <= MAX_CONTEXT, "invalid fixture prompt length")
    require(lines[2].split(), "fixture is missing its reference continuation")
    for value in lines[2].split():
        token(int(value))  # The agreed prefix is provenance, not the replay history.
    context = None
    rows = []
    for line in ribn_bytes.decode().splitlines():
        if line.startswith("margins-context"):
            match = CONTEXT.fullmatch(line)
            require(
                match is not None and context is None and not rows,
                "missing/duplicate/late margin context",
            )
            context = (match[1], [token(int(value)) for value in match[2].split(",")])
        elif line.startswith("margins["):
            match = ROW.fullmatch(line)
            require(
                match is not None and context is not None,
                "malformed margin row or missing context",
            )
            require(
                int(match[1]) == len(rows), "margin steps must be contiguous from zero"
            )
            chosen = token(int(match[2]))
            gap = float(match[3])
            require(finite(gap) and gap >= 0, "invalid Ribn margin")
            entries = []
            for pair in match[4].split():
                key, value = pair.split(":")
                entries.append({"id": token(int(key)), "logprob": float(value)})
            rows.append(
                {"chosen": chosen, "gap": gap, "top": top_five(entries, chosen)}
            )
    require(
        context == (lines[0], prompt),
        "Ribn artifact/prompt context differs from fixture",
    )
    require(
        rows and len(prompt) + len(rows) - 1 <= MAX_CONTEXT,
        "missing margins or history exceeds diagnostic context",
    )
    return lines[0], prompt, rows


def reference_row(body, history):
    require(
        isinstance(body, dict) and "error" not in body, "reference returned an error"
    )
    require(
        body.get("truncated") is False and body.get("stop") is True,
        "reference did not finish without truncation",
    )
    require(
        body.get("stop_type") in ("limit", "eos"), "unexpected reference stop filter"
    )
    for key, expected in [
        ("tokens_evaluated", len(history)),
        ("tokens_predicted", 1),
        ("tokens_cached", 0),
    ]:
        require(
            type(body.get(key)) is int and body[key] == expected,
            f"reference {key} differs from request",
        )
    settings = body.get("generation_settings")
    require(isinstance(settings, dict), "missing effective reference settings")
    for key in (
        "n_predict",
        "temperature",
        "n_probs",
        "ignore_eos",
        "samplers",
        "post_sampling_probs",
    ):
        require(
            type(settings.get(key)) is type(POLICY[key])
            and settings[key] == POLICY[key],
            f"reference changed {key}",
        )
    tokens = body.get("tokens")
    require(isinstance(tokens, list) and len(tokens) == 1, "missing generated token ID")
    chosen = token(tokens[0])
    probabilities = body.get("completion_probabilities")
    require(
        isinstance(probabilities, list) and len(probabilities) == 1,
        "missing single-step raw probabilities",
    )
    entry = probabilities[0]
    require(
        isinstance(entry, dict) and token(entry.get("id")) == chosen,
        "probability token differs from generation",
    )
    top = top_five(entry.get("top_logprobs"), chosen)
    return {"chosen": chosen, "gap": top[0]["logprob"] - top[1]["logprob"], "top": top}


def comparison(ribn, reference):
    expected = {entry["id"]: entry["logprob"] for entry in reference["top"]}
    shared = [
        {
            "id": entry["id"],
            "ribn_minus_reference": entry["logprob"] - expected[entry["id"]],
        }
        for entry in ribn["top"]
        if entry["id"] in expected
    ]
    return {
        "winner_agreement": ribn["chosen"] == reference["chosen"],
        "ribn": ribn,
        "reference": reference,
        "shared_top_logprob_deltas": shared,
        "ribn_winner_reference_logprob": expected.get(ribn["chosen"]),
    }


def endpoint(base):
    url = urllib.parse.urlsplit(base)
    require(
        url.scheme == "http"
        and url.hostname in ("localhost", "127.0.0.1", "::1")
        and url.username is None
        and url.password is None
        and not url.query
        and not url.fragment
        and url.path in ("", "/"),
        "use an HTTP loopback server base URL",
    )
    return base.rstrip("/") + "/completion"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def no_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def collect(fixture, ribn_log, base_url, reference_build, output, timeout=1800):
    fixture_bytes, ribn_bytes = fixture.read_bytes(), ribn_log.read_bytes()
    artifact, prompt, rows = parse_inputs(fixture_bytes, ribn_bytes)
    url = endpoint(base_url)
    require(reference_build.strip(), "record the independently checked reference build")
    require(finite(timeout) and timeout > 0, "timeout must be finite and positive")
    output.mkdir(parents=True)  # Existing evidence is never overwritten.
    (output / "fixture.tokens").write_bytes(fixture_bytes)
    (output / "ribn-margins.log").write_bytes(ribn_bytes)
    report = {
        "schema_version": 1,
        "complete": False,
        "artifact_id": artifact,
        "reference_build_assertion": reference_build,
        "reference_identity_verified": False,
        "reference_url": url,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "ribn_log_sha256": hashlib.sha256(ribn_bytes).hexdigest(),
        "ribn_logprob_decimal_places": 4,
        "history_source": "Ribn chosen token IDs; reference fully recomputes each prefix",
        "scope": "shared top-five logprobs only; not full-distribution or task-quality acceptance",
        "expected_rows": len(rows),
        "rows": [],
    }
    history = list(prompt)
    # Local diagnostics must not send prompts through environment proxy settings
    # or follow a response's redirect to an unrelated endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        for index, ribn in enumerate(rows):
            payload = {**POLICY, "prompt": list(history)}
            encoded = json.dumps(payload, allow_nan=False).encode()
            (output / f"{index:04d}.request.json").write_bytes(encoded)
            request = urllib.request.Request(
                url, data=encoded, headers={"Content-Type": "application/json"}
            )
            response_path = output / f"{index:04d}.response.json"
            try:
                with opener.open(request, timeout=timeout) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
            except urllib.error.HTTPError as error:
                with error:
                    response_path.write_bytes(error.read(MAX_RESPONSE_BYTES + 1))
                raise ValueError(f"reference HTTP status {error.code}") from error
            response_path.write_bytes(raw)
            require(
                len(raw) <= MAX_RESPONSE_BYTES,
                "reference response exceeded diagnostic bound",
            )
            body = json.loads(raw, parse_constant=no_constant)
            result = comparison(ribn, reference_row(body, history))
            report["rows"].append(
                {"index": index, "history_tokens": len(history), **result}
            )
            # Crucial: never append the reference's prediction after a divergence.
            history.append(ribn["chosen"])
        report["winner_disagreements"] = [
            row["index"] for row in report["rows"] if not row["winner_agreement"]
        ]
        report["complete"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["error"] = str(error)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--ribn-margins", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8099")
    parser.add_argument(
        "--reference-build",
        required=True,
        help="caller assertion; preserve server command/log separately",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory; includes prompts and raw responses",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    try:
        report = collect(
            args.fixture,
            args.ribn_margins,
            args.url,
            args.reference_build,
            args.output,
            args.timeout,
        )
    except (OSError, ValueError) as error:
        print(f"margin collection failed: {error}", file=sys.stderr)
        return 1
    if not report["complete"]:
        print(
            f"margin collection incomplete: {report['error']}; see {args.output}",
            file=sys.stderr,
        )
        return 1
    print(
        f"compared {len(report['rows'])} common-history rows; winner disagreements: {report['winner_disagreements']}"
    )
    print("diagnostic only; no numerical or task-quality acceptance threshold applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
