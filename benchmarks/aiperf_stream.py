"""Protocol and monotonic content/terminal clocks from normalized AIPerf SSE exports."""

import json

from endpoint_config import count, number, require


def timestamp(value):
    require(count(value), "missing/invalid monotonic nanosecond timestamp")
    return value


def validate_clocks(raw):
    """Check exported clocks even for failed streams with no terminal protocol."""
    start = previous = timestamp(raw.get("start_perf_ns"))
    for response in raw["responses"]:
        require(isinstance(response, dict), "invalid response record")
        received = timestamp(response.get("perf_ns"))
        require(received >= previous, "response clock regressed or preceded request")
        previous = received
    return start


def completion_stream(raw, manifest):
    """Audit one-choice completion events, not SSE wire framing or HTTP EOF."""
    payload = raw["payload"]
    require(isinstance(payload, dict), "missing request payload")
    require(
        payload.get("model") == manifest["model"]["id"]
        and payload.get("stream") is True,
        "request model/stream differs from manifest",
    )
    require(
        number(payload.get("temperature"))
        and payload["temperature"] == 0
        and type(payload.get("n", 1)) is int
        and payload.get("n", 1) == 1,
        "request is not one-choice greedy generation",
    )
    cap = manifest["workload"]["output_tokens_mean"]
    require(
        type(payload.get("max_tokens")) is int and payload["max_tokens"] == cap,
        "request output cap differs from manifest",
    )
    require(
        payload.get("ignore_eos", False) is False
        and payload.get("min_tokens", 0) == 0
        and payload.get("min_new_tokens", 0) == 0,
        "request bypasses EOS",
    )
    options = payload.get("stream_options")
    require(
        isinstance(options, dict) and options.get("include_usage") is True,
        "usage was not requested",
    )
    start = validate_clocks(raw)
    first = last = done = None
    finish = usage = None
    for response in raw["responses"]:
        received = response["perf_ns"]
        packets = response["packets"]
        require(
            isinstance(packets, list) and all(isinstance(p, dict) for p in packets),
            "invalid SSE packets",
        )
        require(
            not any(
                p.get("name") == "event" and p.get("value") == "error" for p in packets
            ),
            "SSE error event",
        )
        values = [p["value"] for p in packets if p["name"] == "data"]
        if not values:
            continue
        require(
            len(values) == 1 and done is None,
            "unexpected data after [DONE] or multiple data packets",
        )
        data = values[0]
        if data == "[DONE]":
            require(
                finish is not None and usage is not None, "[DONE] before finish/usage"
            )
            done = received
            continue
        event = json.loads(data)
        require(isinstance(event, dict) and "error" not in event, "JSON error response")
        choices = event.get("choices")
        require(
            isinstance(choices, list) and len(choices) <= 1,
            "invalid completion choices",
        )
        # MLX-LM labels its terminal usage-only frame as chat.completion,
        # even for text completions. Content must still use completion objects.
        require(
            event.get("object") in ("text_completion", "completion")
            or (
                event.get("object") == "chat.completion"
                and choices == []
                and finish is not None
                and isinstance(event.get("usage"), dict)
            ),
            "unexpected completion object",
        )
        if choices:
            require(
                finish is None and usage is None, "choice after terminal finish/usage"
            )
            choice = choices[0]
            require(isinstance(choice, dict), "invalid completion choice")
            require(
                choice.get("index") == 0 and isinstance(choice.get("text"), str),
                "invalid completion choice",
            )
            if choice["text"]:
                if first is None:
                    first = received
                last = received
            finish = choice.get("finish_reason")
            require(finish in (None, "stop", "length"), "unsupported finish reason")
        if event.get("usage") is not None:
            require(
                finish is not None and usage is None,
                "usage before finish or duplicate usage",
            )
            usage = event["usage"]
            require(isinstance(usage, dict), "invalid usage object")
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                require(count(usage.get(field)), f"invalid usage.{field}")
            require(
                usage["total_tokens"]
                == usage["prompt_tokens"] + usage["completion_tokens"],
                "incoherent token totals",
            )
            require(
                usage["completion_tokens"] <= cap, "completion exceeded requested cap"
            )
            require(
                finish != "length" or usage["completion_tokens"] == cap,
                "length finish below token cap",
            )
    require(done is not None, "missing [DONE] (truncated stream)")
    require(first is not None, "successful stream has no nonempty content clock")
    return usage, {
        "first_content": first - start,
        "last_content": last - start,
        "done_receipt": done - start,
    }
