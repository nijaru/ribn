"""Pinned AIPerf command, manifest validation and effective-config reconciliation."""

import math
import tomllib
from urllib.parse import urlsplit

AIPERF_VERSION = "0.13.0"
SLOS = {
    "ttft_ms": "time_to_first_token",
    "itl_ms": "inter_token_latency",
    "e2e_ms": "request_latency",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def count(value):
    return type(value) is int and value >= 0


def parse_manifest(contents):
    manifest = tomllib.loads(contents.decode("utf-8"))
    require(
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
        "unsupported manifest schema",
    )
    for section, fields in {
        "run": ["name", "engine", "engine_version", "engine_commit", "quality_scope"],
        "model": [
            "id",
            "revision",
            "artifact_sha256",
            "tokenizer",
            "tokenizer_revision",
            "dtype",
            "quantization",
        ],
        "hardware": ["device", "driver", "cuda"],
        "server": ["base_url", "command"],
    }.items():
        require(isinstance(manifest.get(section), dict), f"missing [{section}]")
        for field in fields:
            value = manifest[section].get(field)
            require(
                isinstance(value, str) and value.strip() and "REPLACE_ME" not in value,
                f"set {section}.{field}",
            )
    require(
        manifest["run"].get("benchmark_client_version") == AIPERF_VERSION,
        f"benchmark_client_version must be {AIPERF_VERSION}",
    )
    require(count(manifest["run"].get("seed")), "seed must be a nonnegative integer")
    require(
        type(manifest["run"].get("gpu_telemetry")) is bool,
        "gpu_telemetry must be boolean",
    )
    digest = manifest["model"]["artifact_sha256"]
    require(
        len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
        "artifact_sha256 must be a SHA-256 digest",
    )
    require(
        count(manifest["hardware"].get("vram_bytes")), "vram_bytes must be nonnegative"
    )
    server = manifest["server"]
    url = urlsplit(server["base_url"])
    require(
        url.scheme in ("http", "https")
        and url.hostname
        and not url.username
        and not url.password
        and url.path in ("", "/")
        and not url.query
        and not url.fragment,
        "base_url must be an HTTP(S) origin without credentials",
    )
    require(
        server.get("endpoint_type") == "completions", "only completions is supported"
    )
    require(server.get("streaming") is True, "streaming must be enabled")
    require(
        server.get("endpoint") == "/v1/completions", "endpoint must be /v1/completions"
    )
    sampling = manifest.get("sampling", {})
    require(
        isinstance(sampling, dict)
        and sampling.keys() == {"temperature", "ignore_eos"}
        and number(sampling.get("temperature"))
        and sampling["temperature"] == 0
        and sampling.get("ignore_eos") is False,
        "sampling must be greedy with EOS enabled",
    )
    workload = manifest.get("workload", {})
    require(isinstance(workload, dict), "missing [workload]")
    require(
        workload.get("kind") == "synthetic", "only synthetic workloads are supported"
    )
    for field in (
        "input_tokens_mean",
        "output_tokens_mean",
        "request_count",
        "max_concurrency",
    ):
        require(
            count(workload.get(field)) and workload[field] > 0,
            f"{field} must be positive",
        )
    require(
        count(workload.get("input_tokens_stddev")),
        "input_tokens_stddev must be a nonnegative integer",
    )
    require(
        count(workload.get("output_tokens_stddev"))
        and workload["output_tokens_stddev"] == 0,
        "output_tokens_stddev must be zero for fixed output caps",
    )
    require(
        count(workload.get("warmup_requests")), "warmup_requests must be nonnegative"
    )
    require(
        number(workload.get("request_rate")),
        "request_rate must be nonnegative (0 = unlimited)",
    )
    require(
        number(workload.get("request_timeout_seconds"), 1),
        "request_timeout_seconds must be >= 1",
    )
    require(
        workload.get("arrival_pattern") in ("constant", "poisson"),
        "unsupported arrival_pattern",
    )
    slos = manifest.get("slo", {})
    require(isinstance(slos, dict) and slos.keys() <= SLOS.keys(), "unknown SLO")
    require(
        all(number(value) and value > 0 for value in slos.values()),
        "SLOs must be positive; omit unset SLOs",
    )
    return manifest


def check_config(actual, manifest):
    """Bind the sparse v0.13 export's effective workload to the requested run."""
    require(isinstance(actual, dict), "missing effective AIPerf configuration")
    workload = manifest["workload"]
    phase = {"type": "concurrency", "concurrency": workload["max_concurrency"]}
    if workload["request_rate"]:
        phase.update(type=workload["arrival_pattern"], rate=workload["request_rate"])
    phases = []
    if workload["warmup_requests"]:
        phases.append(
            {
                **phase,
                "name": "warmup",
                "kind": "warmup",
                "exclude_from_results": True,
                "requests": workload["warmup_requests"],
            }
        )
    phases.append(
        {
            **phase,
            "name": "profiling",
            "kind": "profiling",
            "requests": workload["request_count"],
        }
    )
    expected = {
        "models": {"items": [{"name": manifest["model"]["id"]}]},
        "datasets": [
            {
                "name": "main",
                "type": "synthetic",
                "entries": workload["request_count"],
                "random_seed": manifest["run"]["seed"],
                "prompts": {
                    "isl": {
                        "mean": workload["input_tokens_mean"],
                        "stddev": workload["input_tokens_stddev"],
                    },
                    "osl": {"mean": workload["output_tokens_mean"], "stddev": 0},
                },
            }
        ],
        "phases": phases,
        "tokenizer": {
            "name": manifest["model"]["tokenizer"],
            "revision": manifest["model"]["tokenizer_revision"],
        },
        "slos": {SLOS[name]: value for name, value in manifest.get("slo", {}).items()},
        "endpoint": {
            "urls": [manifest["server"]["base_url"]],
            "type": "completions",
            "path": "/v1/completions",
            "timeout": workload["request_timeout_seconds"],
            "streaming": True,
            "connection_reuse": "never",
            "use_server_token_count": True,
            "extra": {"temperature": 0},
            "wait_for_model_mode": "inference",
            "wait_for_model_timeout": 0,
        },
    }
    for section, value in expected.items():
        require(
            actual.get(section, {}) == value,
            f"effective {section} differs from manifest",
        )
    artifacts = actual.get("artifacts", {})
    require(
        isinstance(artifacts, dict)
        and artifacts.get("raw") is True
        and artifacts.get("trace") is True
        and artifacts.get("records") == ["jsonl"],
        "required exports were not enabled",
    )
    telemetry = actual.get("gpu_telemetry", {})
    require(
        isinstance(telemetry, dict)
        and telemetry.get("enabled", True) == manifest["run"]["gpu_telemetry"],
        "effective GPU telemetry differs from manifest",
    )


def command(manifest, executable, output):
    workload = manifest["workload"]
    args = [
        executable,
        "profile",
        "--model",
        manifest["model"]["id"],
        "--tokenizer",
        manifest["model"]["tokenizer"],
        "--tokenizer-revision",
        manifest["model"]["tokenizer_revision"],
        "--endpoint-type",
        "completions",
        "--endpoint",
        "/v1/completions",
        "--streaming",
        "--url",
        manifest["server"]["base_url"],
        "--isl",
        str(workload["input_tokens_mean"]),
        "--isl-stddev",
        str(workload["input_tokens_stddev"]),
        "--osl",
        str(workload["output_tokens_mean"]),
        "--osl-stddev",
        "0",
        "--concurrency",
        str(workload["max_concurrency"]),
        "--request-count",
        str(workload["request_count"]),
        "--num-prompts",
        str(workload["request_count"]),
        "--warmup-request-count",
        str(workload["warmup_requests"]),
        "--random-seed",
        str(manifest["run"]["seed"]),
        "--use-server-token-count",
        "--extra-inputs",
        '{"temperature":0}',
        "--connection-reuse-strategy",
        "never",
        "--wait-for-model-mode",
        "inference",
        "--wait-for-model-timeout",
        "0",
        "--request-timeout-seconds",
        str(workload["request_timeout_seconds"]),
        "--export-level",
        "raw",
        "--export-http-trace",
        "--no-server-metrics",
        "--ui",
        "none",
        "--no-auto-plot",
        "--output-artifact-dir",
        str(output / "client"),
    ]
    if workload["request_rate"]:
        args += [
            "--request-rate",
            str(workload["request_rate"]),
            "--arrival-pattern",
            workload["arrival_pattern"],
        ]
    if manifest["run"].get("gpu_telemetry", False) is False:
        args.append("--no-gpu-telemetry")
    if manifest.get("slo"):
        args += [
            "--goodput",
            " ".join(
                f"{SLOS[name]}:{value}" for name, value in manifest["slo"].items()
            ),
        ]
    return args
