#!/usr/bin/env python3
"""Run pinned AIPerf and audit its native exports; never generate load ourselves."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from aiperf_results import validate
from endpoint_config import AIPERF_VERSION, command, parse_manifest, require
from endpoint_trace import parse_trace


def run(manifest_path, output, executable):
    snapshot = manifest_path.read_bytes()
    manifest = parse_manifest(snapshot)
    trace = None
    if manifest["workload"]["kind"] == "fixed_schedule":
        trace = (manifest_path.parent / manifest["workload"]["trace_file"]).read_bytes()
        parse_trace(trace, manifest["workload"])
    executable = shutil.which(executable)
    require(executable is not None, "AIPerf executable not found")
    version = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    require(
        version == AIPERF_VERSION, f"expected AIPerf {AIPERF_VERSION}, got {version}"
    )
    output.mkdir(parents=True)  # Never replace a previous run or its evidence.
    (output / "manifest.toml").write_bytes(snapshot)
    if trace is not None:
        (output / manifest["workload"]["trace_file"]).write_bytes(trace)
    args = command(manifest, executable, output.resolve())
    provenance = {
        "command": args,
        "python": sys.version,
        "platform": platform.platform(),
        "manifest_sha256": hashlib.sha256(snapshot).hexdigest(),
    }
    if trace is not None:
        provenance["trace_sha256"] = hashlib.sha256(trace).hexdigest()
    (output / "invocation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (output / "client.log").open("w") as log:
        result = subprocess.run(args, check=False, stdout=log, stderr=subprocess.STDOUT)
    (output / "client-exit-code").write_text(f"{result.returncode}\n")
    require(
        result.returncode == 0,
        f"AIPerf exited {result.returncode}; see {output / 'client.log'}",
    )
    return validate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    execute = commands.add_parser("run", help="profile an already running endpoint")
    execute.add_argument("manifest", type=Path)
    execute.add_argument(
        "--output", type=Path, required=True, help="new artifact directory"
    )
    execute.add_argument("--aiperf", default="aiperf")
    audit = commands.add_parser(
        "validate", help="audit the copied manifest and native exports"
    )
    audit.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.output if args.action == "run" else args.directory
    try:
        report = (
            run(args.manifest, directory, args.aiperf)
            if args.action == "run"
            else validate(directory)
        )
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        report = {"protocol_valid": False, "error": str(error)}
        # Failed run creation never grants ownership of an existing directory.
        # Only explicit revalidation may replace an earlier validation report.
        if args.action == "validate" and directory.is_dir():
            (directory / "validation.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        print(f"benchmark validation failed: {error}", file=sys.stderr)
        return 1
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["protocol_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
