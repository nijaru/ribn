"""Host-only entry-point checks; Cargo and NVIDIA tooling are external boundaries."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class NativeSweepTests(unittest.TestCase):
    def test_uses_cargo_reported_executable_and_preserves_build_failure(self):
        # A non-default path containing spaces also catches accidental shell splitting.
        for build_exit, emit_artifact in [(0, True), (23, True), (0, False)]:
            with (
                self.subTest(build_exit=build_exit, emit_artifact=emit_artifact),
                tempfile.TemporaryDirectory(prefix="ribn sweep ") as directory,
            ):
                root = Path(directory)
                tools = root / "tools"
                tools.mkdir()
                marker = root / "executed"
                executable = root / "custom target" / "qwen_serving_bench"
                executable.parent.mkdir()
                executable.write_text('#!/bin/sh\nprintf invoked > "$MARKER"\n')
                executable.chmod(0o755)
                artifact = {
                    "reason": "compiler-artifact",
                    "target": {"name": "qwen_serving_bench", "kind": ["example"]},
                    "executable": str(executable),
                }
                cargo = tools / "cargo"
                cargo.write_text(
                    "#!/bin/sh\n"
                    'if [ "$1" = --version ]; then echo fake-cargo; exit 0; fi\n'
                    + (
                        f"printf '%s\\n' '{json.dumps(artifact)}'\n"
                        if emit_artifact
                        else ""
                    )
                    + f"exit {build_exit}\n"
                )
                cargo.chmod(0o755)
                nvidia = tools / "nvidia-smi"
                nvidia.write_text(
                    "#!/bin/sh\n"
                    'case "$1" in\n'
                    "  --query-gpu=memory.used) echo 0;;\n"
                    "esac\n"
                )
                nvidia.chmod(0o755)
                model = root / "model.gguf"
                model.touch()
                result = subprocess.run(
                    ["bash", "benchmarks/run-qwen-serving-sweep.sh"],
                    cwd=ROOT,
                    check=False,
                    env={
                        **os.environ,
                        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                        "ENGINE_QWEN_GGUF": str(model),
                        "CARGO_TARGET_DIR": str(executable.parent),
                        "OUT_DIR": str(root / "results"),
                        "CONCURRENCIES": "1",
                        "WORKLOAD": "",
                        "MARKER": str(marker),
                    },
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if build_exit:
                    self.assertEqual(result.returncode, build_exit, result.stderr)
                    self.assertFalse(marker.exists())
                elif emit_artifact:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(marker.read_text(), "invoked")
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
