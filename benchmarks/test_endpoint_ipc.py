"""Client IPC location is explicit and recorded without altering the workload."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from endpoint import run
from test_endpoint import MANIFEST


class IpcPathTests(unittest.TestCase):
    def exercise(self, configured):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.toml"
            manifest.write_text(MANIFEST)
            output = root / "run"
            ipc = root / "socket directory" if configured else None
            with (
                patch("endpoint.shutil.which", return_value="/client/aiperf"),
                patch("endpoint.platform.platform", return_value="fixture host"),
                patch(
                    "endpoint.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, stdout="0.13.0\n"),
                        subprocess.CompletedProcess([], 0),
                    ],
                ) as execute,
                patch("endpoint.validate", return_value={}),
            ):
                run(manifest, output, "aiperf", ipc_path=ipc)
            actual = execute.call_args_list[1].args[0]
            recorded = json.loads((output / "invocation.json").read_text())["command"]
            self.assertEqual(actual, recorded)
            self.assertEqual((output / "manifest.toml").read_text(), MANIFEST)
            if configured:
                self.assertEqual(actual[-2:], ["--zmq-ipc-path", str(ipc.resolve())])
                self.assertEqual(actual.count("--zmq-ipc-path"), 1)
            else:
                self.assertNotIn("--zmq-ipc-path", actual)

    def test_explicit_path_is_one_argument_and_preserved_in_provenance(self):
        self.exercise(True)

    def test_unset_path_keeps_native_default(self):
        self.exercise(False)


if __name__ == "__main__":
    unittest.main()
