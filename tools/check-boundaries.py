#!/usr/bin/env python3
"""Check production dependency direction; development-only oracles may cross it."""

import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "foundation": {
        "engine-core",
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn",
        "ribn-batch",
        "ribn-safetensors",
        "ribn-hf",
        "ribn-text",
        "ribn-cli",
    },
    "safetensors": {
        "engine-core",
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn",
        "ribn-batch",
        "ribn-hf",
        "ribn-text",
        "ribn-cli",
    },
    "hf": {
        "engine-core",
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn",
        "ribn-batch",
        "ribn-text",
        "ribn-cli",
    },
    "bert": {
        "engine-core",
        "engine-gguf",
        "engine-qwen",
        "ribn",
        "ribn-text",
        "ribn-cli",
    },
    "batch": {
        "engine-core",
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn",
        "ribn-text",
        "ribn-cli",
    },
    "runtime": {
        "engine-core",
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn-text",
        "ribn-cli",
    },
    "core": {
        "engine-gguf",
        "engine-nvidia",
        "engine-qwen",
        "ribn",
        "ribn-text",
        "ribn-cli",
    },
    "gguf": {"engine-nvidia", "engine-qwen", "ribn", "ribn-text", "ribn-cli"},
    "qwen": {"ribn-text", "ribn-cli"},
    "gemma4": {"engine-gguf", "engine-bert", "engine-qwen", "ribn-text", "ribn-cli"},
    "cli": {"engine-core", "engine-nvidia", "engine-qwen"},
    "server": {
        "engine-core",
        "engine-nvidia",
        "engine-qwen",
        "engine-gguf",
        "ribn-cli",
    },
}
for crate, forbidden in FORBIDDEN.items():
    if crate != "gemma4":
        forbidden.add("engine-gemma4")
    if crate not in {"cli", "server"}:
        forbidden.add("ribn-server")


def dependency_names(manifest: dict) -> set[str]:
    sections = [manifest, *manifest.get("target", {}).values()]
    return {
        spec.get("package", name) if isinstance(spec, dict) else name
        for section in sections
        for kind in ("dependencies", "build-dependencies")
        for name, spec in section.get(kind, {}).items()
    }


def main() -> int:
    errors = []
    for crate, forbidden in FORBIDDEN.items():
        path = ROOT / "crates" / crate / "Cargo.toml"
        manifest = tomllib.loads(path.read_text())
        invalid = dependency_names(manifest) & forbidden
        if invalid:
            errors.append(
                f"{path.relative_to(ROOT)}: reversed dependency: {sorted(invalid)}"
            )
    for obsolete in (
        "crates/core/src/nvidia.rs",
        "crates/gguf/src/qwen.rs",
        "crates/cli/src/local.rs",
        "crates/core/src/runtime.rs",
        "crates/core/src/scheduler.rs",
        "crates/core/src/serving.rs",
        "crates/core/src/serving_runtime.rs",
        "crates/core/src/model.rs",
    ):
        if (ROOT / obsolete).exists():
            errors.append(f"{obsolete}: implementation is back in the wrong owner")
    # The model/backend seam is a prepared batch, continuation state and weight
    # descriptions. A plan/description/capability layer between them was removed
    # because nothing read it; keep it from returning as bookkeeping.
    removed_layer = (
        "struct ExecutionPlan",
        "struct ExecutionStage",
        "struct ModelDescription",
        "struct ModelCapabilities",
        "trait ModelProvider",
        "struct BackendCapabilities",
        "struct BackendFeatures",
        "struct PolicyVersion",
    )
    for path in sorted((ROOT / "crates" / "core" / "src").glob("*.rs")):
        text = path.read_text()
        for symbol in removed_layer:
            if symbol in text:
                errors.append(
                    f"{path.relative_to(ROOT)}: {symbol!r} re-introduces the removed plan layer"
                )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Production dependency boundaries passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
