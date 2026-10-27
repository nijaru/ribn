#!/usr/bin/env python3
"""CPU-only resized-RGB patch references; no model construction or weight loading.

Use the existing pinned vision-reference environment. Resizing is deliberately
turned off: these tensors qualify packing/rescaling/padding, not image decoding,
interpolation, media admission or the encoder's failed full-checkpoint fidelity gate.
"""

import argparse
import copy
import hashlib
import json
import tomllib
from pathlib import Path

import gemma4_reference as reference
import gemma4_vision_reference as vision_reference

ROOT = Path(__file__).resolve().parent.parent


def collect(output):
    paths = [
        Path(__file__).resolve(),
        Path(reference.__file__),
        Path(vision_reference.__file__),
        reference.MANIFEST,
        vision_reference.VISION_MANIFEST,
        reference.MANIFEST.with_name("config.json"),
        ROOT / "crates/gemma4/tests/fixtures/vision.json",
    ]
    frozen = {path: path.read_bytes() for path in paths}
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(b).hexdigest()
        for p, b in frozen.items()
    }
    manifest = tomllib.loads(frozen[reference.MANIFEST].decode())
    report = {
        "complete": False,
        "scope": "already-resized RGB8 HWC, CPU, no model",
        "sources": sources,
        "cases": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    reference.write_json(output / "report.json", report)
    torch = vision_reference.runtime(report, manifest, 2)
    from safetensors.torch import save_file
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

    real = json.loads(frozen[reference.MANIFEST.with_name("config.json")])
    small = copy.deepcopy(real)
    small["vision_config"] = json.loads(
        frozen[ROOT / "crates/gemma4/tests/fixtures/vision.json"]
    )["vision_config"]
    small["vision_config"]["patch_size"] = 2
    golden = {
        "provenance": {
            "packages": report["packages"],
            "vision_sources": report["vision_sources"],
            "rgb_pattern": "flat HWC byte index % 256",
            "do_resize": False,
            "policy": "CPU float32 rescale, no normalize; exact bits",
        },
        "vision_config": small["vision_config"],
        "cases": [],
    }
    for kind, config, dimensions, budgets in [
        ("small", small, [(16, 8), (8, 16)], [70]),
        ("e4b", real, [(96, 48), (48, 96)], [70, 140, 280, 560, 1120]),
    ]:
        reference.write_json(output / f"{kind}-config.json", config)
        patch = config["vision_config"]["patch_size"]
        pool = config["vision_config"]["pooling_kernel_size"]
        for budget in budgets:
            processor = Gemma4ImageProcessor(
                patch_size=patch, pooling_kernel_size=pool, max_soft_tokens=budget
            )
            for width, height in dimensions:
                name = f"{kind}-{budget}-{width}x{height}"
                rgb = (
                    (torch.arange(height * width * 3, dtype=torch.int64) % 256)
                    .to(torch.uint8)
                    .reshape(height, width, 3)
                )
                result = processor(
                    images=[rgb.permute(2, 0, 1)],
                    do_resize=False,
                    do_rescale=True,
                    do_normalize=False,
                    input_data_format="channels_first",
                    return_tensors="pt",
                )
                pixels, positions = result["pixel_values"], result["image_position_ids"]
                reference.require(
                    pixels.device.type == positions.device.type == "cpu", "CPU only"
                )
                reference.require(
                    pixels.dtype == torch.float32 and positions.dtype == torch.int64,
                    "unexpected stock dtypes",
                )
                reference.require(bool(pixels.isfinite().all()), "nonfinite pixels")
                target = output / f"{name}.safetensors"
                save_file(
                    {"rgb": rgb, "pixels": pixels, "positions": positions}, target
                )
                record = {
                    "name": name,
                    "config": f"{kind}-config.json",
                    "width": width,
                    "height": height,
                    "budget": budget,
                    "soft_tokens": int(result["num_soft_tokens_per_image"][0]),
                    "pixels_shape": list(pixels.shape),
                    "positions_shape": list(positions.shape),
                    "file_sha256": reference.sha256(target),
                }
                report["cases"].append(record)
                if kind == "small":
                    valid = width * height // (patch * patch)
                    golden["cases"].append(
                        {
                            **record,
                            "valid_pixel_bits": pixels[0, :valid]
                            .flatten()
                            .view(torch.int32)
                            .tolist(),
                            "valid_positions": positions[0, :valid].tolist(),
                        }
                    )
                reference.write_json(output / "report.json", report)
    reference.require(
        all(path.read_bytes() == data for path, data in frozen.items()),
        "producer/config source changed",
    )
    # Recheck the pinned installed sources too; hashes are file consistency,
    # not whole-installation authentication or execution attestation.
    import transformers

    base = Path(transformers.__file__).parent
    reference.verify_reference_sources(base, manifest["reference"]["source_sha256"])
    vision = tomllib.loads(frozen[vision_reference.VISION_MANIFEST].decode())
    reference.verify_reference_sources(base, vision["source_sha256"])
    reference.write_json(output / "small-golden.json", golden)
    report["complete"] = True
    reference.write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="new directory, never overwrite a prior attempt"
    )
    collect(parser.parse_args().output)


if __name__ == "__main__":
    main()
