"""Stock reduced-width vision/text integration: taps, insertion and retirement.

Run with the vision reference's pinned local environment. No checkpoint is loaded.
These tests protect the capture contract, not the synthetic model's quality.
"""

import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemma4_reference as reference
import gemma4_vision_reference as vision


class VisionCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = tomllib.loads(reference.MANIFEST.read_text())
        cls.torch = vision.runtime({}, manifest, 2)
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration

        config = json.loads((reference.MANIFEST.parent / "config.json").read_text())
        config["audio_config"] = None
        config.update(image_token_id=30, boi_token_id=28, eoi_token_id=29)
        config["text_config"].update(
            vocab_size=32,
            vocab_size_per_layer_input=32,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            global_head_dim=8,
            hidden_size_per_layer_input=2,
            sliding_window=4,
            max_position_embeddings=2048,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=[31],
        )
        config["vision_config"].update(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            patch_size=16,
            pooling_kernel_size=3,
        )
        cfg = Gemma4Config.from_dict(config)
        cfg._attn_implementation = "eager"
        cls.model = Gemma4ForConditionalGeneration(cfg).to(cls.torch.bfloat16).eval()
        # Never stop the small fixture early: compare a real continuation as well.
        cls.model.generation_config.eos_token_id = [100]

    def inputs(self):
        torch = self.torch
        positions = torch.full((2, 18, 2), -1, dtype=torch.long)
        positions[:, :9] = torch.tensor([[x, y] for y in range(3) for x in range(3)])
        pixels = torch.zeros((2, 18, 768), dtype=torch.float32)
        pixels[0, :9] = 1
        pixels[1, :9] = 0.25
        return {
            "input_ids": torch.tensor([[1, 28, 30, 29, 3, 28, 30, 29, 4]]),
            "pixel_values": pixels,
            "image_position_ids": positions,
            "attention_mask": torch.ones((1, 9), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0, 0, 1, 0, 0]]),
        }

    def test_actual_stock_forwards_are_unchanged_and_features_are_inserted(self):
        torch = self.torch
        inputs = self.inputs()

        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return str(ids)

        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            archive = vision.Archive(Path(tmp))
            captured = vision.capture_case(
                self.model,
                Tokenizer(),
                inputs,
                {"soft_token_counts": [1, 1]},
                archive,
                3,
            )
            result = self.model(**inputs, use_cache=True, logits_to_keep=1)
            history = inputs["input_ids"][0].tolist()
            for step, row in enumerate(captured["rows"]):
                self.assertTrue(
                    torch.equal(
                        archive.tensors[f"step.{step}.logits"], result.logits[0, -1]
                    )
                )
                self.assertEqual(row["history_ids"], history)
                if step == 2:
                    break
                history.append(row["chosen"])
                result = self.model(
                    input_ids=torch.tensor([[row["chosen"]]]),
                    attention_mask=torch.ones((1, len(history)), dtype=torch.long),
                    past_key_values=result.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            self.assertEqual(list(archive.tensors["vision.valid"].shape), [2, 16])
            self.assertEqual(
                archive.tensors["vision.pooler_mask"].tolist(),
                [[True, False], [True, False]],
            )
            self.assertTrue(
                torch.equal(
                    archive.tensors["text.inputs_embeds"][0, [2, 6]],
                    archive.tensors["vision.projected"],
                )
            )
            for name in ("vision.layer.0", "vision.layer.1", "text.per_layer_inputs"):
                self.assertIn(name, archive.tensors)
            meta = archive.flush("case")
            from safetensors.torch import load_file

            saved = load_file(Path(tmp) / meta["file"])
            self.assertEqual(saved["vision.valid"].dtype, torch.bfloat16)
            self.assertEqual(saved["vision.pooled_padded"].dtype, torch.float32)

    def test_collection_serializes_real_stock_loader_metadata(self):
        # Isolate the full-checkpoint pin only; use the real saved stock model,
        # loader, forwards, archive writer and final report. HF returns sets in
        # output_loading_info, even when every diagnostic is empty.
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, prepared = root / "snapshot", root / "prepared"
            prepared.mkdir()
            self.model.save_pretrained(snapshot)
            from tokenizers import Tokenizer, models
            from transformers import PreTrainedTokenizerFast

            tokenizer = Tokenizer(
                models.WordLevel(
                    {"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}},
                    unk_token="[UNK]",
                )
            )
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]"
            ).save_pretrained(snapshot)
            inputs = self.inputs()
            save_file(
                {f"small.{key}": value for key, value in inputs.items()},
                prepared / "inputs.safetensors",
            )
            metadata = {
                "complete": True,
                "manifest_sha256": reference.sha256(reference.MANIFEST),
                "vision_manifest_sha256": reference.sha256(vision.VISION_MANIFEST),
                "generator_sha256": reference.sha256(Path(vision.__file__)),
                "image_token_id": 30,
                "pooling_kernel": 3,
                "tensor_file": {
                    "sha256": reference.sha256(prepared / "inputs.safetensors")
                },
                "cases": [
                    {
                        "name": "small",
                        "images": ["first", "second"],
                        "prompt_ids": inputs["input_ids"][0].tolist(),
                        **vision.validate_inputs(inputs, 30, 3),
                    }
                ],
            }
            reference.write_json(prepared / "inputs.json", metadata)
            with (
                patch.object(reference, "verify_files", return_value={}),
                patch.object(vision, "runtime", return_value=self.torch),
            ):
                result = vision.collect(snapshot, prepared, root / "output", "bf16", 2)
            report = json.loads((root / "output/reference.json").read_text())
            self.assertTrue(result["complete"] and report["complete"])
            self.assertTrue(
                all(
                    isinstance(value, list) and not value
                    for value in report["loading"].values()
                )
            )
            self.assertEqual(len(report["cases"]), 1)

    def test_processor_boundary_rejects_invalid_media_layout_and_padding(self):
        torch = self.torch
        valid = self.inputs()
        valid["pixel_values"] = torch.zeros((2, 18, 768), dtype=torch.float32)
        layout = vision.validate_inputs(valid, 30, 3)
        self.assertEqual(
            layout,
            {
                "patch_counts": [9, 9],
                "soft_token_counts": [1, 1],
                "image_spans": [[2, 3], [6, 7]],
            },
        )
        for defect, message in (
            ("span", "spans disagree"),
            ("position", "partial padding"),
            ("pixels", "nonzero padding"),
        ):
            with self.subTest(defect=defect):
                inputs = {key: value.clone() for key, value in valid.items()}
                if defect == "span":
                    inputs["input_ids"][0, 4] = 30
                elif defect == "position":
                    inputs["image_position_ids"][0, -1, 0] = 0
                else:
                    inputs["pixel_values"][0, -1, 0] = 1
                with self.assertRaisesRegex(ValueError, message):
                    vision.validate_inputs(inputs, 30, 3)

    def test_failed_observer_does_not_leave_hooks_or_mutate_model(self):
        torch = self.torch
        inputs = self.inputs()
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            expected = self.model(**inputs, logits_to_keep=1).logits.clone()
            with self.assertRaisesRegex(ValueError, "byte budget"):
                with vision.vision_taps(self.model, vision.Archive(Path(tmp), limit=1)):
                    self.model(**inputs, logits_to_keep=1)
            self.assertTrue(
                all(
                    not m._forward_hooks and not m._forward_pre_hooks
                    for m in self.model.modules()
                )
            )
            self.assertTrue(
                torch.equal(expected, self.model(**inputs, logits_to_keep=1).logits)
            )


if __name__ == "__main__":
    unittest.main()
