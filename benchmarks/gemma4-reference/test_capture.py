# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.14.0", "transformers==5.20.0", "safetensors==0.8.0", "tokenizers==0.23.3"]
# ///
"""Exercise the collector with pinned stock Gemma, without checkpoint weights.

Run explicitly with uv; the lightweight benchmark tests do not require PyTorch.
The reduced widths/vocabulary retain all 42 layers and the real shared-KV topology.
"""

import copy
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemma4_reference as reference


class StockCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = tomllib.loads(reference.MANIFEST.read_text())
        cls.torch = reference.prepare_cpu_reference({}, manifest["reference"], 2)
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration

        config = json.loads((reference.MANIFEST.parent / "config.json").read_text())
        config["vision_config"] = None
        config["audio_config"] = None
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
        config = Gemma4Config.from_dict(config)
        config._attn_implementation = "eager"
        cls.model = Gemma4ForConditionalGeneration(config).to(cls.torch.bfloat16).eval()
        cls.model.generation_config.eos_token_id = [31]
        cls.case = {
            "name": "small",
            "prompt_ids": [1, 3, 5, 2, 4, 6, 1, 2, 3],
            "policy": "forced",
            "forced_ids": [4, 5, 6],
        }

    def capture(self, case, stages=False, chunk_size=None):
        report, tensors = {"cases": []}, {}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(reference, "cases", return_value=[copy.deepcopy(case)]),
        ):
            reference.capture_cases(
                self.model,
                None,
                report,
                tensors,
                Path(tmp),
                stages=stages,
                prefill_chunk_size=chunk_size,
            )
        return report["cases"][0], tensors

    def test_default_and_chunked_captures_match_stock_forwards(self):
        prompt = self.case["prompt_ids"]
        schedules = [
            (None, [prompt]),
            (4, [prompt[:4], prompt[4:8], prompt[8:]]),
            (5, [prompt[:5], prompt[5:]]),
        ]
        for stages in (False, True):
            for size, chunks in schedules:
                with self.subTest(stages=stages, chunk_size=size):
                    captured, tensors = self.capture(self.case, stages, size)
                    self.assertEqual(
                        len(captured["prefill_intermediates"]), len(chunks) - 1
                    )
                    self.assertEqual(
                        [row["index"] for row in captured["steps"]], list(range(4))
                    )
                    rows = captured["prefill_intermediates"] + captured["steps"]
                    prefixes = [f"small.prefill.{i}" for i in range(len(chunks) - 1)]
                    prefixes += [f"small.{i}" for i in range(4)]
                    inputs = chunks + [[token] for token in self.case["forced_ids"]]
                    cache, history = None, []
                    with self.torch.inference_mode():
                        for row, prefix, tokens in zip(
                            rows, prefixes, inputs, strict=True
                        ):
                            start = len(history)
                            history.extend(tokens)
                            positions = list(range(start, len(history)))
                            self.assertEqual(row["input_ids"], tokens)
                            self.assertEqual(row["history_ids"], history)
                            self.assertEqual(row["position_ids"], positions)
                            expected = self.model(
                                input_ids=self.torch.tensor([tokens]),
                                attention_mask=self.torch.ones(
                                    (1, len(history)), dtype=self.torch.long
                                ),
                                position_ids=self.torch.tensor([positions]),
                                past_key_values=cache,
                                use_cache=True,
                                output_hidden_states=True,
                                logits_to_keep=1,
                            )
                            cache = expected.past_key_values
                            self.assertTrue(
                                self.torch.equal(
                                    tensors[f"{prefix}.logits"], expected.logits[0, -1]
                                )
                            )
                            for layer in (0, 22, 40, 42):
                                self.assertTrue(
                                    self.torch.equal(
                                        tensors[f"{prefix}.hidden.{layer}"],
                                        expected.hidden_states[layer][0, -1],
                                    )
                                )
                            self.assertEqual(
                                row["donor_spans"]["22"],
                                [1, 1, min(start, 3) + len(tokens), 4],
                            )
                            self.assertEqual(
                                row["donor_spans"]["23"], [1, 1, len(history), 8]
                            )
                    self.assertEqual("small.0.layer.22.input" in tensors, stages)

    def test_intermediate_eos_predictions_do_not_stop_prompt_ingestion(self):
        # Every prediction is an EOS, but only the completed prompt may stop generation.
        case = {**self.case, "policy": "greedy"}
        original = self.model.generation_config.eos_token_id
        self.model.generation_config.eos_token_id = list(range(32))
        try:
            captured, _ = self.capture(case, chunk_size=4)
        finally:
            self.model.generation_config.eos_token_id = original
        self.assertEqual(len(captured["prefill_intermediates"]), 2)
        self.assertEqual(len(captured["steps"]), 1)
        self.assertEqual(captured["steps"][0]["history_ids"], case["prompt_ids"])

    def test_failed_capture_removes_hooks_before_another_forward(self):
        with (
            patch.object(reference, "retain", side_effect=ValueError("capture failed")),
            self.assertRaisesRegex(ValueError, "capture failed"),
        ):
            self.capture(self.case, stages=True)
        with self.torch.inference_mode():
            result = self.model(input_ids=self.torch.tensor([[1]]), use_cache=False)
        self.assertTrue(result.logits.isfinite().all().item())


if __name__ == "__main__":
    unittest.main()
