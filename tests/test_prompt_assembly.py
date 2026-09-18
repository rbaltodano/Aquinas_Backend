import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RecordingTokenizer:
    def __init__(self) -> None:
        self.messages = None
        self.add_generation_prompt = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        return "rendered-prompt"


class PromptAssemblyTests(unittest.TestCase):
    def test_identity_and_action_are_separate_system_and_user_messages(self) -> None:
        tokenizer = RecordingTokenizer()
        processor = SimpleNamespace(tokenizer=tokenizer)
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4"))
        fake_mlx = SimpleNamespace(
            apply_chat_template=lambda _processor, _config, messages, **kwargs: (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=kwargs["add_generation_prompt"],
                )
            ),
            generate=lambda *_args, **_kwargs: SimpleNamespace(text=""),
            load=lambda _path, **_kwargs: (model, processor),
            stream_generate=lambda *_args, **_kwargs: iter(()),
        )
        module_path = REPOSITORY_ROOT / "main.py"
        spec = importlib.util.spec_from_file_location(
            "_aquinas_prompt_assembly_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, {"mlx_vlm": fake_mlx}):
            spec.loader.exec_module(module)

        rendered = module._prompt_for(
            "<TASK:CONTEXTUAL_DEFINITION>Define prudence.</TASK:CONTEXTUAL_DEFINITION>",
            response_prefix="{",
        )

        self.assertEqual(rendered, "rendered-prompt{")
        self.assertEqual(
            [message["role"] for message in tokenizer.messages],
            ["system", "user"],
        )
        self.assertIn("You are Aquinas", tokenizer.messages[0]["content"])
        self.assertNotIn(
            "care and ease of a loving older brother",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "Follow reasoning wherever it leads",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "previous claims as revisable positions",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "strongest reasonable version of the user's argument",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "explicitly revise the affected conclusion",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "Do not revise merely because the user disagrees",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "State the earlier claim that failed",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "narrowest corrected claim",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "Begin with a direct acknowledgment",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "do not call a real contradiction merely apparent",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "Keep the correction concise",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "Do not introduce auxiliary theories",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "Preserve category distinctions",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "A canonical example",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "defeats the universal claim",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "voluntary in itself through present choice",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "must not conclude that the defeated",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "both your reasoning and the user's by the same intellectual standard",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "structured application tasks",
            tokenizer.messages[0]["content"],
        )
        self.assertIn(
            "Never invent a citation",
            tokenizer.messages[0]["content"],
        )
        self.assertEqual(
            tokenizer.messages[1]["content"],
            "<TASK:CONTEXTUAL_DEFINITION>Define prudence.</TASK:CONTEXTUAL_DEFINITION>",
        )
        self.assertTrue(tokenizer.add_generation_prompt)

    def test_rejects_a_hidden_legacy_tokenizer_identity(self) -> None:
        tokenizer = RecordingTokenizer()
        processor = SimpleNamespace(tokenizer=tokenizer)
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4"))
        fake_mlx = SimpleNamespace(
            apply_chat_template=lambda _processor, _config, messages, **kwargs: (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=kwargs["add_generation_prompt"],
                )
            ),
            generate=lambda *_args, **_kwargs: SimpleNamespace(text=""),
            load=lambda _path, **_kwargs: (model, processor),
            stream_generate=lambda *_args, **_kwargs: iter(()),
        )
        module_path = REPOSITORY_ROOT / "main.py"
        spec = importlib.util.spec_from_file_location(
            "_aquinas_legacy_identity_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, {"mlx_vlm": fake_mlx}):
            spec.loader.exec_module(module)

        legacy_tokenizer = SimpleNamespace(
            init_kwargs={
                "system_prompt": "You are the Thomistic Logic Engine. DO NOT PLAN."
            },
            chat_template="Assistant: Objection 1:",
        )
        with self.assertRaises(RuntimeError):
            module._assert_no_legacy_tokenizer_identity(legacy_tokenizer)


if __name__ == "__main__":
    unittest.main()
