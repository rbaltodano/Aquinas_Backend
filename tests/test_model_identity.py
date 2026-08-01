import json
from pathlib import Path
import unittest

from model_identity import (
    MODEL_ADAPTER_PATH,
    MODEL_BASE_ID,
    MODEL_CONTEXT_WINDOW_TOKENS,
    MODEL_DISPLAY_NAME,
    MODEL_PATH,
    MODEL_RUNTIME_PATH,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ModelIdentityTests(unittest.TestCase):
    def test_checkpoint_and_adapter_identify_the_canonical_base_model(self):
        checkpoint_root = REPOSITORY_ROOT / MODEL_PATH
        checkpoint_readme = (checkpoint_root / "README.md").read_text()
        adapter_config = json.loads(
            (REPOSITORY_ROOT / MODEL_ADAPTER_PATH / "adapter_config.json").read_text()
        )
        training_config = (REPOSITORY_ROOT / "lora_config_v4.yaml").read_text()

        self.assertIn(f"base_model: {MODEL_BASE_ID}", checkpoint_readme)
        self.assertEqual(adapter_config["model"], MODEL_BASE_ID)
        self.assertIn(f'model: "{MODEL_BASE_ID}"', training_config)
        self.assertIn("E2B", MODEL_DISPLAY_NAME)
        self.assertEqual(MODEL_RUNTIME_PATH, MODEL_BASE_ID)

    def test_context_window_matches_the_live_checkpoint(self):
        checkpoint_config = json.loads(
            (REPOSITORY_ROOT / MODEL_PATH / "config.json").read_text()
        )

        self.assertEqual(
            checkpoint_config["text_config"]["max_position_embeddings"],
            MODEL_CONTEXT_WINDOW_TOKENS,
        )


if __name__ == "__main__":
    unittest.main()
