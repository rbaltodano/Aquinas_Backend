"""HTTP-layer tests for server.py -- routes, auth, CORS, rate limiting.

server.py eagerly imports main.py, which eagerly loads the real MLX model at
import time. To keep these tests fast and hermetic (no real weights, no
network, no touching the real production SQLite file), mlx_vlm is faked
before server.py is imported, mirroring the pattern already used in
test_prompt_assembly.py for main.py itself. The Insight Tree store is then
swapped for a throwaway temp-file-backed instance.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from insight_tree import TreeInsight
from structured_generation import (
    GeneratedQuoteNotability,
    GeneratedTreeInsightCandidate,
    GeneratedTreeUpdate,
    StructuredGenerationError,
)
from tree_store import DynamicDefinitionRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_server_module():
    fake_mlx = SimpleNamespace(
        apply_chat_template=lambda _processor, _config, messages, **kwargs: "rendered-prompt",
        generate=lambda *_args, **_kwargs: SimpleNamespace(text=""),
        load=lambda _path, **_kwargs: (
            SimpleNamespace(config=SimpleNamespace(model_type="gemma4")),
            SimpleNamespace(tokenizer=SimpleNamespace(
                apply_chat_template=lambda *_a, **_k: "rendered-prompt",
            )),
        ),
        stream_generate=lambda *_args, **_kwargs: iter(()),
    )
    # Deliberately not using unittest.mock.patch.dict here: its teardown
    # (which clears/restores the whole sys.modules bookkeeping) segfaults in
    # this environment when it runs after `main` has pulled in
    # sentence_transformers/sentencepiece's native extension. A plain
    # try/finally restore avoids that path entirely.
    had_mlx_vlm = "mlx_vlm" in sys.modules
    original_mlx_vlm = sys.modules.get("mlx_vlm")
    sys.modules["mlx_vlm"] = fake_mlx
    sys.modules.pop("main", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "_aquinas_server_test",
            REPOSITORY_ROOT / "server.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if had_mlx_vlm:
            sys.modules["mlx_vlm"] = original_mlx_vlm
        else:
            sys.modules.pop("mlx_vlm", None)
    return module


class ServerHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _load_server_module()
        cls._tmpdir = tempfile.TemporaryDirectory()
        temp_db_path = Path(cls._tmpdir.name) / "test_insight_tree.sqlite3"

        tree_store_module = sys.modules["tree_store"]
        fresh_store = tree_store_module.InsightTreeStore(temp_db_path)
        cls.server.tree_store = fresh_store
        cls.server.persistent_tree_service.store = fresh_store

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        import os

        os.environ.pop(self.server.API_KEY_ENV_VAR, None)
        self.server._rate_limit_hits.clear()
        self.addCleanup(self._restore_generation_service_methods)

    def _restore_generation_service_methods(self) -> None:
        # Tests that monkeypatch generation_service methods set them as
        # instance attributes; deleting those reverts to the shared
        # instance's normal bound methods for every other test in this class.
        for attribute in ("analyze_tree_update", "assess_quote_notability"):
            self.server.generation_service.__dict__.pop(attribute, None)

    def test_health_endpoint_reports_structured_status(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["status"], {"ready", "degraded"})
        for key in (
            "generation_model",
            "relatedness_provider",
            "grounding_index",
            "insight_tree_store",
        ):
            self.assertIn(key, body["checks"])

    def test_requests_unauthenticated_by_default(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_missing_api_key_rejected_once_configured(self) -> None:
        import os

        os.environ[self.server.API_KEY_ENV_VAR] = "correct-secret"
        try:
            with TestClient(self.server.app) as client:
                response = client.get("/health")
        finally:
            os.environ.pop(self.server.API_KEY_ENV_VAR, None)

        self.assertEqual(response.status_code, 401)

    def test_wrong_api_key_rejected(self) -> None:
        import os

        os.environ[self.server.API_KEY_ENV_VAR] = "correct-secret"
        try:
            with TestClient(self.server.app) as client:
                response = client.get(
                    "/health",
                    headers={"X-Aquinas-Api-Key": "wrong-secret"},
                )
        finally:
            os.environ.pop(self.server.API_KEY_ENV_VAR, None)

        self.assertEqual(response.status_code, 401)

    def test_correct_api_key_accepted(self) -> None:
        import os

        os.environ[self.server.API_KEY_ENV_VAR] = "correct-secret"
        try:
            with TestClient(self.server.app) as client:
                response = client.get(
                    "/health",
                    headers={"X-Aquinas-Api-Key": "correct-secret"},
                )
        finally:
            os.environ.pop(self.server.API_KEY_ENV_VAR, None)

        self.assertEqual(response.status_code, 200)

    def test_cors_allows_any_origin_without_credentials(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.get("/health", headers={"Origin": "https://example.com"})

        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")
        self.assertNotIn("access-control-allow-credentials", response.headers)

    def test_ask_endpoint_rate_limited_after_threshold(self) -> None:
        with TestClient(self.server.app) as client:
            statuses = [
                client.post("/ask", json={"query": "What is prudence?"}).status_code
                for _ in range(self.server._RATE_LIMIT_MAX_REQUESTS + 5)
            ]

        self.assertEqual(
            statuses.count(200),
            self.server._RATE_LIMIT_MAX_REQUESTS,
        )
        self.assertTrue(all(status == 429 for status in statuses[self.server._RATE_LIMIT_MAX_REQUESTS:]))

    def test_malformed_request_returns_422(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.post("/ask", json={"not_query": "missing required field"})

        self.assertEqual(response.status_code, 422)

    def test_loose_thread_returns_null_for_a_conversation_with_no_nodes(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/loose-thread",
                json={"conversation_id": "loose-thread-empty"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json())

    def test_loose_thread_returns_a_disconnected_node(self) -> None:
        self.server.persistent_tree_service.assign_and_save(
            conversation_id="loose-thread-happy",
            insight=TreeInsight(
                id="only-insight",
                title="Solitary Node",
                definition="A Node with nothing else in the tree to connect to.",
            ),
        )

        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/loose-thread",
                json={"conversation_id": "loose-thread-happy"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["insight_count"], 1)

    def test_glossed_terms_returns_null_when_nothing_qualifies(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/glossed-terms",
                json={"conversation_id": "glossed-empty"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json())

    def test_glossed_terms_returns_a_stale_unpromoted_term(self) -> None:
        self.server.persistent_tree_service.save_dynamic_definition(
            conversation_id="glossed-happy",
            term_key="prudence",
            source_hash="source-a",
            requested_term="Prudence",
            source_excerpt="Prudence is right reason applied to action.",
            definition=DynamicDefinitionRecord(
                title="Prudence",
                part_of_speech="noun",
                pronunciation="PROO-dns",
                definition="Right reason applied to action.",
                example="Prudence governs the choice of means.",
            ),
        )
        # Backdate the lookup so it clears the default 24h staleness window
        # without waiting for real time to pass.
        import sqlite3

        with sqlite3.connect(self.server.tree_store.database_path) as connection:
            connection.execute(
                "UPDATE dynamic_definitions SET created_at = datetime('now', '-2 days') "
                "WHERE conversation_id = 'glossed-happy'"
            )

        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/glossed-terms",
                json={"conversation_id": "glossed-happy"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["term_key"], "prudence")

    def test_saving_an_insight_promotes_its_matching_glossed_term_idempotently(self) -> None:
        conversation_id = "glossed-promotion"
        self.server.persistent_tree_service.save_dynamic_definition(
            conversation_id=conversation_id,
            term_key="temperance",
            source_hash="source-a",
            requested_term="Temperance",
            source_excerpt="Temperance moderates desire.",
            definition=DynamicDefinitionRecord(
                title="Temperance",
                part_of_speech="noun",
                pronunciation="TEM-per-ns",
                definition="Moderation of desire for sensory pleasure.",
                example="Temperance restrains excess.",
            ),
        )

        payload = {
            "insight": {
                "id": "temperance-insight",
                "title": "Temperance",
                "definition": "Moderation of desire for sensory pleasure.",
            }
        }
        with TestClient(self.server.app) as client:
            first = client.post(
                f"/insight-tree/{conversation_id}/insights",
                json=payload,
            )
            second = client.post(
                f"/insight-tree/{conversation_id}/insights",
                json=payload,
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)

        import sqlite3

        with sqlite3.connect(self.server.tree_store.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT promoted_to_insight FROM dynamic_definitions "
                "WHERE conversation_id = ? AND term_key = 'temperance'",
                (conversation_id,),
            ).fetchone()
        self.assertEqual(row["promoted_to_insight"], 1)

    def test_today_in_history_returns_null_for_an_unseeded_date(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/today-in-history",
                json={"conversation_id": "history-empty", "override_date": "02-30"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json())

    def test_today_in_history_returns_a_seeded_entry_without_a_link(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/today-in-history",
                json={"conversation_id": "history-happy", "override_date": "03-07"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["related_entity"], "Thomas Aquinas")
        self.assertIsNone(body["linked_node_id"])

    def test_your_quote_returns_null_when_nothing_is_flagged(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/your-quote",
                json={"conversation_id": "quote-empty"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json())

    def test_quotes_cannot_be_flagged_manually(self) -> None:
        with TestClient(self.server.app) as client:
            response = client.post(
                "/home/flag-quote",
                json={
                    "conversation_id": "quote-manual-disabled",
                    "response_id": "response-1",
                    "quote_text": "A user-flagged quote worth remembering.",
                },
            )

        self.assertEqual(response.status_code, 404)

    def test_analyze_response_flags_a_heuristically_notable_quote(self) -> None:
        conversation_id = "analyze-flags-quote"
        response_id = "response-1"
        long_declarative_question = (
            "If prudence governs the choice of means, then a habit acquired through "
            "repeated deliberate acts seems to retain its voluntary character even "
            "once it becomes second nature."
        )
        self.server.generation_service.analyze_tree_update = lambda **_kwargs: GeneratedTreeUpdate(
            subject_label="Prudence and habit",
            subject_summary="A summary about prudence and habitual action.",
            insight_candidate=None,
        )
        self.server.generation_service.assess_quote_notability = (
            lambda _quote_text: GeneratedQuoteNotability(
                is_notable_insight=True,
                reason="An original synthesis about habituation.",
            )
        )

        with TestClient(self.server.app) as client:
            response = client.post(
                f"/insight-tree/{conversation_id}/responses/{response_id}/analyze",
                json={
                    "branch_id": "branch-1",
                    "question": long_declarative_question,
                    "response": "A response about prudence and habitual action.",
                },
            )
            quote_response = client.post(
                "/home/your-quote",
                json={"conversation_id": conversation_id},
            )

        self.assertEqual(response.status_code, 200)
        quote_body = quote_response.json()
        self.assertIsNotNone(quote_body)
        self.assertEqual(quote_body["response_id"], response_id)
        self.assertEqual(quote_body["source"], "heuristic_llm")

    def test_analyze_response_does_not_flag_a_short_question(self) -> None:
        conversation_id = "analyze-skips-question"
        self.server.generation_service.analyze_tree_update = lambda **_kwargs: GeneratedTreeUpdate(
            subject_label="Topic",
            subject_summary="A summary.",
            insight_candidate=None,
        )
        self.server.generation_service.assess_quote_notability = lambda _quote_text: (
            _ for _ in ()
        ).throw(AssertionError("Tier 2 must not run when Tier 1 rejects the message."))

        with TestClient(self.server.app) as client:
            response = client.post(
                f"/insight-tree/{conversation_id}/responses/response-1/analyze",
                json={
                    "branch_id": "branch-1",
                    "question": "What comes next?",
                    "response": "A short response.",
                },
            )
            quote_response = client.post(
                "/home/your-quote",
                json={"conversation_id": conversation_id},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(quote_response.json())

    def test_quote_notability_failure_does_not_fail_tree_analysis(self) -> None:
        conversation_id = "analyze-quote-failure-isolated"
        long_declarative_question = (
            "Prudence seems to preserve freedom by making good action more ready, "
            "rather than by forcing a person to act without deliberation."
        )
        self.server.generation_service.analyze_tree_update = lambda **_kwargs: GeneratedTreeUpdate(
            subject_label="Prudence and freedom",
            subject_summary="A summary about prudence and free action.",
            insight_candidate=None,
        )
        self.server.generation_service.assess_quote_notability = lambda _quote_text: (
            _ for _ in ()
        ).throw(RuntimeError("quote classifier unavailable"))

        with TestClient(self.server.app) as client:
            response = client.post(
                f"/insight-tree/{conversation_id}/responses/response-1/analyze",
                json={
                    "branch_id": "branch-1",
                    "question": long_declarative_question,
                    "response": "A response about prudence and free action.",
                },
            )
            quote_response = client.post(
                "/home/your-quote",
                json={"conversation_id": conversation_id},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(quote_response.json())

    def test_analyze_response_survives_a_failed_quote_notability_check(self) -> None:
        conversation_id = "analyze-survives-quote-failure"
        long_declarative_question = (
            "If prudence governs the choice of means, then a habit acquired through "
            "repeated deliberate acts seems to retain its voluntary character even "
            "once it becomes second nature."
        )
        self.server.generation_service.analyze_tree_update = lambda **_kwargs: GeneratedTreeUpdate(
            subject_label="Prudence and habit",
            subject_summary="A summary about prudence and habitual action.",
            insight_candidate=None,
        )

        def _raise_quote_error(_quote_text):
            raise StructuredGenerationError("The model did not return a valid JSON object.")

        self.server.generation_service.assess_quote_notability = _raise_quote_error

        with TestClient(self.server.app) as client:
            response = client.post(
                f"/insight-tree/{conversation_id}/responses/response-1/analyze",
                json={
                    "branch_id": "branch-1",
                    "question": long_declarative_question,
                    "response": "A response about prudence and habitual action.",
                },
            )
            quote_response = client.post(
                "/home/your-quote",
                json={"conversation_id": conversation_id},
            )

        # The Tier-2 failure must not surface as an error on /analyze -- the
        # tree extraction result is the load-bearing part of that call.
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(quote_response.json())


if __name__ == "__main__":
    unittest.main()
