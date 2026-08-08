import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from insight_tree import InsightTreeEngine, TreeInsight
from relatedness import MiniLMRelatednessProvider
from structured_generation import (
    GeneratedTreeInsightCandidate,
    GeneratedTreeUpdate,
)
from tree_store import DynamicDefinitionRecord, InsightTreeStore, PersistentInsightTreeService


class InjectedScoreProvider:
    dimensions = 2

    vectors = {
        "First Principle": np.asarray([1.0, 0.0], dtype=np.float32),
        "Near Duplicate": np.asarray([0.9, 0.4358899], dtype=np.float32),
        "Distinct Subject": np.asarray([0.0, 1.0], dtype=np.float32),
        "Third Subject": np.asarray([-1.0, 0.0], dtype=np.float32),
    }

    def embed(self, text):
        for title, vector in self.vectors.items():
            if text.startswith(title):
                return vector
        raise AssertionError(f"Missing injected vector for {text}")

    def compare_embeddings(self, left, right):
        similarity = float(np.dot(left, right))
        return {
            "similarity": similarity,
            "relatedness": max(0.0, similarity),
            "distance": 1.0 - similarity,
        }

    def centroid(self, embeddings):
        center = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
        return center / np.linalg.norm(center)


class PersistentInsightTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = MiniLMRelatednessProvider()
        cls.engine = InsightTreeEngine(cls.provider)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "insight-tree.sqlite3"
        )
        self.store = InsightTreeStore(self.database_path)
        self.store.initialize()
        self.service = PersistentInsightTreeService(
            store=self.store,
            engine=self.engine,
            provider=self.provider,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_first_insight_survives_store_recreation(self) -> None:
        decision = self.service.assign_and_save(
            conversation_id="conversation-1",
            insight=TreeInsight(
                id="insight-1",
                title="Natural Law",
                definition="Reason's participation in eternal law.",
            ),
        )

        reopened_store = InsightTreeStore(self.database_path)
        snapshot = reopened_store.snapshot("conversation-1")

        self.assertEqual(decision.action, "created")
        self.assertEqual(len(snapshot["nodes"]), 1)
        self.assertEqual(snapshot["nodes"][0]["insights"][0]["id"], "insight-1")
        self.assertNotIn("embedding", snapshot["nodes"][0]["insights"][0])

        connection = sqlite3.connect(self.database_path)
        try:
            row = connection.execute(
                """
                SELECT embedding_json, embedding_model, embedding_version
                FROM insights
                WHERE id = ?
                """,
                ("insight-1",),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(len(json.loads(row[0])), 384)
        self.assertEqual(
            row[1],
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertEqual(row[2], 1)

    def test_response_update_seeds_tree_and_is_idempotent(self) -> None:
        extraction = GeneratedTreeUpdate(
            subject_label="Natural Law",
            subject_summary="Reason's participation in eternal law.",
            insight_candidate=GeneratedTreeInsightCandidate(
                label="Participation in Eternal Law",
                summary="Natural law participates rationally in eternal law.",
                evidence_excerpt="participation in eternal law",
            ),
        )

        first = self.service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            extraction,
        )
        second = self.service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            extraction,
        )
        snapshot = self.store.snapshot("conversation-1")

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "updated")
        self.assertEqual(len(snapshot["nodes"]), 1)
        self.assertEqual(snapshot["nodes"][0]["label"], "Natural Law")
        self.assertEqual(len(snapshot["nodes"][0]["insights"]), 1)
        self.assertEqual(
            snapshot["nodes"][0]["insights"][0]["title"],
            "Participation in Eternal Law",
        )
        self.assertEqual(len(first["added_insight_ids"]), 1)
        self.assertEqual(len(first["added_node_ids"]), 1)

    def test_generated_candidate_is_persisted_as_an_insight(self) -> None:
        extraction = GeneratedTreeUpdate(
            subject_label="Grace",
            subject_summary="Grace perfects nature.",
            insight_candidate=GeneratedTreeInsightCandidate(
                label="Grace Perfects Nature",
                summary="Grace elevates rather than destroys nature.",
                evidence_excerpt="Grace perfects nature",
            ),
        )
        result = self.service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            extraction,
        )

        nodes = self.store.snapshot("conversation-1")["nodes"]
        self.assertEqual(len(result["added_insight_ids"]), 1)
        self.assertEqual(len(result["added_node_ids"]), 1)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["label"], "Grace")
        self.assertEqual(len(nodes[0]["insights"]), 1)
        self.assertEqual(nodes[0]["insights"][0]["title"], "Grace Perfects Nature")
        self.assertEqual(
            nodes[0]["insights"][0]["source_type"],
            "automatic_response",
        )

    def test_empty_extraction_is_no_change(self) -> None:
        result = self.service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            GeneratedTreeUpdate("", "", None),
        )

        self.assertEqual(result["status"], "no_change")
        self.assertEqual(self.store.snapshot("conversation-1")["nodes"], [])

    def test_subject_seeds_an_empty_tree_without_inventing_insights(self) -> None:
        result = self.service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            GeneratedTreeUpdate(
                "Natural Law",
                "The inquiry concerns reason's participation in eternal law.",
                None,
            ),
        )

        snapshot = self.store.snapshot("conversation-1")

        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["added_insight_ids"], [])
        self.assertEqual(len(result["added_node_ids"]), 1)
        self.assertEqual(len(snapshot["nodes"]), 1)
        self.assertEqual(snapshot["nodes"][0]["label"], "Natural Law")
        self.assertEqual(snapshot["nodes"][0]["insights"], [])

        decision = self.service.assign_and_save(
            conversation_id="conversation-1",
            insight=TreeInsight(
                id="saved-natural-law",
                title="Natural Law",
                definition="Reason's participation in eternal law.",
            ),
            membership_threshold=0.0,
        )
        populated_snapshot = self.store.snapshot("conversation-1")

        self.assertEqual(decision.action, "attached")
        self.assertEqual(decision.node_id, result["added_node_ids"][0])
        self.assertEqual(
            [item["id"] for item in populated_snapshot["nodes"][0]["insights"]],
            ["saved-natural-law"],
        )

    def test_injected_scores_gate_duplicates_and_build_connected_mst(self) -> None:
        provider = InjectedScoreProvider()
        service = PersistentInsightTreeService(
            store=self.store,
            engine=InsightTreeEngine(provider),
            provider=provider,
        )

        for response_id, title in [
            ("response-1", "First Principle"),
            ("response-2", "Near Duplicate"),
            ("response-3", "Distinct Subject"),
            ("response-4", "Third Subject"),
        ]:
            service.apply_response_update(
                "conversation-1",
                response_id,
                "branch-1",
                GeneratedTreeUpdate(
                    title,
                    f"A summary of {title}.",
                    GeneratedTreeInsightCandidate(
                        label=title,
                        summary=f"The durable meaning of {title}.",
                        evidence_excerpt=title,
                    ),
                ),
            )

        snapshot = self.store.snapshot("conversation-1")
        node_labels = {node["label"] for node in snapshot["nodes"]}
        connected_node_ids = {
            endpoint
            for edge in snapshot["edges"]
            for endpoint in (edge["from_node_id"], edge["to_node_id"])
        }

        self.assertNotIn("Near Duplicate", node_labels)
        self.assertEqual(len(snapshot["nodes"]), 3)
        self.assertEqual(len(snapshot["edges"]), 2)
        self.assertEqual(
            connected_node_ids,
            {node["id"] for node in snapshot["nodes"]},
        )

    def test_initialize_migrates_existing_tree_tables(self) -> None:
        legacy_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE conversations (id TEXT PRIMARY KEY);
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version INTEGER NOT NULL,
                    needs_generated_label INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE insights (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    owning_node_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version INTEGER NOT NULL,
                    relatedness REAL NOT NULL,
                    distance REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

        InsightTreeStore(legacy_path).initialize()
        with sqlite3.connect(legacy_path) as connection:
            node_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(nodes)")
            }
            insight_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(insights)")
            }
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }

        self.assertIn("origin_node_id", node_columns)
        self.assertIn("source_response_id", insight_columns)
        self.assertIn("response_tree_analyses", tables)
        self.assertIn("node_edges", tables)

    def test_related_insight_is_saved_under_existing_node(self) -> None:
        first = self.service.assign_and_save(
            conversation_id="conversation-1",
            insight=TreeInsight(
                id="natural-law",
                title="Natural Law",
                definition="Reason's participation in God's eternal ordering.",
            ),
        )
        second = self.service.assign_and_save(
            conversation_id="conversation-1",
            insight=TreeInsight(
                id="moral-knowledge",
                title="Moral Knowledge",
                definition="Reason recognizes principles directing action toward the good.",
            ),
            membership_threshold=0.25,
        )

        snapshot = self.store.snapshot("conversation-1")

        self.assertEqual(second.action, "attached")
        self.assertEqual(second.node_id, first.node_id)
        self.assertEqual(len(snapshot["nodes"]), 1)
        self.assertEqual(len(snapshot["nodes"][0]["insights"]), 2)
        for insight in snapshot["nodes"][0]["insights"]:
            self.assertGreaterEqual(insight["relatedness"], 0.0)
            self.assertLessEqual(insight["relatedness"], 1.0)

    def test_conversations_remain_isolated(self) -> None:
        for conversation_id, insight_id in [
            ("conversation-1", "insight-1"),
            ("conversation-2", "insight-2"),
        ]:
            self.service.assign_and_save(
                conversation_id=conversation_id,
                insight=TreeInsight(
                    id=insight_id,
                    title="Existence",
                    definition="The act by which a thing is.",
                ),
            )

        first_snapshot = self.store.snapshot("conversation-1")
        second_snapshot = self.store.snapshot("conversation-2")

        self.assertEqual(
            first_snapshot["nodes"][0]["insights"][0]["id"],
            "insight-1",
        )
        self.assertEqual(
            second_snapshot["nodes"][0]["insights"][0]["id"],
            "insight-2",
        )

    def test_duplicate_insight_id_is_rejected(self) -> None:
        insight = TreeInsight(
            id="duplicate",
            title="Virtue",
            definition="A stable disposition toward good action.",
        )
        self.service.assign_and_save("conversation-1", insight)

        with self.assertRaisesRegex(ValueError, "already exists"):
            self.service.assign_and_save("conversation-1", insight)

    def test_manual_save_attaches_to_generated_insight_node(self) -> None:
        provider = InjectedScoreProvider()
        service = PersistentInsightTreeService(
            store=self.store,
            engine=InsightTreeEngine(provider),
            provider=provider,
        )
        service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            GeneratedTreeUpdate(
                subject_label="First Principles",
                subject_summary="The foundation of the inquiry.",
                insight_candidate=GeneratedTreeInsightCandidate(
                    label="First Principle",
                    summary="An automatic contextual definition.",
                    evidence_excerpt="first principle",
                ),
            ),
        )

        decision = service.assign_and_save(
            "conversation-1",
            TreeInsight(
                id="saved-first-principle",
                title="First Principle",
                definition="The manually saved contextual definition.",
            ),
        )
        snapshot = self.store.snapshot("conversation-1")
        stored = snapshot["nodes"][0]["insights"]

        self.assertEqual(decision.action, "attached")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["id"], "saved-first-principle")
        self.assertEqual(stored[0]["title"], "First Principle")
        self.assertEqual(stored[0]["source_type"], "saved_definition")

    def test_promoting_same_name_automatic_insight_requests_node_rename(self) -> None:
        self.service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            GeneratedTreeUpdate(
                subject_label="Predestination",
                subject_summary="Divine governance of creaturely ends.",
                insight_candidate=GeneratedTreeInsightCandidate(
                    label="Predestination",
                    summary="God orders creatures toward their final end.",
                    evidence_excerpt="predestination",
                ),
            ),
        )

        self.service.assign_and_save(
            "conversation-1",
            TreeInsight(
                id="saved-predestination",
                title="Predestination",
                definition="God's ordering of creatures toward their final end.",
            ),
        )

        snapshot = self.store.snapshot("conversation-1")
        self.assertTrue(snapshot["nodes"][0]["needs_generated_label"])

        # Older databases may have cleared this flag while retaining the collision.
        node_id = snapshot["nodes"][0]["id"]
        self.store.set_node_label("conversation-1", node_id, "Predestination")
        reconciled = self.store.snapshot("conversation-1")
        self.assertTrue(reconciled["nodes"][0]["needs_generated_label"])

    def test_snapshot_prefers_definition_generated_in_that_conversation(self) -> None:
        self.store.save_dynamic_definition(
            conversation_id="conversation-1",
            term_key="predestination",
            source_hash="source-1",
            requested_term="Predestination",
            source_excerpt="The current conversation's use.",
            definition=DynamicDefinitionRecord(
                title="Predestination",
                part_of_speech="noun",
                pronunciation="",
                definition="God decrees salvation and permits damnation.",
                example="",
                context="In the current conversation",
            ),
        )
        self.service.assign_and_save(
            "conversation-1",
            TreeInsight(
                id="saved-predestination",
                title="Predestination",
                definition=(
                    "In another conversation: God predetermines every event.\n"
                    "In the current conversation: God decrees salvation and permits damnation."
                ),
            ),
            suggested_node_label="Divine Providence",
        )

        snapshot = self.store.snapshot("conversation-1")

        self.assertEqual(
            snapshot["nodes"][0]["insights"][0]["definition"],
            "In the current conversation: God decrees salvation and permits damnation.",
        )

    def test_removing_an_insight_repairs_or_removes_its_node(self) -> None:
        first = TreeInsight(
            id="natural-law",
            title="Natural Law",
            definition="Reason's participation in God's eternal ordering.",
        )
        second = TreeInsight(
            id="moral-knowledge",
            title="Moral Knowledge",
            definition="Reason recognizes principles directing action toward the good.",
        )
        self.service.assign_and_save("conversation-1", first)
        self.service.assign_and_save(
            "conversation-1",
            second,
            membership_threshold=0.25,
        )

        self.service.remove_insight("conversation-1", first.id)
        snapshot = self.store.snapshot("conversation-1")

        self.assertEqual(len(snapshot["nodes"]), 1)
        self.assertEqual(
            [item["id"] for item in snapshot["nodes"][0]["insights"]],
            [second.id],
        )
        self.assertAlmostEqual(
            snapshot["nodes"][0]["insights"][0]["relatedness"],
            1.0,
            places=6,
        )

        self.service.remove_insight("conversation-1", second.id)
        self.assertEqual(self.store.snapshot("conversation-1")["nodes"], [])

    def test_removing_an_unknown_insight_is_idempotent(self) -> None:
        self.service.remove_insight("conversation-1", "missing")
        self.assertEqual(self.store.snapshot("conversation-1")["nodes"], [])

    def test_dynamic_definition_cache_is_scoped_to_conversation_and_source(self) -> None:
        record = DynamicDefinitionRecord(
            title="Analogy",
            part_of_speech="noun",
            pronunciation="uh-NAL-uh-jee",
            definition="A way of naming different things according to ordered likeness.",
            example="Being is said analogically of God and creatures.",
            context="Names applied across different subjects",
        )
        self.store.save_dynamic_definition(
            conversation_id="conversation-1",
            term_key="analogy",
            source_hash="source-a",
            requested_term="Analogy",
            source_excerpt="Analogy names likeness with difference.",
            definition=record,
        )

        cached = self.store.load_dynamic_definition(
            conversation_id="conversation-1",
            term_key="analogy",
            source_hash="source-a",
        )

        self.assertEqual(cached, record)
        self.assertIsNone(
            self.store.load_dynamic_definition(
                conversation_id="conversation-2",
                term_key="analogy",
                source_hash="source-a",
            )
        )
        self.assertIsNone(
            self.store.load_dynamic_definition(
                conversation_id="conversation-1",
                term_key="analogy",
                source_hash="source-b",
            )
        )

    def test_promoting_definition_flags_it_as_promoted(self) -> None:
        record = DynamicDefinitionRecord(
            title="Prudence",
            part_of_speech="noun",
            pronunciation="PROO-dns",
            definition="Right reason applied to action.",
            example="Prudence governs the choice of means to a good end.",
        )
        self.store.save_dynamic_definition(
            conversation_id="conversation-1",
            term_key="prudence",
            source_hash="source-a",
            requested_term="Prudence",
            source_excerpt="Prudence is the charioteer of the virtues.",
            definition=record,
        )

        glossed = self.store.find_glossed_term("conversation-1", staleness_hours=0)
        self.assertIsNotNone(glossed)
        self.assertEqual(glossed["term_key"], "prudence")

        self.store.mark_definition_promoted("conversation-1", "prudence")

        self.assertIsNone(
            self.store.find_glossed_term("conversation-1", staleness_hours=0)
        )

    def test_marking_unknown_term_promoted_is_a_no_op(self) -> None:
        # Retry-safe: no matching row, so this must not raise.
        self.store.mark_definition_promoted("conversation-1", "nonexistent")

    def test_glossed_term_excludes_terms_within_the_staleness_window(self) -> None:
        record = DynamicDefinitionRecord(
            title="Temperance",
            part_of_speech="noun",
            pronunciation="TEM-per-ns",
            definition="Moderation of desire for sensory pleasure.",
            example="Temperance restrains excess in food and drink.",
        )
        self.store.save_dynamic_definition(
            conversation_id="conversation-1",
            term_key="temperance",
            source_hash="source-a",
            requested_term="Temperance",
            source_excerpt="Temperance is one of the cardinal virtues.",
            definition=record,
        )

        self.assertIsNone(
            self.store.find_glossed_term("conversation-1", staleness_hours=24)
        )

    def test_find_related_node_matches_closest_node_above_threshold(self) -> None:
        provider = InjectedScoreProvider()
        service = PersistentInsightTreeService(
            store=self.store,
            engine=InsightTreeEngine(provider),
            provider=provider,
        )
        service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            GeneratedTreeUpdate(
                "First Principle",
                "A summary of First Principle.",
                GeneratedTreeInsightCandidate(
                    label="First Principle",
                    summary="The durable meaning of First Principle.",
                    evidence_excerpt="First Principle",
                ),
            ),
        )

        matched_node_id = self.store.find_related_node(
            "conversation-1", "First Principle", threshold=0.5, provider=provider
        )
        snapshot = self.store.snapshot("conversation-1")
        self.assertEqual(matched_node_id, snapshot["nodes"][0]["id"])

        # A threshold above every available similarity yields no match.
        self.assertIsNone(
            self.store.find_related_node(
                "conversation-1", "First Principle", threshold=1.5, provider=provider
            )
        )

    def test_find_related_node_returns_none_for_empty_conversation(self) -> None:
        provider = InjectedScoreProvider()
        self.assertIsNone(
            self.store.find_related_node(
                "conversation-1", "First Principle", threshold=0.5, provider=provider
            )
        )

    def test_flagged_quote_round_trip_is_idempotent_same_day(self) -> None:
        self.assertIsNone(self.store.find_surfaceable_quote("conversation-1"))

        self.store.save_flagged_quote(
            conversation_id="conversation-1",
            response_id="response-1",
            quote_text="Order in the universe implies an orderer.",
            source="heuristic_llm",
            reason="Original synthesis about design.",
        )
        # Retried save of the same (conversation, response, source) is a no-op.
        self.store.save_flagged_quote(
            conversation_id="conversation-1",
            response_id="response-1",
            quote_text="Order in the universe implies an orderer.",
            source="heuristic_llm",
            reason="Original synthesis about design.",
        )

        first_read = self.store.find_surfaceable_quote("conversation-1")
        self.assertIsNotNone(first_read)
        self.assertEqual(first_read["response_id"], "response-1")

        second_read = self.store.find_surfaceable_quote("conversation-1")
        self.assertEqual(second_read["response_id"], first_read["response_id"])

    def test_flagged_quote_respects_resurface_cooldown(self) -> None:
        self.store.save_flagged_quote(
            conversation_id="conversation-1",
            response_id="response-1",
            quote_text="A quote worth resurfacing.",
            source="user_flagged",
            reason=None,
        )
        surfaced = self.store.find_surfaceable_quote("conversation-1", cooldown_days=14)
        self.assertIsNotNone(surfaced)

        # Still within the cooldown window and already surfaced today, so a
        # second flagged quote for a different response must not replace it.
        self.store.save_flagged_quote(
            conversation_id="conversation-1",
            response_id="response-2",
            quote_text="A different quote.",
            source="user_flagged",
            reason=None,
        )
        still_same = self.store.find_surfaceable_quote("conversation-1", cooldown_days=14)
        self.assertEqual(still_same["response_id"], "response-1")

    def test_loose_thread_excludes_strongly_connected_nodes(self) -> None:
        class LooseThreadProvider:
            dimensions = 2
            vectors = {
                "Connected A": np.asarray([1.0, 0.0], dtype=np.float32),
                "Connected B": np.asarray([0.75, 0.6614378], dtype=np.float32),
                "Isolated C": np.asarray([0.0, -1.0], dtype=np.float32),
            }

            def embed(self, text):
                for title, vector in self.vectors.items():
                    if text.startswith(title):
                        return vector
                raise AssertionError(f"Missing injected vector for {text}")

            def compare_embeddings(self, left, right):
                similarity = float(np.dot(left, right))
                return {
                    "similarity": similarity,
                    "relatedness": max(0.0, similarity),
                    "distance": 1.0 - similarity,
                }

            def centroid(self, embeddings):
                center = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
                return center / np.linalg.norm(center)

        provider = LooseThreadProvider()
        service = PersistentInsightTreeService(
            store=self.store,
            engine=InsightTreeEngine(provider),
            provider=provider,
        )
        # A high membership_threshold forces each insight to become its own
        # Node even though Connected A/B are similar enough (0.75) to end up
        # with a strong edge between their two Nodes once rebuilt.
        for insight_id, title in [
            ("a", "Connected A"),
            ("b", "Connected B"),
            ("c", "Isolated C"),
        ]:
            service.assign_and_save(
                conversation_id="conversation-1",
                insight=TreeInsight(id=insight_id, title=title, definition=f"Definition of {title}."),
                membership_threshold=0.9,
            )

        loose_thread = self.store.find_loose_thread("conversation-1")
        self.assertIsNotNone(loose_thread)
        self.assertEqual(loose_thread["node_label"], "Isolated C")

    def test_loose_thread_tie_break_prefers_more_insights(self) -> None:
        provider = InjectedScoreProvider()
        service = PersistentInsightTreeService(
            store=self.store,
            engine=InsightTreeEngine(provider),
            provider=provider,
        )
        service.apply_response_update(
            "conversation-1",
            "response-1",
            "branch-1",
            GeneratedTreeUpdate(
                "Distinct Subject",
                "A summary of Distinct Subject.",
                GeneratedTreeInsightCandidate(
                    label="Distinct Subject",
                    summary="The durable meaning of Distinct Subject.",
                    evidence_excerpt="Distinct Subject",
                ),
            ),
        )
        service.apply_response_update(
            "conversation-1",
            "response-2",
            "branch-1",
            GeneratedTreeUpdate(
                "Third Subject",
                "A summary of Third Subject.",
                GeneratedTreeInsightCandidate(
                    label="Third Subject",
                    summary="The durable meaning of Third Subject.",
                    evidence_excerpt="Third Subject",
                ),
            ),
        )
        # A second insight attached to Third Subject's node gives it the higher
        # insight count, so it should win the tie-break over Distinct Subject
        # -- neither has any edge crossing the strong threshold.
        service.assign_and_save(
            conversation_id="conversation-1",
            insight=TreeInsight(
                id="extra-insight",
                title="Third Subject Detail",
                definition="A closely related elaboration.",
            ),
            membership_threshold=0.05,
        )

        loose_thread = self.store.find_loose_thread("conversation-1")
        self.assertIsNotNone(loose_thread)
        self.assertEqual(loose_thread["node_label"], "Third Subject")
        self.assertEqual(loose_thread["insight_count"], 2)

    def test_loose_thread_returns_none_for_empty_conversation(self) -> None:
        self.assertIsNone(self.store.find_loose_thread("conversation-1"))


if __name__ == "__main__":
    unittest.main()
