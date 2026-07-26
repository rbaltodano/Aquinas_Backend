import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from insight_tree import InsightTreeEngine, TreeInsight
from relatedness import MiniLMRelatednessProvider
from structured_generation import GeneratedTreeNodeSeed, GeneratedTreeUpdate
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
            node_seed=GeneratedTreeNodeSeed(
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
        self.assertEqual(snapshot["nodes"][0]["label"], "Participation in Eternal Law")
        self.assertEqual(snapshot["nodes"][0]["insights"], [])
        self.assertEqual(first["added_insight_ids"], [])
        self.assertEqual(len(first["added_node_ids"]), 1)

    def test_generated_seed_is_never_persisted_as_an_insight(self) -> None:
        extraction = GeneratedTreeUpdate(
            subject_label="Grace",
            subject_summary="Grace perfects nature.",
            node_seed=GeneratedTreeNodeSeed(
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
        self.assertEqual(result["added_insight_ids"], [])
        self.assertEqual(len(result["added_node_ids"]), 1)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["label"], "Grace Perfects Nature")
        self.assertEqual(nodes[0]["insights"], [])

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
            engine=self.engine,
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
                    GeneratedTreeNodeSeed(
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

    def test_manual_save_attaches_to_generated_node_seed(self) -> None:
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
                node_seed=GeneratedTreeNodeSeed(
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


if __name__ == "__main__":
    unittest.main()
