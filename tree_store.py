"""SQLite persistence for per-conversation Insight Trees."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterator
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from insight_tree import (
    DEFAULT_MEMBERSHIP_THRESHOLD,
    AssignmentDecision,
    InsightTreeEngine,
    TreeInsight,
    TreeNode,
    insight_tree_engine,
)
from relatedness import (
    DEFAULT_MODEL_NAME,
    Embedding,
    MiniLMRelatednessProvider,
    relatedness_provider,
)


EMBEDDING_VERSION = 1
INSIGHT_DUPLICATE_THRESHOLD = 0.86
NEW_NODE_COHESION_THRESHOLD = 0.55
NODE_EDGE_STRONG_THRESHOLD = 0.70
DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "insight_tree.sqlite3"


@dataclass(frozen=True)
class DynamicDefinitionRecord:
    title: str
    part_of_speech: str
    pronunciation: str
    definition: str
    example: str
    context: str = ""


class InsightTreeStore:
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version INTEGER NOT NULL,
                    needs_generated_label INTEGER NOT NULL DEFAULT 1,
                    origin_node_id TEXT,
                    is_seed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS nodes_conversation_index
                    ON nodes(conversation_id);

                CREATE TABLE IF NOT EXISTS insights (
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
                    source_type TEXT NOT NULL DEFAULT 'saved_definition',
                    source_response_id TEXT,
                    source_branch_id TEXT,
                    evidence_excerpt TEXT NOT NULL DEFAULT '',
                    extraction_role TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (owning_node_id) REFERENCES nodes(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS insights_conversation_index
                    ON insights(conversation_id);

                CREATE INDEX IF NOT EXISTS insights_node_index
                    ON insights(owning_node_id);

                CREATE TABLE IF NOT EXISTS dynamic_definitions (
                    conversation_id TEXT NOT NULL,
                    term_key TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    requested_term TEXT NOT NULL,
                    source_excerpt TEXT NOT NULL,
                    title TEXT NOT NULL,
                    part_of_speech TEXT NOT NULL,
                    pronunciation TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    example TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (conversation_id, term_key, source_hash),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS dynamic_definitions_conversation_index
                    ON dynamic_definitions(conversation_id);

                CREATE TABLE IF NOT EXISTS node_edges (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    from_node_id TEXT NOT NULL,
                    to_node_id TEXT NOT NULL,
                    relatedness REAL NOT NULL,
                    distance REAL NOT NULL,
                    is_strong_extra INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (from_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (to_node_id) REFERENCES nodes(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS node_edges_conversation_index
                    ON node_edges(conversation_id);

                CREATE TABLE IF NOT EXISTS response_tree_analyses (
                    conversation_id TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    analysis_id TEXT,
                    mutation_id TEXT,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (conversation_id, response_id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS insight_tombstones (
                    conversation_id TEXT NOT NULL,
                    source_response_id TEXT NOT NULL,
                    title_key TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (conversation_id, source_response_id, title_key),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column(connection, "nodes", "origin_node_id", "TEXT")
            self._ensure_column(
                connection,
                "nodes",
                "is_seed",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                connection,
                "insights",
                "source_type",
                "TEXT NOT NULL DEFAULT 'saved_definition'",
            )
            self._ensure_column(connection, "insights", "source_response_id", "TEXT")
            self._ensure_column(connection, "insights", "source_branch_id", "TEXT")
            self._ensure_column(
                connection,
                "insights",
                "evidence_excerpt",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(connection, "insights", "extraction_role", "TEXT")
            self._ensure_column(
                connection, "response_tree_analyses", "analysis_id", "TEXT"
            )
            self._ensure_column(
                connection, "response_tree_analyses", "mutation_id", "TEXT"
            )
            self._ensure_column(
                connection,
                "dynamic_definitions",
                "context",
                "TEXT NOT NULL DEFAULT ''",
            )

    def load_nodes(self, conversation_id: str) -> list[TreeNode]:
        with self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, label, embedding_json
                FROM nodes
                WHERE conversation_id = ?
                ORDER BY created_at, id
                """,
                (conversation_id,),
            ).fetchall()

            nodes: list[TreeNode] = []
            for node_row in node_rows:
                insight_rows = connection.execute(
                    """
                    SELECT id, title, definition, embedding_json
                    FROM insights
                    WHERE conversation_id = ? AND owning_node_id = ?
                    ORDER BY created_at, id
                    """,
                    (conversation_id, node_row["id"]),
                ).fetchall()
                nodes.append(
                    TreeNode(
                        id=node_row["id"],
                        label=node_row["label"],
                        insights=tuple(
                            TreeInsight(
                                id=row["id"],
                                title=row["title"],
                                definition=row["definition"],
                                embedding=self._decode_embedding(row["embedding_json"]),
                            )
                            for row in insight_rows
                        ),
                        embedding=self._decode_embedding(
                            node_row["embedding_json"]
                        ),
                    )
                )
            return nodes

    def save_assignment(
        self,
        conversation_id: str,
        insight: TreeInsight,
        decision: AssignmentDecision,
        node_embedding: Embedding,
        member_scores: dict[str, dict[str, float]],
    ) -> None:
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT OR IGNORE INTO conversations(id) VALUES (?)",
                    (conversation_id,),
                )

                if decision.action == "created":
                    connection.execute(
                        """
                        INSERT INTO nodes(
                            id, conversation_id, label, embedding_json,
                            embedding_model, embedding_version, needs_generated_label
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            decision.node_id,
                            conversation_id,
                            decision.node_label,
                            self._encode_embedding(node_embedding),
                            DEFAULT_MODEL_NAME,
                            EMBEDDING_VERSION,
                            int(decision.needs_generated_label),
                        ),
                    )
                else:
                    owner = connection.execute(
                        "SELECT conversation_id FROM nodes WHERE id = ?",
                        (decision.node_id,),
                    ).fetchone()
                    if owner is None or owner["conversation_id"] != conversation_id:
                        raise ValueError("The selected owning Node is not in this conversation.")
                    connection.execute(
                        """
                        UPDATE nodes
                        SET embedding_json = ?, embedding_model = ?, embedding_version = ?,
                            needs_generated_label = CASE
                                WHEN ? THEN 1 ELSE needs_generated_label END
                        WHERE id = ?
                        """,
                        (
                            self._encode_embedding(node_embedding),
                            DEFAULT_MODEL_NAME,
                            EMBEDDING_VERSION,
                            int(decision.needs_generated_label),
                            decision.node_id,
                        ),
                    )

                new_score = member_scores[insight.id]
                connection.execute(
                    """
                    INSERT INTO insights(
                        id, conversation_id, owning_node_id, title, definition,
                        embedding_json, embedding_model, embedding_version,
                        relatedness, distance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        insight.id,
                        conversation_id,
                        decision.node_id,
                        insight.title,
                        insight.definition,
                        self._encode_embedding(decision.insight_embedding),
                        DEFAULT_MODEL_NAME,
                        EMBEDDING_VERSION,
                        new_score["relatedness"],
                        new_score["distance"],
                    ),
                )

                for insight_id, score in member_scores.items():
                    connection.execute(
                        """
                        UPDATE insights
                        SET relatedness = ?, distance = ?
                        WHERE id = ? AND owning_node_id = ?
                        """,
                        (
                            score["relatedness"],
                            score["distance"],
                            insight_id,
                            decision.node_id,
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "An Insight or Node with that ID already exists."
                ) from error

    def promote_matching_automatic_insight(
        self,
        conversation_id: str,
        insight: TreeInsight,
        provider: MiniLMRelatednessProvider,
    ) -> AssignmentDecision | None:
        """Replace an automatic term alias with its manually saved definition.

        Response extraction occasionally names a candidate ``"<term> defined"`` or
        ``"<term> definition"``. When that highlighted term is later bookmarked,
        it is the same conceptual Insight, not a second tree member.
        """
        requested_key = self._canonical_tree_title_key(insight.title)
        with self._connect() as connection:
            automatic_rows = connection.execute(
                """
                SELECT id, owning_node_id, title
                FROM insights
                WHERE conversation_id = ? AND source_type = 'automatic_response'
                ORDER BY created_at, id
                """,
                (conversation_id,),
            ).fetchall()
            matches = [
                row
                for row in automatic_rows
                if self._canonical_tree_title_key(row["title"]) == requested_key
            ]
            if not matches:
                return None

            existing_saved = connection.execute(
                """
                SELECT id, owning_node_id
                FROM insights
                WHERE conversation_id = ? AND id = ?
                """,
                (conversation_id, insight.id),
            ).fetchone()
            affected_node_ids = {row["owning_node_id"] for row in matches}
            insight_embedding = provider.embed(insight.semantic_text)

            if existing_saved is not None:
                owner_node_id = existing_saved["owning_node_id"]
                connection.executemany(
                    "DELETE FROM insights WHERE id = ? AND conversation_id = ?",
                    [(row["id"], conversation_id) for row in matches],
                )
                affected_node_ids.add(owner_node_id)
            else:
                promoted = matches[0]
                owner_node_id = promoted["owning_node_id"]
                connection.execute(
                    """
                    UPDATE insights
                    SET id = ?, title = ?, definition = ?, embedding_json = ?,
                        embedding_model = ?, embedding_version = ?,
                        source_type = 'saved_definition', source_response_id = NULL,
                        source_branch_id = NULL, evidence_excerpt = '',
                        extraction_role = NULL
                    WHERE id = ? AND conversation_id = ?
                    """,
                    (
                        insight.id,
                        insight.title,
                        insight.definition,
                        self._encode_embedding(insight_embedding),
                        DEFAULT_MODEL_NAME,
                        EMBEDDING_VERSION,
                        promoted["id"],
                        conversation_id,
                    ),
                )
                connection.executemany(
                    "DELETE FROM insights WHERE id = ? AND conversation_id = ?",
                    [
                        (row["id"], conversation_id)
                        for row in matches[1:]
                    ],
                )

            for node_id in affected_node_ids:
                self._repair_node(connection, node_id, provider)
            self._rebuild_node_edges(connection, conversation_id, provider)

            owner = connection.execute(
                """
                SELECT label, needs_generated_label
                FROM nodes WHERE id = ? AND conversation_id = ?
                """,
                (owner_node_id, conversation_id),
            ).fetchone()
            needs_generated_label = bool(owner["needs_generated_label"]) if owner else False
            if owner is not None and (
                self._canonical_tree_title_key(owner["label"])
                == self._canonical_tree_title_key(insight.title)
            ):
                needs_generated_label = True
                connection.execute(
                    """
                    UPDATE nodes SET needs_generated_label = 1
                    WHERE id = ? AND conversation_id = ?
                    """,
                    (owner_node_id, conversation_id),
                )
            stored = connection.execute(
                """
                SELECT relatedness, distance, embedding_json
                FROM insights WHERE id = ? AND conversation_id = ?
                """,
                (insight.id, conversation_id),
            ).fetchone()
            member_ids = tuple(
                row["id"]
                for row in connection.execute(
                    """
                    SELECT id FROM insights
                    WHERE owning_node_id = ? ORDER BY created_at, id
                    """,
                    (owner_node_id,),
                ).fetchall()
            )
            if owner is None or stored is None:
                raise ValueError("Unable to promote the matching automatic Insight.")
            return AssignmentDecision(
                action="attached",
                node_id=owner_node_id,
                node_label=owner["label"],
                relatedness=stored["relatedness"],
                distance=stored["distance"],
                member_insight_ids=member_ids,
                evaluated_nodes=(),
                needs_generated_label=needs_generated_label,
                insight_embedding=self._decode_embedding(stored["embedding_json"]),
            )

    def reconcile_saved_automatic_aliases(
        self,
        provider: MiniLMRelatednessProvider,
    ) -> int:
        """Clean up aliases written before save-time promotion was introduced."""
        with self._connect() as connection:
            saved_rows = connection.execute(
                """
                SELECT conversation_id, id, title, definition
                FROM insights
                WHERE source_type = 'saved_definition'
                ORDER BY created_at, id
                """
            ).fetchall()

        reconciled = 0
        for row in saved_rows:
            result = self.promote_matching_automatic_insight(
                conversation_id=row["conversation_id"],
                insight=TreeInsight(
                    id=row["id"],
                    title=row["title"],
                    definition=row["definition"],
                ),
                provider=provider,
            )
            if result is not None:
                reconciled += 1
        return reconciled

    def remove_insight(
        self,
        conversation_id: str,
        insight_id: str,
        provider: MiniLMRelatednessProvider,
    ) -> None:
        """Remove one Insight and repair only its owning Node's centroid and scores."""
        with self._connect() as connection:
            insight_row = connection.execute(
                """
                SELECT owning_node_id, source_response_id, title
                FROM insights
                WHERE id = ? AND conversation_id = ?
                """,
                (insight_id, conversation_id),
            ).fetchone()
            if insight_row is None:
                return

            node_id = insight_row["owning_node_id"]
            if insight_row["source_response_id"]:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO insight_tombstones(
                        conversation_id, source_response_id, title_key
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        conversation_id,
                        insight_row["source_response_id"],
                        self._title_key(insight_row["title"]),
                    ),
                )
            connection.execute(
                "DELETE FROM insights WHERE id = ? AND conversation_id = ?",
                (insight_id, conversation_id),
            )
            remaining_rows = connection.execute(
                """
                SELECT id, embedding_json
                FROM insights
                WHERE conversation_id = ? AND owning_node_id = ?
                ORDER BY created_at, id
                """,
                (conversation_id, node_id),
            ).fetchall()

            if not remaining_rows:
                seed_row = connection.execute(
                    """
                    SELECT label, summary, is_seed
                    FROM nodes
                    WHERE id = ? AND conversation_id = ?
                    """,
                    (node_id, conversation_id),
                ).fetchone()
                if seed_row is not None and seed_row["is_seed"]:
                    seed_embedding = provider.embed(
                        f"{seed_row['label']}. {seed_row['summary']}"
                    )
                    connection.execute(
                        """
                        UPDATE nodes
                        SET embedding_json = ?, embedding_model = ?,
                            embedding_version = ?
                        WHERE id = ? AND conversation_id = ?
                        """,
                        (
                            self._encode_embedding(seed_embedding),
                            DEFAULT_MODEL_NAME,
                            EMBEDDING_VERSION,
                            node_id,
                            conversation_id,
                        ),
                    )
                else:
                    connection.execute(
                        "DELETE FROM nodes WHERE id = ? AND conversation_id = ?",
                        (node_id, conversation_id),
                    )
            else:
                embeddings = [
                    self._decode_embedding(row["embedding_json"])
                    for row in remaining_rows
                ]
                node_embedding = provider.centroid(embeddings)
                connection.execute(
                    """
                    UPDATE nodes
                    SET embedding_json = ?, embedding_model = ?, embedding_version = ?
                    WHERE id = ? AND conversation_id = ?
                    """,
                    (
                        self._encode_embedding(node_embedding),
                        DEFAULT_MODEL_NAME,
                        EMBEDDING_VERSION,
                        node_id,
                        conversation_id,
                    ),
                )
                for row, embedding in zip(remaining_rows, embeddings):
                    score = provider.compare_embeddings(embedding, node_embedding)
                    connection.execute(
                        """
                        UPDATE insights
                        SET relatedness = ?, distance = ?
                        WHERE id = ? AND conversation_id = ?
                        """,
                        (
                            score["relatedness"],
                            score["distance"],
                            row["id"],
                            conversation_id,
                        ),
                    )
            self._rebuild_node_edges(connection, conversation_id, provider)

    def rebuild_node_edges(
        self,
        conversation_id: str,
        provider: MiniLMRelatednessProvider,
    ) -> None:
        with self._connect() as connection:
            self._rebuild_node_edges(connection, conversation_id, provider)

    def set_node_label(
        self,
        conversation_id: str,
        node_id: str,
        label: str,
    ) -> None:
        cleaned = " ".join(label.split())
        if not cleaned:
            raise ValueError("Node label cannot be blank.")
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE nodes
                SET label = ?, needs_generated_label = 0
                WHERE id = ? AND conversation_id = ?
                """,
                (cleaned, node_id, conversation_id),
            )
            if updated.rowcount == 0:
                raise ValueError("The Node is not in this conversation.")

    def load_response_analysis(
        self,
        conversation_id: str,
        response_id: str,
    ) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json
                FROM response_tree_analyses
                WHERE conversation_id = ? AND response_id = ?
                """,
                (conversation_id, response_id),
            ).fetchone()
            return json.loads(row["result_json"]) if row is not None else None

    def apply_response_update(
        self,
        conversation_id: str,
        response_id: str,
        branch_id: str,
        extraction,
        provider: MiniLMRelatednessProvider,
        engine: InsightTreeEngine = insight_tree_engine,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    ) -> dict:
        """Atomically filter, assign, persist, and connect one generated update."""
        with self._connect() as connection:
            existing_result = connection.execute(
                """
                SELECT result_json
                FROM response_tree_analyses
                WHERE conversation_id = ? AND response_id = ?
                """,
                (conversation_id, response_id),
            ).fetchone()
            if existing_result is not None:
                return json.loads(existing_result["result_json"])

            connection.execute(
                "INSERT OR IGNORE INTO conversations(id) VALUES (?)",
                (conversation_id,),
            )
            existing_insights = connection.execute(
                """
                SELECT id, title, embedding_json
                FROM insights
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchall()
            nodes = self.load_nodes(conversation_id)
            added_insight_ids: list[str] = []
            added_node_ids: list[str] = []

            if extraction.insight_candidate is not None:
                candidate = extraction.insight_candidate
                candidate_id = self._stable_id(
                    f"{conversation_id}:{response_id}:automatic-insight:"
                    f"{self._title_key(candidate.label)}"
                )
                candidate_insight = TreeInsight(
                    id=candidate_id,
                    title=candidate.label,
                    definition=candidate.summary,
                )
                candidate_embedding = provider.embed(candidate_insight.semantic_text)
                is_duplicate = any(
                    provider.compare_embeddings(
                        candidate_embedding,
                        self._decode_embedding(row["embedding_json"]),
                    )["similarity"] >= INSIGHT_DUPLICATE_THRESHOLD
                    for row in existing_insights
                )
                if not is_duplicate:
                    decision = engine.assign_new_insight(
                        insight=TreeInsight(
                            id=candidate_id,
                            title=candidate.label,
                            definition=candidate.summary,
                            embedding=candidate_embedding,
                        ),
                        nodes=nodes,
                        membership_threshold=membership_threshold,
                        suggested_node_label=(
                            extraction.subject_label or candidate.label
                        ),
                    )
                    existing_owner = next(
                        (node for node in nodes if node.id == decision.node_id),
                        None,
                    )
                    member_embeddings = (
                        [
                            member.embedding
                            for member in existing_owner.insights
                            if member.embedding is not None
                        ]
                        if existing_owner is not None
                        else []
                    )
                    member_embeddings.append(decision.insight_embedding)
                    node_embedding = provider.centroid(member_embeddings)

                    if decision.action == "created":
                        connection.execute(
                            """
                            INSERT INTO nodes(
                                id, conversation_id, label, summary, embedding_json,
                                embedding_model, embedding_version,
                                needs_generated_label, is_seed
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                            """,
                            (
                                decision.node_id,
                                conversation_id,
                                decision.node_label,
                                extraction.subject_summary,
                                self._encode_embedding(node_embedding),
                                DEFAULT_MODEL_NAME,
                                EMBEDDING_VERSION,
                                int(decision.needs_generated_label),
                            ),
                        )
                        added_node_ids.append(decision.node_id)
                    else:
                        connection.execute(
                            """
                            UPDATE nodes
                            SET embedding_json = ?, embedding_model = ?,
                                embedding_version = ?,
                                needs_generated_label = CASE
                                    WHEN ? THEN 1 ELSE needs_generated_label END
                            WHERE id = ? AND conversation_id = ?
                            """,
                            (
                                self._encode_embedding(node_embedding),
                                DEFAULT_MODEL_NAME,
                                EMBEDDING_VERSION,
                                int(decision.needs_generated_label),
                                decision.node_id,
                                conversation_id,
                            ),
                        )

                    member_scores: dict[str, dict[str, float]] = {}
                    if existing_owner is not None:
                        for member in existing_owner.insights:
                            if member.embedding is None:
                                raise ValueError(
                                    "A saved Insight is missing its embedding."
                                )
                            member_scores[member.id] = (
                                provider.compare_embeddings(
                                    member.embedding,
                                    node_embedding,
                                )
                            )
                    member_scores[candidate_id] = provider.compare_embeddings(
                        decision.insight_embedding,
                        node_embedding,
                    )
                    candidate_score = member_scores[candidate_id]
                    connection.execute(
                        """
                        INSERT INTO insights(
                            id, conversation_id, owning_node_id, title, definition,
                            embedding_json, embedding_model, embedding_version,
                            relatedness, distance, source_type, source_response_id,
                            source_branch_id, evidence_excerpt, extraction_role
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            candidate_id,
                            conversation_id,
                            decision.node_id,
                            candidate.label,
                            candidate.summary,
                            self._encode_embedding(decision.insight_embedding),
                            DEFAULT_MODEL_NAME,
                            EMBEDDING_VERSION,
                            candidate_score["relatedness"],
                            candidate_score["distance"],
                            "automatic_response",
                            response_id,
                            branch_id,
                            candidate.evidence_excerpt,
                            "durable_insight",
                        ),
                    )
                    for insight_id, score in member_scores.items():
                        connection.execute(
                            """
                            UPDATE insights
                            SET relatedness = ?, distance = ?
                            WHERE id = ? AND owning_node_id = ?
                            """,
                            (
                                score["relatedness"],
                                score["distance"],
                                insight_id,
                                decision.node_id,
                            ),
                        )
                    added_insight_ids.append(candidate_id)
            elif (
                not nodes
                and extraction.subject_label
                and extraction.subject_summary
            ):
                subject_node_id = self._stable_id(
                    f"{conversation_id}:{response_id}:subject-node"
                )
                subject_embedding = provider.embed(
                    f"{extraction.subject_label}. {extraction.subject_summary}"
                )
                connection.execute(
                    """
                    INSERT INTO nodes(
                        id, conversation_id, label, summary, embedding_json,
                        embedding_model, embedding_version, needs_generated_label,
                        origin_node_id, is_seed
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, 1)
                    """,
                    (
                        subject_node_id,
                        conversation_id,
                        extraction.subject_label,
                        extraction.subject_summary,
                        self._encode_embedding(subject_embedding),
                        DEFAULT_MODEL_NAME,
                        EMBEDDING_VERSION,
                    )
                )
                added_node_ids.append(subject_node_id)

            self._rebuild_node_edges(connection, conversation_id, provider)

            did_mutate = bool(added_insight_ids or added_node_ids)
            result = {
                "status": "updated" if did_mutate else "no_change",
                "analysis_id": self._stable_id(
                    f"{conversation_id}:{response_id}:analysis"
                ),
                "mutation_id": (
                    self._stable_id(f"{conversation_id}:{response_id}:mutation")
                    if did_mutate
                    else None
                ),
                "added_insight_ids": added_insight_ids,
                "added_node_ids": added_node_ids,
            }
            connection.execute(
                """
                INSERT INTO response_tree_analyses(
                    conversation_id, response_id, branch_id, analysis_id,
                    mutation_id, result_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    response_id,
                    branch_id,
                    result["analysis_id"],
                    result["mutation_id"],
                    json.dumps(result),
                ),
            )
            return result

    def _repair_node(
        self,
        connection: sqlite3.Connection,
        node_id: str,
        provider: MiniLMRelatednessProvider,
    ) -> None:
        rows = connection.execute(
            """
            SELECT id, embedding_json FROM insights
            WHERE owning_node_id = ? ORDER BY created_at, id
            """,
            (node_id,),
        ).fetchall()
        if not rows:
            connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            return
        embeddings = [self._decode_embedding(row["embedding_json"]) for row in rows]
        centroid = provider.centroid(embeddings)
        connection.execute(
            """
            UPDATE nodes SET embedding_json = ?, embedding_model = ?,
                embedding_version = ? WHERE id = ?
            """,
            (
                self._encode_embedding(centroid),
                DEFAULT_MODEL_NAME,
                EMBEDDING_VERSION,
                node_id,
            ),
        )
        for row, embedding in zip(rows, embeddings):
            score = provider.compare_embeddings(embedding, centroid)
            connection.execute(
                "UPDATE insights SET relatedness = ?, distance = ? WHERE id = ?",
                (score["relatedness"], score["distance"], row["id"]),
            )

    def _evaluate_buds(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
        response_id: str,
        affected_node_ids: set[str],
        label: str,
        summary: str,
        new_insight_ids: set[str],
        provider: MiniLMRelatednessProvider,
    ) -> list[str]:
        bud_node_ids: list[str] = []
        for origin_node_id in tuple(affected_node_ids):
            loose = connection.execute(
                """
                SELECT id, embedding_json FROM insights
                WHERE owning_node_id = ? AND relatedness < ?
                ORDER BY created_at, id
                """,
                (origin_node_id, DEFAULT_MEMBERSHIP_THRESHOLD),
            ).fetchall()
            if len(loose) <= 3 or not any(
                row["id"] in new_insight_ids for row in loose
            ):
                continue
            cohesive = []
            for row in loose:
                embedding = self._decode_embedding(row["embedding_json"])
                if all(
                    provider.compare_embeddings(
                        embedding,
                        self._decode_embedding(other["embedding_json"]),
                    )["similarity"] >= NEW_NODE_COHESION_THRESHOLD
                    for other in cohesive
                ):
                    cohesive.append(row)
            if len(cohesive) < 3:
                continue
            bud_id = self._stable_id(
                f"{conversation_id}:{response_id}:bud:{origin_node_id}"
            )
            bud_embeddings = [
                self._decode_embedding(row["embedding_json"]) for row in cohesive
            ]
            connection.execute(
                """
                INSERT INTO nodes(
                    id, conversation_id, label, summary, embedding_json,
                    embedding_model, embedding_version, needs_generated_label,
                    origin_node_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    bud_id,
                    conversation_id,
                    label,
                    summary,
                    self._encode_embedding(provider.centroid(bud_embeddings)),
                    DEFAULT_MODEL_NAME,
                    EMBEDDING_VERSION,
                    origin_node_id,
                ),
            )
            connection.executemany(
                "UPDATE insights SET owning_node_id = ? WHERE id = ?",
                [(bud_id, row["id"]) for row in cohesive],
            )
            self._repair_node(connection, origin_node_id, provider)
            self._repair_node(connection, bud_id, provider)
            bud_node_ids.append(bud_id)
        return bud_node_ids

    def _rebuild_node_edges(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
        provider: MiniLMRelatednessProvider,
    ) -> None:
        connection.execute(
            "DELETE FROM node_edges WHERE conversation_id = ?",
            (conversation_id,),
        )
        rows = connection.execute(
            """
            SELECT id, embedding_json, origin_node_id FROM nodes
            WHERE conversation_id = ? ORDER BY created_at, id
            """,
            (conversation_id,),
        ).fetchall()
        if len(rows) < 2:
            return
        pairs = []
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                score = provider.compare_embeddings(
                    self._decode_embedding(left["embedding_json"]),
                    self._decode_embedding(right["embedding_json"]),
                )
                pairs.append((left["id"], right["id"], score))

        parent = {row["id"]: row["id"] for row in rows}

        def find(node_id: str) -> str:
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id

        selected: dict[tuple[str, str], tuple[dict, bool]] = {}
        for left_id, right_id, score in sorted(
            pairs,
            key=lambda item: item[2]["distance"],
        ):
            left_root, right_root = find(left_id), find(right_id)
            if left_root != right_root:
                parent[left_root] = right_root
                selected[tuple(sorted((left_id, right_id)))] = (score, False)
        for left_id, right_id, score in pairs:
            key = tuple(sorted((left_id, right_id)))
            if score["similarity"] >= NODE_EDGE_STRONG_THRESHOLD and key not in selected:
                selected[key] = (score, True)
        pair_scores = {
            frozenset((left_id, right_id)): score
            for left_id, right_id, score in pairs
        }
        for row in rows:
            origin_id = row["origin_node_id"]
            if not origin_id:
                continue
            left_id, right_id = sorted((row["id"], origin_id))
            key = (left_id, right_id)
            if key not in selected:
                score = pair_scores.get(frozenset((left_id, right_id)))
                if score is not None:
                    selected[key] = (score, True)

        for (left_id, right_id), (score, is_extra) in selected.items():
            edge_id = self._stable_id(
                f"{conversation_id}:edge:{min(left_id, right_id)}:{max(left_id, right_id)}"
            )
            connection.execute(
                """
                INSERT INTO node_edges(
                    id, conversation_id, from_node_id, to_node_id,
                    relatedness, distance, is_strong_extra
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_id,
                    conversation_id,
                    left_id,
                    right_id,
                    score["relatedness"],
                    score["distance"],
                    int(is_extra),
                ),
            )

    def snapshot(self, conversation_id: str) -> dict:
        with self._connect() as connection:
            dynamic_rows = connection.execute(
                """
                SELECT term_key, title, definition, context
                FROM dynamic_definitions
                WHERE conversation_id = ?
                ORDER BY updated_at DESC, created_at DESC, source_hash DESC
                """,
                (conversation_id,),
            ).fetchall()
            conversation_definitions: dict[str, sqlite3.Row] = {}
            for row in dynamic_rows:
                for candidate in (row["term_key"], row["title"]):
                    key = self._canonical_tree_title_key(candidate)
                    conversation_definitions.setdefault(key, row)

            node_rows = connection.execute(
                """
                SELECT id, label, summary, needs_generated_label, origin_node_id
                FROM nodes
                WHERE conversation_id = ?
                ORDER BY created_at, id
                """,
                (conversation_id,),
            ).fetchall()

            nodes = []
            for node in node_rows:
                insight_rows = connection.execute(
                    """
                    SELECT id, title, definition, relatedness, distance,
                           source_type, source_response_id, source_branch_id,
                           evidence_excerpt, extraction_role
                    FROM insights
                    WHERE conversation_id = ? AND owning_node_id = ?
                    ORDER BY created_at, id
                    """,
                    (conversation_id, node["id"]),
                ).fetchall()
                serialized_insights = []
                for row in insight_rows:
                    serialized = dict(row)
                    if row["source_type"] == "saved_definition":
                        contextual = conversation_definitions.get(
                            self._canonical_tree_title_key(row["title"])
                        )
                        if contextual is not None:
                            context = contextual["context"].strip()
                            definition = contextual["definition"].strip()
                            serialized["definition"] = (
                                f"{context}: {definition}" if context else definition
                            )
                    serialized_insights.append(serialized)
                label_matches_insight = any(
                    self._canonical_tree_title_key(node["label"])
                    == self._canonical_tree_title_key(row["title"])
                    for row in insight_rows
                )
                nodes.append(
                    {
                        "id": node["id"],
                        "label": node["label"],
                        "summary": node["summary"],
                        "needs_generated_label": (
                            bool(node["needs_generated_label"])
                            or label_matches_insight
                        ),
                        "origin_node_id": node["origin_node_id"],
                        "insights": serialized_insights,
                    }
                )

            edge_rows = connection.execute(
                """
                SELECT id, from_node_id, to_node_id, relatedness, distance,
                       is_strong_extra
                FROM node_edges
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
            return {
                "conversation_id": conversation_id,
                "embedding_model": DEFAULT_MODEL_NAME,
                "embedding_version": EMBEDDING_VERSION,
                "nodes": nodes,
                "edges": [
                    {
                        **dict(row),
                        "is_strong_extra": bool(row["is_strong_extra"]),
                    }
                    for row in edge_rows
                ],
            }

    def load_dynamic_definition(
        self,
        conversation_id: str,
        term_key: str,
        source_hash: str,
    ) -> DynamicDefinitionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT title, part_of_speech, pronunciation, definition, example, context
                FROM dynamic_definitions
                WHERE conversation_id = ? AND term_key = ? AND source_hash = ?
                """,
                (conversation_id, term_key, source_hash),
            ).fetchone()
            if row is None:
                return None
            return DynamicDefinitionRecord(
                title=row["title"],
                part_of_speech=row["part_of_speech"],
                pronunciation=row["pronunciation"],
                definition=row["definition"],
                example=row["example"],
                context=row["context"],
            )

    def save_dynamic_definition(
        self,
        conversation_id: str,
        term_key: str,
        source_hash: str,
        requested_term: str,
        source_excerpt: str,
        definition: DynamicDefinitionRecord,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations(id) VALUES (?)",
                (conversation_id,),
            )
            connection.execute(
                """
                INSERT INTO dynamic_definitions(
                    conversation_id, term_key, source_hash, requested_term, source_excerpt,
                    title, part_of_speech, pronunciation, definition, example, context
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id, term_key, source_hash) DO UPDATE SET
                    requested_term = excluded.requested_term,
                    source_excerpt = excluded.source_excerpt,
                    title = excluded.title,
                    part_of_speech = excluded.part_of_speech,
                    pronunciation = excluded.pronunciation,
                    definition = excluded.definition,
                    example = excluded.example,
                    context = excluded.context,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    conversation_id,
                    term_key,
                    source_hash,
                    requested_term,
                    source_excerpt,
                    definition.title,
                    definition.part_of_speech,
                    definition.pronunciation,
                    definition.definition,
                    definition.example,
                    definition.context,
                ),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _encode_embedding(embedding: Embedding) -> str:
        return json.dumps(np.asarray(embedding, dtype=np.float32).tolist())

    @staticmethod
    def _decode_embedding(value: str) -> Embedding:
        return np.asarray(json.loads(value), dtype=np.float32)

    @staticmethod
    def _stable_id(value: str) -> str:
        return str(uuid5(NAMESPACE_URL, value))

    @staticmethod
    def _title_key(value: str) -> str:
        return " ".join(value.casefold().split())

    @classmethod
    def _canonical_tree_title_key(cls, value: str) -> str:
        key = cls._title_key(value).strip(" .,:;!?")
        for suffix in (" defined", " definition"):
            if key.endswith(suffix):
                return key[: -len(suffix)].rstrip()
        return key

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )


class PersistentInsightTreeService:
    """Serializes assignment and storage so one conversation update is consistent."""

    def __init__(
        self,
        store: InsightTreeStore,
        engine: InsightTreeEngine = insight_tree_engine,
        provider: MiniLMRelatednessProvider = relatedness_provider,
    ) -> None:
        self.store = store
        self.engine = engine
        self.provider = provider
        self._mutation_lock = Lock()

    def assign_and_save(
        self,
        conversation_id: str,
        insight: TreeInsight,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        suggested_node_label: str | None = None,
    ) -> AssignmentDecision:
        with self._mutation_lock:
            promoted = self.store.promote_matching_automatic_insight(
                conversation_id=conversation_id,
                insight=insight,
                provider=self.provider,
            )
            if promoted is not None:
                return promoted

            nodes = self.store.load_nodes(conversation_id)
            decision = self.engine.assign_new_insight(
                insight=insight,
                nodes=nodes,
                membership_threshold=membership_threshold,
                suggested_node_label=suggested_node_label,
            )

            existing_owner = next(
                (node for node in nodes if node.id == decision.node_id),
                None,
            )
            member_embeddings = (
                [
                    member.embedding
                    for member in existing_owner.insights
                    if member.embedding is not None
                ]
                if existing_owner is not None
                else []
            )
            member_embeddings.append(decision.insight_embedding)
            node_embedding = self.provider.centroid(member_embeddings)

            member_scores: dict[str, dict[str, float]] = {}
            if existing_owner is not None:
                for member in existing_owner.insights:
                    if member.embedding is None:
                        raise ValueError("A saved Insight is missing its embedding.")
                    member_scores[member.id] = self.provider.compare_embeddings(
                        member.embedding,
                        node_embedding,
                    )
            member_scores[insight.id] = self.provider.compare_embeddings(
                decision.insight_embedding,
                node_embedding,
            )

            self.store.save_assignment(
                conversation_id=conversation_id,
                insight=insight,
                decision=decision,
                node_embedding=node_embedding,
                member_scores=member_scores,
            )
            self.store.rebuild_node_edges(conversation_id, self.provider)
            return decision

    def promote_matching_automatic_insight(
        self,
        conversation_id: str,
        insight: TreeInsight,
    ) -> AssignmentDecision | None:
        with self._mutation_lock:
            return self.store.promote_matching_automatic_insight(
                conversation_id=conversation_id,
                insight=insight,
                provider=self.provider,
            )

    def remove_insight(self, conversation_id: str, insight_id: str) -> None:
        with self._mutation_lock:
            self.store.remove_insight(
                conversation_id=conversation_id,
                insight_id=insight_id,
                provider=self.provider,
            )

    def apply_response_update(
        self,
        conversation_id: str,
        response_id: str,
        branch_id: str,
        extraction,
    ) -> dict:
        with self._mutation_lock:
            return self.store.apply_response_update(
                conversation_id=conversation_id,
                response_id=response_id,
                branch_id=branch_id,
                extraction=extraction,
                provider=self.provider,
                engine=self.engine,
            )


tree_store = InsightTreeStore()
persistent_tree_service = PersistentInsightTreeService(tree_store)
