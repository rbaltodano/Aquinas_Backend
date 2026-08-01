"""Deterministic Insight Tree membership decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
from uuid import uuid4

from relatedness import Embedding, MiniLMRelatednessProvider, relatedness_provider


# Placeholder until calibrated against real Aquinas conversations.
DEFAULT_MEMBERSHIP_THRESHOLD = 0.40


@dataclass(frozen=True)
class TreeInsight:
    id: str
    title: str
    definition: str
    embedding: Embedding | None = None

    @property
    def semantic_text(self) -> str:
        return f"{self.title.strip()}. {self.definition.strip()}".strip()


@dataclass(frozen=True)
class TreeNode:
    id: str
    label: str
    insights: tuple[TreeInsight, ...]
    embedding: Embedding | None = None


@dataclass(frozen=True)
class EvaluatedNode:
    node_id: str
    similarity: float
    distance: float


@dataclass(frozen=True)
class AssignmentDecision:
    action: Literal["attached", "created"]
    node_id: str
    node_label: str
    relatedness: float
    distance: float
    member_insight_ids: tuple[str, ...]
    evaluated_nodes: tuple[EvaluatedNode, ...]
    needs_generated_label: bool
    insight_embedding: Embedding


class InsightTreeEngine:
    """Assigns a newly saved Insight without reorganizing settled membership."""

    def __init__(
        self,
        provider: MiniLMRelatednessProvider = relatedness_provider,
    ) -> None:
        self.provider = provider

    def assign_new_insight(
        self,
        insight: TreeInsight,
        nodes: Sequence[TreeNode],
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        suggested_node_label: str | None = None,
    ) -> AssignmentDecision:
        if not 0.0 <= membership_threshold <= 1.0:
            raise ValueError("Membership threshold must be between 0 and 1.")
        self._validate_topology(insight, nodes)

        insight_embedding = self._embedding_for(insight)
        evaluated: list[EvaluatedNode] = []

        for node in nodes:
            node_embedding = node.embedding
            if node_embedding is None:
                member_embeddings = [
                    self._embedding_for(member)
                    for member in node.insights
                ]
                node_embedding = self.provider.centroid(list(member_embeddings))
            score = self.provider.compare_embeddings(
                insight_embedding,
                node_embedding,
            )
            evaluated.append(
                EvaluatedNode(
                    node_id=node.id,
                    similarity=score["similarity"],
                    distance=score["distance"],
                )
            )

        evaluated.sort(key=lambda item: item.similarity, reverse=True)
        best = evaluated[0] if evaluated else None

        if best is not None and best.similarity >= membership_threshold:
            owner = next(node for node in nodes if node.id == best.node_id)
            return AssignmentDecision(
                action="attached",
                node_id=owner.id,
                node_label=owner.label,
                relatedness=max(0.0, min(1.0, best.similarity)),
                distance=best.distance,
                member_insight_ids=tuple(
                    [member.id for member in owner.insights] + [insight.id]
                ),
                evaluated_nodes=tuple(evaluated),
                needs_generated_label=self._labels_match(owner.label, insight.title),
                insight_embedding=insight_embedding,
            )

        generated_label = (suggested_node_label or "").strip()
        node_label = generated_label or insight.title.strip()
        return AssignmentDecision(
            action="created",
            node_id=str(uuid4()),
            node_label=node_label,
            relatedness=1.0,
            distance=0.0,
            member_insight_ids=(insight.id,),
            evaluated_nodes=tuple(evaluated),
            needs_generated_label=(
                not bool(generated_label)
                or self._labels_match(node_label, insight.title)
            ),
            insight_embedding=insight_embedding,
        )

    @staticmethod
    def _labels_match(left: str, right: str) -> bool:
        def canonical(value: str) -> str:
            return " ".join(value.casefold().strip(" .,:;!?").split())

        return canonical(left) == canonical(right)

    def _embedding_for(self, insight: TreeInsight) -> Embedding:
        if insight.embedding is not None:
            return insight.embedding
        return self.provider.embed(insight.semantic_text)

    @staticmethod
    def _validate_topology(
        new_insight: TreeInsight,
        nodes: Sequence[TreeNode],
    ) -> None:
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Existing Node IDs must be unique.")

        existing_insight_ids = [
            insight.id
            for node in nodes
            for insight in node.insights
        ]
        if len(existing_insight_ids) != len(set(existing_insight_ids)):
            raise ValueError("An Insight can belong to only one existing Node.")
        if new_insight.id in existing_insight_ids:
            raise ValueError("The new Insight already exists in the tree.")
        if any(not node.insights and node.embedding is None for node in nodes):
            raise ValueError(
                "An empty existing Node must carry its seed embedding."
            )


insight_tree_engine = InsightTreeEngine()
