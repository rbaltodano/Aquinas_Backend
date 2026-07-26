import unittest

from insight_tree import InsightTreeEngine, TreeInsight, TreeNode
from relatedness import MiniLMRelatednessProvider


class InsightTreeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = InsightTreeEngine(MiniLMRelatednessProvider())

    def test_first_insight_creates_a_node(self) -> None:
        insight = TreeInsight(
            id="insight-1",
            title="Natural Law",
            definition="The rational creature's participation in eternal law.",
        )

        result = self.engine.assign_new_insight(insight, [])

        self.assertEqual(result.action, "created")
        self.assertEqual(result.node_label, "Natural Law")
        self.assertEqual(result.member_insight_ids, ("insight-1",))
        self.assertTrue(result.needs_generated_label)

    def test_related_insight_attaches_to_best_node(self) -> None:
        natural_law_node = TreeNode(
            id="node-natural-law",
            label="Moral Law",
            insights=(
                TreeInsight(
                    id="existing-natural-law",
                    title="Natural Law",
                    definition="Reason's participation in God's eternal ordering.",
                ),
            ),
        )
        craft_node = TreeNode(
            id="node-craft",
            label="Craft",
            insights=(
                TreeInsight(
                    id="existing-carpentry",
                    title="Carpentry",
                    definition="The craft of shaping wooden objects with tools.",
                ),
            ),
        )
        new_insight = TreeInsight(
            id="insight-conscience",
            title="Moral Knowledge",
            definition="Human reason recognizes principles that direct action toward the good.",
        )

        result = self.engine.assign_new_insight(
            new_insight,
            [craft_node, natural_law_node],
            membership_threshold=0.25,
        )

        self.assertEqual(result.action, "attached")
        self.assertEqual(result.node_id, "node-natural-law")
        self.assertEqual(
            result.member_insight_ids,
            ("existing-natural-law", "insight-conscience"),
        )
        self.assertFalse(result.needs_generated_label)

    def test_unrelated_insight_creates_a_new_node(self) -> None:
        node = TreeNode(
            id="node-carpentry",
            label="Carpentry",
            insights=(
                TreeInsight(
                    id="existing-carpentry",
                    title="Carpentry",
                    definition="The craft of shaping wooden objects with tools.",
                ),
            ),
        )
        insight = TreeInsight(
            id="insight-grace",
            title="Grace",
            definition="God's gift elevating human nature toward supernatural life.",
        )

        result = self.engine.assign_new_insight(
            insight,
            [node],
            membership_threshold=0.40,
        )

        self.assertEqual(result.action, "created")
        self.assertNotEqual(result.node_id, node.id)

    def test_rejects_duplicate_insight_membership(self) -> None:
        duplicate = TreeInsight(
            id="duplicate",
            title="Virtue",
            definition="A stable disposition toward good action.",
        )
        nodes = [
            TreeNode(id="node-1", label="One", insights=(duplicate,)),
            TreeNode(id="node-2", label="Two", insights=(duplicate,)),
        ]

        with self.assertRaisesRegex(ValueError, "only one"):
            self.engine.assign_new_insight(
                TreeInsight(
                    id="new",
                    title="Prudence",
                    definition="Right reason applied to action.",
                ),
                nodes,
            )


if __name__ == "__main__":
    unittest.main()
