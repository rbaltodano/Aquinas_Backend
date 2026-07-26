import unittest

import numpy as np

from relatedness import MiniLMRelatednessProvider


class MiniLMRelatednessProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = MiniLMRelatednessProvider()
        cls.provider.load()

    def test_embedding_has_expected_shape_and_is_normalized(self) -> None:
        embedding = self.provider.embed(
            "Natural law is the rational creature's participation in eternal law."
        )

        self.assertEqual(embedding.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)

    def test_related_text_is_closer_than_unrelated_text(self) -> None:
        natural_law = "Natural law directs human action toward the good."
        related = self.provider.compare(
            natural_law,
            "Moral law guides people to choose what is good.",
        )
        unrelated = self.provider.compare(
            natural_law,
            "A carpenter uses a plane to smooth a wooden board.",
        )

        self.assertGreater(related["similarity"], unrelated["similarity"])
        self.assertAlmostEqual(
            related["distance"],
            1.0 - related["similarity"],
            places=6,
        )

    def test_centroid_is_normalized(self) -> None:
        embeddings = self.provider.embed_many(
            [
                "Essence describes what a thing is.",
                "Existence describes that a thing is.",
            ]
        )

        centroid = self.provider.centroid(list(embeddings))

        self.assertEqual(centroid.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
