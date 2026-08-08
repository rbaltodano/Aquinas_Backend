import unittest

from ingest_corpus import CHUNK_MAX_WORDS, chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_section_marker_forces_a_chunk_boundary(self) -> None:
        book_seven_tail = "word " * 20 + "Such were the events in Sicily."
        book_eight_marker = "[History of the Peloponnesian War/Book 8]"
        book_eight_head = "Book 8\nThe Twentieth Year of the War begins."
        text = f"{book_seven_tail}\n\n{book_eight_marker}\n\n{book_eight_head}"

        chunks = chunk_text(text)

        self.assertEqual(len(chunks), 2)
        self.assertNotIn("Book 8", chunks[0])
        self.assertIn("Such were the events in Sicily.", chunks[0])
        self.assertIn("[History of the Peloponnesian War/Book 8]", chunks[1])

    def test_marker_glued_to_its_own_extract_stays_together(self) -> None:
        text = "[History of the Peloponnesian War/Book 8]\nThe war continued."

        chunks = chunk_text(text)

        self.assertEqual(len(chunks), 1)
        self.assertIn("[History of the Peloponnesian War/Book 8]", chunks[0])
        self.assertIn("The war continued.", chunks[0])

    def test_oversized_paragraph_still_splits_by_word_count(self) -> None:
        text = "word " * (CHUNK_MAX_WORDS + 50)

        chunks = chunk_text(text.strip())

        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
