import unittest

from today_in_history import TODAY_IN_HISTORY_ENTRIES, entry_for_date


class TodayInHistoryDatasetTests(unittest.TestCase):
    def test_entry_for_date_returns_none_for_a_date_not_in_the_seed_set(self) -> None:
        self.assertIsNone(entry_for_date("02-30"))

    def test_seed_set_has_no_duplicate_dates(self) -> None:
        dates = [entry.date for entry in TODAY_IN_HISTORY_ENTRIES]
        self.assertEqual(len(dates), len(set(dates)))

    def test_seed_set_entries_have_required_fields_populated(self) -> None:
        for entry in TODAY_IN_HISTORY_ENTRIES:
            self.assertRegex(entry.date, r"^\d{2}-\d{2}$")
            self.assertTrue(entry.title.strip())
            self.assertTrue(entry.description.strip())
            self.assertTrue(entry.related_entity.strip())

    def test_entry_for_date_returns_the_matching_seeded_entry(self) -> None:
        entry = entry_for_date("03-07")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.related_entity, "Thomas Aquinas")


if __name__ == "__main__":
    unittest.main()
