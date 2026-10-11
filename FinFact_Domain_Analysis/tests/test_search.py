import unittest
from src.indexing.inverted_index import PositionalInvertedIndex
from src.retrieval.search_engine import SearchEngine


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.index = PositionalInvertedIndex()
        # Unit 1: "biden wants to end shareholder capitalism"
        self.index.add_unit_terms("U1", [
            ("biden", 0), ("want", 1), ("end", 3), ("shareholder", 4), ("capitalism", 5)
        ])
        # Unit 2: "capitalism without shareholder accountability"
        self.index.add_unit_terms("U2", [
            ("capitalism", 0), ("without", 1), ("shareholder", 2), ("accountability", 3)
        ])
        # Unit 3: "pension funds invested in bonds"
        self.index.add_unit_terms("U3", [
            ("pension", 0), ("fund", 1), ("invest", 2), ("bond", 4)
        ])

        self.unit_store = {
            "U1": {"unit_id": "U1", "text": "biden wants to end shareholder capitalism", "label": "false"},
            "U2": {"unit_id": "U2", "text": "capitalism without shareholder accountability", "label": "true"},
            "U3": {"unit_id": "U3", "text": "pension funds invested in bonds", "label": "true"},
        }
        self.engine = SearchEngine(self.index, self.unit_store)

    def test_exact_phrase_positional_verification(self):
        # "shareholder capitalism": positions must be consecutive (4, 5 in U1; in U2 it is 2 then not 3)
        res = self.engine.search('"shareholder capitalism"')
        matched_ids = [r["unit_id"] for r in res["results"]]
        self.assertIn("U1", matched_ids)
        self.assertNotIn("U2", matched_ids)

    def test_boolean_and_query(self):
        res = self.engine.search('biden AND capitalism')
        matched_ids = [r["unit_id"] for r in res["results"]]
        self.assertEqual(matched_ids, ["U1"])

    def test_boolean_or_query(self):
        res = self.engine.search('biden OR pension')
        matched_ids = [r["unit_id"] for r in res["results"]]
        self.assertIn("U1", matched_ids)
        self.assertIn("U3", matched_ids)

    def test_boolean_not_query(self):
        res = self.engine.search('capitalism AND NOT biden')
        matched_ids = [r["unit_id"] for r in res["results"]]
        self.assertIn("U2", matched_ids)
        self.assertNotIn("U1", matched_ids)


if __name__ == "__main__":
    unittest.main()
