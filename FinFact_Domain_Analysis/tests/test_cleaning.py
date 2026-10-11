import unittest
from src.preprocessing.cleaner import ConservativeFinancialCleaner


class TestConservativeFinancialCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = ConservativeFinancialCleaner()

    def test_preserves_currency_and_rates(self):
        text = "Texas will have about $11 billion in reserve and 5.5% growth in FY2025."
        cleaned = self.cleaner.clean(text)
        self.assertIn("$11 billion", cleaned)
        self.assertIn("5.5%", cleaned)
        self.assertIn("FY2025", cleaned)

    def test_preserves_401k(self):
        text = "Biden wants to end shareholder capitalism... that's your 401(k) pension."
        cleaned = self.cleaner.clean(text)
        self.assertIn("401(k)", cleaned)

    def test_repairs_glued_punctuation(self):
        text = "In the Feb. 25, 2015 interview,which we caught online, Patrick said so."
        cleaned = self.cleaner.clean(text)
        self.assertIn("interview, which", cleaned)

    def test_filters_scraper_boilerplate(self):
        text = "Become a Founding Member to help us hire more fact-checkers. Snopes investigated the claim."
        cleaned = self.cleaner.clean(text)
        self.assertNotIn("Become a Founding Member", cleaned)
        self.assertIn("Snopes investigated the claim.", cleaned)


if __name__ == "__main__":
    unittest.main()
