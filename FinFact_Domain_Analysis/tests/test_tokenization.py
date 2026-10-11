import unittest
from src.preprocessing.tokenizer import CustomFinanceTokenizer
from src.preprocessing.lemmatizer import FinancialLemmatizer
from src.preprocessing.stopwords import DomainAwareStopwordFilter


class TestFinancialTokenizationAndMorphology(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CustomFinanceTokenizer()
        self.lemmatizer = FinancialLemmatizer()
        self.stopwords = DomainAwareStopwordFilter()

    def test_preserves_atomic_financial_tokens(self):
        text = "The fund holds $11B with 5.5% interest and 25bps margin."
        tokens = [t.text for t in self.tokenizer.tokenize(text)]
        self.assertIn("$11B", tokens)
        self.assertIn("5.5%", tokens)
        self.assertIn("25bps", tokens)

    def test_preserves_401k_and_hyphens(self):
        text = "COVID-19 impacted 401(k) accounts under shareholder-capitalism."
        tokens = [t.text for t in self.tokenizer.tokenize(text)]
        self.assertIn("401(k)", tokens)
        self.assertIn("COVID-19", tokens)
        self.assertIn("shareholder-capitalism", tokens)

    def test_lemmatizer_pipeline_a(self):
        self.assertEqual(self.lemmatizer.lemmatize("investments"), "investment")
        self.assertEqual(self.lemmatizer.lemmatize("liquidating"), "liquidate")
        self.assertEqual(self.lemmatizer.lemmatize("companies"), "company")
        self.assertEqual(self.lemmatizer.lemmatize("reserves"), "reserve")
        self.assertEqual(self.lemmatizer.lemmatize("taxes"), "tax")

    def test_polarity_stopword_preservation(self):
        self.assertFalse(self.stopwords.is_stopword("not"))
        self.assertFalse(self.stopwords.is_stopword("never"))
        self.assertFalse(self.stopwords.is_stopword("no"))
        self.assertFalse(self.stopwords.is_stopword("without"))

    def test_financial_keyword_stopword_preservation(self):
        self.assertFalse(self.stopwords.is_stopword("tax"))
        self.assertFalse(self.stopwords.is_stopword("debt"))
        self.assertFalse(self.stopwords.is_stopword("bank"))
        self.assertFalse(self.stopwords.is_stopword("rate"))
        self.assertFalse(self.stopwords.is_stopword("pension"))


if __name__ == "__main__":
    unittest.main()
