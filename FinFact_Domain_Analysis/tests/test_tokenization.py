import unittest
from src.preprocessing.tokenizer import CustomFinanceTokenizer
from src.preprocessing.lemmatizer import FinancialLemmatizer
from src.preprocessing.stopwords import DomainAwareStopwordFilter
from src.preprocessing.pos_tagger import FinancialPOSTagger, penn_to_wordnet
from src.preprocessing.pipeline import pipeline_a_baseline, pipeline_a_pos


class TestFinancialTokenizationAndMorphology(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CustomFinanceTokenizer()
        self.lemmatizer = FinancialLemmatizer()
        self.stopwords = DomainAwareStopwordFilter()
        self.pos_tagger = FinancialPOSTagger()

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

    def test_lemmatizer_baseline_without_pos(self):
        self.assertEqual(self.lemmatizer.lemmatize("investments"), "investment")
        self.assertEqual(self.lemmatizer.lemmatize("liquidating"), "liquidate")
        self.assertEqual(self.lemmatizer.lemmatize("companies"), "company")
        self.assertEqual(self.lemmatizer.lemmatize("reserves"), "reserve")
        self.assertEqual(self.lemmatizer.lemmatize("taxes"), "tax")

    def test_pos_guided_lemmatization(self):
        # With POS tag 'v' (verb), 'increased' lemmatizes to valid verb 'increase'
        self.assertEqual(self.lemmatizer.lemmatize("increased", pos="v"), "increase")
        self.assertEqual(self.lemmatizer.lemmatize("spending", pos="v"), "spend")
        self.assertEqual(self.lemmatizer.lemmatize("taxes", pos="n"), "tax")

    def test_pos_tagger_penn_treebank(self):
        tagged = self.pos_tagger.tag_tokens(["Biden", "increased", "corporate", "taxes", "by", "5.5%"])
        tag_dict = dict(tagged)
        self.assertEqual(tag_dict.get("Biden"), "NNP")
        self.assertEqual(tag_dict.get("increased"), "VBD")
        self.assertEqual(tag_dict.get("corporate"), "JJ")
        self.assertEqual(tag_dict.get("taxes"), "NNS")
        self.assertEqual(tag_dict.get("5.5%"), "CD")

    def test_noun_phrase_extraction(self):
        tagged = self.pos_tagger.tag_tokens(["Biden", "increased", "corporate", "taxes", "and", "rainy", "day", "fund"])
        nps = self.pos_tagger.extract_noun_phrases(tagged)
        self.assertIn("corporate taxes", nps)
        self.assertIn("rainy day fund", nps)

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

    def test_pipeline_comparison_processors(self):
        text = "Biden increased corporate taxes by 5.5%"
        res_base = pipeline_a_baseline.process_text(text)
        res_pos = pipeline_a_pos.process_text(text)
        self.assertEqual(res_base["pipeline_variant"] if "pipeline_variant" in res_base else len(res_base["tokens"]), 6)
        self.assertEqual(len(res_pos["tokens"]), 6)
        self.assertTrue(any(t["pos_tag"] == "VBD" for t in res_pos["tokens"]))


if __name__ == "__main__":
    unittest.main()
