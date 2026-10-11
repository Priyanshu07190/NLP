import unittest
from src.retrieval.query_parser import parse_query, TermNode, PhraseNode, AndNode, OrNode, NotNode


class TestQueryParser(unittest.TestCase):
    def test_single_term(self):
        ast = parse_query("capitalism")
        self.assertIsInstance(ast, TermNode)
        self.assertEqual(ast.term, "capitalism")

    def test_exact_phrase(self):
        ast = parse_query('"shareholder capitalism"')
        self.assertIsInstance(ast, PhraseNode)
        self.assertEqual(ast.raw_phrase, "shareholder capitalism")
        self.assertIn("shareholder", ast.terms)
        self.assertIn("capitalism", ast.terms)

    def test_boolean_and_or_not(self):
        ast = parse_query('biden AND pension')
        self.assertIsInstance(ast, AndNode)

        ast_or = parse_query('deficit OR debt')
        self.assertIsInstance(ast_or, OrNode)

        ast_not = parse_query('NOT pension')
        self.assertIsInstance(ast_not, NotNode)

    def test_nested_parentheses(self):
        ast = parse_query('(biden OR trump) AND "tax cuts"')
        self.assertIsInstance(ast, AndNode)
        self.assertIsInstance(ast.left, OrNode)
        self.assertIsInstance(ast.right, PhraseNode)

    def test_rejects_unquoted_ampersand(self):
        with self.assertRaises(ValueError) as ctx:
            parse_query('biden & pension')
        self.assertIn("& is not a supported Boolean operator", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
