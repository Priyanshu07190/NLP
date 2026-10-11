"""Abstract Syntax Tree (AST) Recursive Descent Boolean Query Parser.

Supports:
- Single terms (e.g. capitalism)
- Exact quoted phrases (e.g. "shareholder capitalism")
- Boolean operators: AND, OR, NOT
- Nested parenthesized expressions: (biden OR trump) AND "tax cuts"
- Strict operator validation (rejects standalone unquoted '&' with clear hints)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# AST Node Hierarchy
# ---------------------------------------------------------------------------
@dataclass
class QueryNode:
    pass


@dataclass
class TermNode(QueryNode):
    term: str


@dataclass
class PhraseNode(QueryNode):
    terms: List[str]
    raw_phrase: str


@dataclass
class AndNode(QueryNode):
    left: QueryNode
    right: QueryNode


@dataclass
class OrNode(QueryNode):
    left: QueryNode
    right: QueryNode


@dataclass
class NotNode(QueryNode):
    child: QueryNode


# ---------------------------------------------------------------------------
# Lexer / Tokenizer
# ---------------------------------------------------------------------------
class TokenType:
    TERM = "TERM"
    PHRASE = "PHRASE"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"


@dataclass
class QueryToken:
    type: str
    value: str


def tokenize_query(query_str: str) -> List[QueryToken]:
    # Check for invalid standalone unquoted '&'
    if re.search(r'(?<!["\w])&+(?!["\w])', query_str):
        raise ValueError(
            "& is not a supported Boolean operator. Use AND instead. (e.g. \"shareholder capitalism\" AND \"pension\")"
        )

    tokens: List[QueryToken] = []
    i = 0
    n = len(query_str)

    while i < n:
        c = query_str[i]

        if c.isspace():
            i += 1
            continue

        if c == "(":
            tokens.append(QueryToken(TokenType.LPAREN, "("))
            i += 1
            continue

        if c == ")":
            tokens.append(QueryToken(TokenType.RPAREN, ")"))
            i += 1
            continue

        # Quoted phrase: "..."
        if c in ('"', "'"):
            quote_char = c
            start = i + 1
            end = query_str.find(quote_char, start)
            if end == -1:
                # Unclosed quote - consume rest as phrase
                phrase_text = query_str[start:].strip()
                tokens.append(QueryToken(TokenType.PHRASE, phrase_text))
                i = n
            else:
                phrase_text = query_str[start:end].strip()
                tokens.append(QueryToken(TokenType.PHRASE, phrase_text))
                i = end + 1
            continue

        # Word / Identifier
        start = i
        while i < n and not query_str[i].isspace() and query_str[i] not in '()"\'':
            i += 1
        word = query_str[start:i]

        upper = word.upper()
        if upper == "AND":
            tokens.append(QueryToken(TokenType.AND, "AND"))
        elif upper == "OR":
            tokens.append(QueryToken(TokenType.OR, "OR"))
        elif upper == "NOT":
            tokens.append(QueryToken(TokenType.NOT, "NOT"))
        else:
            tokens.append(QueryToken(TokenType.TERM, word))

    tokens.append(QueryToken(TokenType.EOF, ""))
    return tokens


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# Grammar:
#   Expr     ::= OrExpr
#   OrExpr   ::= AndExpr ( 'OR' AndExpr )*
#   AndExpr  ::= NotExpr ( 'AND'? NotExpr )*    (implicit AND allowed)
#   NotExpr  ::= 'NOT' NotExpr | Primary
#   Primary  ::= '(' Expr ')' | PHRASE | TERM
# ---------------------------------------------------------------------------
class QueryParser:
    def __init__(self, tokens: List[QueryToken]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> QueryToken:
        return self.tokens[self.pos]

    def consume(self, expected_type: Optional[str] = None) -> QueryToken:
        tok = self.current()
        if expected_type and tok.type != expected_type:
            raise ValueError(f"Syntax error: expected {expected_type}, got {tok.type} ('{tok.value}')")
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if self.current().type == TokenType.EOF:
            return TermNode("")
        node = self.expr()
        if self.current().type != TokenType.EOF:
            raise ValueError(f"Unexpected token at end of query: '{self.current().value}'")
        return node

    def expr(self) -> QueryNode:
        return self.or_expr()

    def or_expr(self) -> QueryNode:
        node = self.and_expr()
        while self.current().type == TokenType.OR:
            self.consume(TokenType.OR)
            right = self.and_expr()
            node = OrNode(left=node, right=right)
        return node

    def and_expr(self) -> QueryNode:
        node = self.not_expr()
        while True:
            if self.current().type == TokenType.AND:
                self.consume(TokenType.AND)
                right = self.not_expr()
                node = AndNode(left=node, right=right)
            elif self.current().type in (TokenType.TERM, TokenType.PHRASE, TokenType.LPAREN, TokenType.NOT):
                # Implicit AND between adjacent terms (e.g. biden capitalism)
                right = self.not_expr()
                node = AndNode(left=node, right=right)
            else:
                break
        return node

    def not_expr(self) -> QueryNode:
        if self.current().type == TokenType.NOT:
            self.consume(TokenType.NOT)
            child = self.not_expr()
            return NotNode(child=child)
        return self.primary()

    def primary(self) -> QueryNode:
        tok = self.current()

        if tok.type == TokenType.LPAREN:
            self.consume(TokenType.LPAREN)
            node = self.expr()
            self.consume(TokenType.RPAREN)
            return node

        if tok.type == TokenType.PHRASE:
            self.consume(TokenType.PHRASE)
            from ..preprocessing.pipeline import pipeline_a
            # Tokenize & lemmatize phrase terms
            res = pipeline_a.process_text(tok.value)
            phrase_terms = [t for t, _ in res["indexed_terms"]]
            return PhraseNode(terms=phrase_terms, raw_phrase=tok.value)

        if tok.type == TokenType.TERM:
            self.consume(TokenType.TERM)
            from ..preprocessing.pipeline import pipeline_a
            res = pipeline_a.process_text(tok.value)
            if res["indexed_terms"]:
                lemma = res["indexed_terms"][0][0]
            else:
                lemma = tok.value.casefold()
            return TermNode(term=lemma)

        raise ValueError(f"Unexpected token in query: '{tok.value}'")


def parse_query(query_str: str) -> QueryNode:
    tokens = tokenize_query(query_str.strip())
    parser = QueryParser(tokens)
    return parser.parse()
