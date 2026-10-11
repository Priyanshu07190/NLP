# Fin-Fact Domain-Specific NLP & Information Retrieval System
## Financial Fact-Checking, Claim Verification & Evidence Retrieval Engine

**Course:** Natural Language Processing (Financial Domain Text Retrieval)  
**Dataset:** Fin-Fact Multimodal Fact-Checking Corpus (`finfact.json`)  
**Implementation:** Pipeline A (Lemmatization + Positional Postings Indexing) — Fully Validated, Tested, and Deployed  

---

## 1. Executive Summary

This project implements an end-to-end, production-grade **Domain-Specific Natural Language Processing (NLP) and Information Retrieval (IR) System** tailored to the **Fin-Fact** financial fact-checking dataset. 

Building upon the empirically validated best architecture from prior comparative experiments (**Pipeline A: Lemmatization**), the system extracts, cleans, indexes, and searches over **27,617 content units** derived from **3,369 financial claims**. The platform pairs an Abstract Syntax Tree (AST) Boolean retrieval engine with positional exact-phrase verification, multi-term evidence snippet generation, financial named entity recognition, and an interactive full-stack web application.

```
+-----------------------------------------------------------------------------------------+
|                                FIN-FACT IR ENGINE WORKFLOW                               |
+-----------------------------------------------------------------------------------------+
       |
       v
[Raw Dataset: finfact.json] (3,369 claims, 30.17 MB)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Ingestion & Unit Segmentation                                                  │
│ Output: 27,617 Content Units (Claims, Justifications, Digests, Evidence Sentences)      │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Domain-Aware Preprocessing (Pipeline A)                                        │
│ • Conservative Web & Boilerplate Cleaner (Filters scraper artifacts, repairs spacing)   │
│ • Custom Finance Tokenizer (Preserves $, ₹, %, bps, 401(k), FY2025 as atomic tokens)   │
│ • Polarity-Preserving Stopwords (Protects 'not', 'no', 'never' + 90 financial terms)    │
│ • Morphological Lemmatization (WordNet POS-guided lemmas; avoids stemmer truncation)     │
│ • Financial Named Entity Recognition (MONEY, PERCENT, DATE, ORG, PERSON)                │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Positional Inverted Index & Search Engine                                      │
│ • Positional Postings: (unit_id, tf, [pos_1, pos_2, ...]) across 56,676 vocabulary terms│
│ • AST Recursive Descent Boolean Parser (AND, OR, NOT, parenthetical nesting)            │
│ • Positional Exact Phrase Verification: pos_{i+1} = pos_i + 1                           │
│ • Deterministic Ranking Score: 1.0 * N_matched + 2.0 * I_phrase + 0.25 * log10(1 + TF)  │
│ • Multi-Term Evidence Snippet Generator with <mark> highlighting                        │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Full-Stack Web GUI & Benchmark Evaluation                                      │
│ • FastAPI REST API Server (Port 8000)                                                   │
│ • Glassmorphic Dark Dashboard UI (Live search, truth badges, claim dossier inspection)  │
│ • 15 Canonical TREC Benchmark Queries (MAP, MRR, P@K, R@K, nDCG@K)                     │
+-----------------------------------------------------------------------------------------+
```

---

## 2. Corpus Characterization & Ingestion

The system ingests the complete [finfact.json](file:///c:/Users/radhe/Downloads/NLP/Fin-Fact/finfact.json) dataset and decomposes each record into atomic, searchable content units with full hierarchical provenance:

$$\text{CLAIM ID} \longrightarrow \text{UNIT ID} \longrightarrow \text{UNIT TYPE} \longrightarrow \text{VERDICT LABEL} \longrightarrow \text{TEXT} \longrightarrow \text{PROVENANCE}$$

### Canonical Corpus Statistics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Fact-Check Claims** | `3,369` | Distinct financial assertions investigated by fact-checkers |
| **Total Extracted Content Units** | `27,617` | Atomic indexed units (claims, justifications, digests, evidence) |
| **Searchable Evidence Sentences** | `18,270` | Ground truth evidence sentences with external citations |
| **Claim Statement Units** | `3,369` | The central assertion to verify |
| **Justification Narrative Units** | `3,367` | In-depth fact-checking rationale |
| **Scientific / Summary Digests** | `2,611` | Concise executive summaries |
| **Total Words (Tokens)** | `3,959,856` | Raw token volume (~12.1× larger than Indian reports corpus) |
| **Total Characters** | `24,489,618` | Raw character count |
| **Vocabulary Size** | `56,676` | Unique non-stopword normalized lemmas |
| **Total Positional Postings** | `1,556,345` | Position entries across the corpus |

### Ground-Truth Label Distribution

| Label | Count | Percentage | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **`false`** | `1,492` | 44.3% | Claim debunked / factually inaccurate |
| **`true`** | `1,275` | 37.8% | Claim verified as factually supported |
| **`nei`** | `602` | 17.9% | *Not Enough Information* to conclusively verify |

---

## 3. NLP Preprocessing Methodology (Pipeline A)

Financial fact-checking text differs fundamentally from standard news corpora. Unregulated stemming and generic stopword removal introduce catastrophic failure modes:
1. Stripping `"not"` flips a claim from False to True.
2. Stripping currency symbols (`$`, `₹`) destroys the numerical ground truth being verified (e.g. `$11 billion` vs `$8.5 billion`).
3. Aggressive stemming collapses distinct financial concepts into ambiguous stems.

### Component Implementations

1. **Conservative Financial Cleaner** ([cleaner.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/cleaner.py)):
   * Strips web scraper promotional noise (e.g., *"Become a Founding Member to help us hire more fact-checkers..."*).
   * Repairs glued extraction artifacts (e.g., `interview,which` $\rightarrow$ `interview, which`).
   * Normalizes Unicode quotes (`\u201c`, `\u201d` $\rightarrow$ `"`) and em-dashes (`\u2014` $\rightarrow$ `—`).
   * Preserves currency symbols (`$`, `€`, `£`, `₹`), percentages (`%`, `bps`), and fiscal codes (`401(k)`).

2. **Custom Finance Tokenizer** ([tokenizer.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/tokenizer.py)):
   * Preserves currency-amount pairs as atomic units (`$11B`, `$8.5`, `₹5,000`).
   * Preserves fiscal year notations (`FY2025`, `Q3FY24`, `H1`) and retirement instruments (`401(k)`).
   * Preserves hyphenated economic expressions (`shareholder-capitalism`, `COVID-19`).
   * Tracks absolute character offsets and zero-indexed token sequence positions.

3. **Domain-Aware Stopwords with Polarity Protection** ([stopwords.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/stopwords.py)):
   * Filters grammatical noise while protecting **90+ high-value financial terms** (`tax`, `debt`, `bank`, `loss`, `profit`, `rate`, `pension`, `bond`, `fund`, `dividend`, `capital`, `inflation`, `recession`).
   * Enforces **Strict Polarity Protection**: Words like `not`, `no`, `never`, `without`, `against`, and `cannot` are **never stripped**, maintaining truth value integrity.

4. **Morphological Lemmatizer (Pipeline A)** ([lemmatizer.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/lemmatizer.py)):
   * Employs POS-guided inflection reduction and irregular exception dictionaries.
   * Accurately maps `investments` $\rightarrow$ `investment`, `liquidated` $\rightarrow$ `liquidate`, `reserves` $\rightarrow$ `reserve`, `currencies` $\rightarrow$ `currency`.
   * Operates 100% offline with zero network latency.

5. **Financial Entity Recognizer** ([entities.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/entities.py)):
   * Extracts domain entities: `MONEY` (`$11 billion`), `PERCENT` (`5.5%`), `DATE` (`Aug. 31, 2015`), `ORG` (`Amazon`, `Federal Reserve`), `PERSON` (`Joe Biden`, `Dan Patrick`).

---

## 4. Information Retrieval Engine

### Positional Inverted Index
Every content unit is indexed into a positional postings list:
$$\text{term} \longrightarrow \left[ \left(\text{unit\_id}_1, \text{tf}_1, [pos_1, pos_2, \dots]\right), \dots \right]$$

### Abstract Syntax Tree (AST) Boolean Parser
The recursive descent parser supports:
* Single terms: `capitalism`
* Exact quoted phrases: `"shareholder capitalism"`
* Boolean operators: `AND`, `OR`, `NOT`
* Nested expressions: `(biden OR trump) AND "tax cuts"`
* Strict syntax validation: Rejects unquoted standalone `&` with informative hints.

### Exact Phrase Distance Verification
To verify exact multi-token phrases `"w_1 w_2 ... w_k"`, candidate units must satisfy strict positional adjacency:
$$\forall i \in \{1, \dots, k-1\}: \quad pos_{i+1} = pos_i + 1$$

### Deterministic Retrieval Ranking Score
Retrieved content units are scored and ranked via the deterministic formula:
$$\text{Score} = 1.0 \cdot N_{\text{matched}} + 2.0 \cdot \mathbb{I}_{\text{phrase}} + 0.25 \cdot \log_{10}(1 + \text{TF}_{\text{matched}})$$
* $N_{\text{matched}}$: Count of distinct query positive terms matched.
* $\mathbb{I}_{\text{phrase}}$: Count of exact quoted phrases matched positionally.
* $\text{TF}_{\text{matched}}$: Aggregate term frequency of matched terms in the unit.

---

## 5. Benchmark Evaluation Results

The system was evaluated against **15 Canonical Financial Benchmark Queries** covering macroeconomics, taxation, retirement funds, monetary policy, and corporate governance:

| Query ID | Query Expression | Description | Total Matched | P@1 | P@5 | P@10 | MAP | MRR | nDCG@10 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q01** | `"shareholder capitalism"` | Corporate governance philosophy | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q02** | `biden AND pension` | Presidential policy on pensions | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q03** | `"tax cuts" AND reserve` | State reserves and tax legislation | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q04** | `deficit OR debt` | National debt & deficit reports | 775 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q05** | `"covid-19" AND testing` | Healthcare contracts & pandemic | 34 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q06** | `"rainy day fund"` | State emergency budget reserves | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q07** | `inflation AND spending` | Inflationary spending pressure | 162 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q08** | `"federal income tax"` | Income tax code obligations | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q09** | `stimulus AND "small business"` | Emergency relief programs | 34 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q10** | `medicare OR medicaid` | Entitlement expenditures | 258 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q11** | `"stock market" AND biden` | Equity market policy reactions | 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q12** | `unemployment AND pandemic` | Labor market economic shocks | 129 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q13** | `corporate AND taxes` | Corporate tax rate debates | 134 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q14** | `"interest rate" OR "federal reserve"` | Central banking and policy rates | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Q15** | `biden AND NOT "stock market"` | Non-equity Biden economic policy | 352 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Overall Macro Metrics
* **Mean Average Precision (MAP)**: **1.0000**
* **Mean Reciprocal Rank (MRR)**: **1.0000**
* **Mean Precision@5 (P@5)**: **1.0000**
* **Mean Precision@10 (P@10)**: **1.0000**
* **Mean nDCG@10**: **1.0000**
* **Mean Recall@10**: **0.9067**

---

## 6. Project Directory Layout

```
FinFact_Domain_Analysis/
├── config/
│   └── pipeline_config.yaml           # Central declarative YAML configuration
├── data/
│   ├── structured/
│   │   └── corpus.jsonl               # 27,617 structured content units
│   └── metadata/
│       ├── claim_registry.csv         # 3,369 claims with authors, dates, labels
│       ├── unit_registry.csv          # Granular unit tracking table
│       └── corpus_summary.json        # Ingestion metrics summary
├── src/
│   ├── ingestion/
│   │   └── ingest.py                  # Deterministic JSON parser and unit extractor
│   ├── preprocessing/
│   │   ├── cleaner.py                 # Conservative financial web cleaner
│   │   ├── tokenizer.py               # Custom finance tokenizer with positions
│   │   ├── stopwords.py               # Domain-aware stopwords with polarity protection
│   │   ├── lemmatizer.py              # Pipeline A rule and dictionary lemmatizer
│   │   ├── entities.py                # Financial entity recognizer (MONEY, ORG, etc.)
│   │   └── pipeline.py                # Preprocessing pipeline orchestrator
│   ├── indexing/
│   │   ├── inverted_index.py          # Positional Inverted Index structure
│   │   └── index_builder.py           # Multi-unit index constructor
│   ├── retrieval/
│   │   ├── query_parser.py            # AST recursive descent Boolean parser
│   │   ├── search_engine.py           # Positional phrase matcher and ranking engine
│   │   └── snippet_generator.py       # Multi-term evidence snippet generator
│   └── evaluation/
│       └── benchmark.py               # 15 Canonical Query TREC evaluation
├── backend/
│   └── main.py                        # FastAPI REST API application
├── frontend/
│   └── index.html                     # Glassmorphic Dark UI dashboard
├── results/
│   ├── index/
│   │   ├── positional_index.json      # Serialized positional inverted index
│   │   └── unit_store.json            # Content unit metadata store
│   └── evaluation/
│       ├── benchmark_report.json      # Benchmark report JSON
│       └── benchmark_table.csv        # Benchmark results CSV table
├── tests/
│   ├── test_cleaning.py               # Cleaner unit tests
│   ├── test_tokenization.py           # Tokenizer & lemmatizer unit tests
│   ├── test_query_parser.py           # AST query parser unit tests
│   └── test_search.py                 # Positional search unit tests
├── run_pipeline.py                    # Master CLI command orchestrator
└── README.md                          # Master documentation
```

---

## 7. Quick Start & Execution Guide

### 1. Run All Tests
```bash
python -m unittest discover -s tests
```
*Executes all 18 unit tests across cleaning, tokenization, lemmatization, AST parsing, and positional search.*

### 2. Run Full Pipeline (End-to-End)
```bash
python run_pipeline.py --all
```
*Performs dataset ingestion, content unit segmentation, Pipeline A preprocessing, positional index construction, and 15-query benchmark evaluation.*

### 3. Launch Web Application & REST API
```bash
python run_pipeline.py --serve
```
*Starts the FastAPI server at `http://127.0.0.1:8000` and serves the interactive Glassmorphic Dashboard.*

### 4. Interactive API Documentation
Open your browser to:
* **Interactive UI**: `http://127.0.0.1:8000/`
* **Swagger OpenAPI Docs**: `http://127.0.0.1:8000/docs`
* **API Health Check**: `http://127.0.0.1:8000/api/health`
* **API Corpus Stats**: `http://127.0.0.1:8000/api/stats`
* **API Search Query**: `http://127.0.0.1:8000/api/search?q="shareholder capitalism"`
* **API Claim Detail**: `http://127.0.0.1:8000/api/claims/CLM_0001`
