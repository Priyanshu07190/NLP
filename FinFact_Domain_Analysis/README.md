# Fin-Fact Domain-Specific NLP & Information Retrieval System
## Financial Fact-Checking, Claim Verification & Comparative Evaluation

**Course:** Natural Language Processing (Financial Domain Text Retrieval)  
**Dataset:** Fin-Fact Multimodal Fact-Checking Corpus (`finfact.json`, 3,369 claims)  
**Architectures Implemented:**
- **Pipeline A (Baseline - Without POS):** Heuristic Morphological Normalization + Positional Postings Indexing
- **Pipeline A+ (With POS Tagging - Active Deployed Engine):** Penn Treebank POS Tagging + WordNet POS-Guided Lemmatization + Noun Phrase Extraction + Positional Indexing

---

## 1. Executive Summary

This project implements an end-to-end, production-grade **Domain-Specific Natural Language Processing (NLP) and Information Retrieval (IR) System** tailored to the **Fin-Fact** financial fact-checking dataset.

Per academic recommendations, the system evaluates and contrasts two architectures:
1. **Pipeline A (Baseline - Without POS):** Operates purely on heuristic morphological rules and dictionary exceptions.
2. **Pipeline A+ (With POS Tagging):** Employs **Penn Treebank POS Tagging** to guide **WordNet lemmatization**, disambiguating parts of speech (e.g. action verbs vs. financial nouns) and extracting domain noun phrases (*"corporate taxes"*, *"rainy day fund"*).

Both pipelines index **27,617 content units** derived from **3,369 financial claims**. The web dashboard automatically serves **Pipeline A+ (With POS Tagging)** by default and provides an interactive side-by-side comparison tab and live syntactic POS analyzer.

```
+-----------------------------------------------------------------------------------------+
|                               FIN-FACT COMPARATIVE IR WORKFLOW                          |
+-----------------------------------------------------------------------------------------+
                                           │
[Raw Dataset: finfact.json] (3,369 claims, 30.17 MB)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Ingestion & Unit Segmentation                                                  │
│ Output: 27,617 Content Units (Claims, Justifications, Digests, Evidence Sentences)      │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│ Pipeline A (Baseline - Without POS)   │ │ Pipeline A+ (Active - With POS Tagging)       │
│ • Conservative Cleaner ($, %, bps)    │ │ • Conservative Cleaner ($, %, bps)            │
│ • Custom Finance Tokenizer            │ │ • Custom Finance Tokenizer                    │
│ • Polarity-Preserving Stopwords       │ │ • Polarity-Preserving Stopwords               │
│ • Heuristic Suffix Rules (No POS)     │ │ • Penn Treebank POS Tagging (Averaged Percep.)│
│                                       │ │ • WordNet POS-Guided Lemmatization            │
│                                       │ │ • Multi-Word Noun Phrase Extraction           │
└───────────────────┬───────────────────┘ └───────────────────────┬───────────────────────┘
                    │                                             │
                    ▼                                             ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│ Baseline Positional Index             │ │ POS-Enriched Positional Index (Active Default)│
│ • Vocab: 56,676 terms                 │ │ • Vocab: 56,206 terms (Eliminates truncation) │
│ • Postings: 1,556,345                 │ │ • Postings: 1,548,748 (Clean consolidated)    │
└───────────────────┬───────────────────┘ └───────────────────────┬───────────────────────┘
                    │                                             │
                    └────────────────────┬────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Comparative Evaluation & Interactive Web Dashboard (FastAPI :8000)                      │
│ • 15 Canonical TREC Benchmark Queries evaluated across both pipelines                   │
│ • Live Interactive Syntactic Sandbox (Token POS tags, lemmas, and noun phrase extraction│
│ • Full Dossier Inspection with token-level POS tag previews                             │
+-----------------------------------------------------------------------------------------+
```

---

## 2. Comparative Evaluation: Without POS vs. With POS Tagging

### A. Linguistic & Morphological Impact
Without POS tagging, heuristic lemmatizers and stemmers make systematic errors on financial verbs and nouns because they assume all words are nouns by default or aggressively slice suffixes. Adding POS tagging resolves these errors:

| Financial Term | Syntactic POS Tag | Without POS (Baseline) | With POS (Pipeline A+) | Academic / Linguistic Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **`increased`** | `VBD` (Past Verb) | `increas` *(stem truncation)* | `increase` *(valid WordNet verb)* | Eliminates root distortion on predicate action verbs |
| **`reserves`** | `NNS` (Plural Noun) | `reserve` *(heuristic)* | `reserve` *(Noun confirmed)* | Protects bank capital reserves from verbal third-person conflation |
| **`shares`** | `NNS` (Plural Noun) | `share` *(heuristic)* | `share` *(Equity confirmed)* | Preserves equity securities semantics vs. verbal distribution |
| **`liquidating`** | `VBG` (Participle Verb) | `liquidate` *(suffix rule)* | `liquidate` *(Verb confirmed)* | Prevents aggressive stem collapse on bankruptcy/liquidity terms |
| **`spending`** | `NN` / `VBG` (Noun/Gerund) | `spend` *(verb strip)* | `spending` / `spend` | Disambiguates economic expenditure (substantive noun) from verb |

### B. Index & Vocabulary Comparison Across 27,617 Units

| Metric | Without POS (Baseline) | With POS (Pipeline A+) | Delta ($\Delta$) | Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Searchable Content Units** | `27,617` | `27,617` | `0` | Identical evaluation base |
| **Total Corpus Tokens** | `3,959,856` | `4,039,547` | `+79,691` | Full token offsets + POS tracking |
| **Vocabulary Size** | `56,676` | `56,206` | **`-470`** | **Consolidates truncated pseudo-stems (`increas` merged into `increase`)** |
| **Total Postings** | `1,556,345` | `1,548,748` | **`-7,597`** | Cleaner, more compact, consolidated index |
| **Noun Phrases Extracted** | `0` (None) | **`18,412`** | `+18,412` | Automatic financial key concept extraction |

### C. 15 Canonical Financial Benchmark Query Results

Both engines were evaluated against the 15 canonical financial domain benchmark queries ([pos_comparison_report.json](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/results/evaluation/pos_comparison_report.json)):

| Benchmark Metric | Without POS (Baseline) | With POS (Pipeline A+) | Delta ($\Delta$) | Operational Result |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Average Precision (MAP)** | **1.0000** | **1.0000** | `0.0000` | Tie (Perfect precision maintained) |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | **1.0000** | `0.0000` | Tie (Rank-1 hit on every query) |
| **Precision @ 1 (P@1)** | **1.0000** | **1.0000** | `0.0000` | Tie (100% relevant at rank 1) |
| **Precision @ 5 (P@5)** | **1.0000** | **1.0000** | `0.0000` | Tie (100% relevant in top 5) |
| **Precision @ 10 (P@10)** | **1.0000** | **1.0000** | `0.0000` | Tie (100% relevant in top 10) |
| **Recall @ 10 (R@10)** | **0.9067** | **0.9067** | `0.0000` | Tie (Captures over 90% of pool) |
| **nDCG @ 10** | **1.0000** | **1.0000** | `0.0000` | Tie (Ideal ranking order) |

**Conclusion for Academic Assessment:**
* **Retrieval Quality:** Both pipelines achieve top-tier precision because the AST parser and positional distance verification strictly enforce phrase constraints.
* **Linguistic Representation:** **Pipeline A+ (With POS Tagging) is distinctly superior** because it eliminates stem truncations (`increas` $\rightarrow$ `increase`), merges spurious vocabulary fragments (-470 terms), and enables automated Noun Phrase concept extraction (*"rainy day fund"*, *"corporate taxes"*).

---

## 3. Corpus Characterization & Ingestion

The system ingests the complete [finfact.json](file:///c:/Users/radhe/Downloads/NLP/Fin-Fact/finfact.json) dataset and decomposes each record into atomic, searchable content units with unbroken provenance:

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
| **Vocabulary Size (With POS)** | `56,206` | Normalized WordNet lemmas |
| **Total Positional Postings** | `1,548,748` | Positional entries across the corpus |

### Ground-Truth Label Distribution

| Label | Count | Percentage | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **`false`** | `1,492` | 44.3% | Claim debunked / factually inaccurate |
| **`true`** | `1,275` | 37.8% | Claim verified as factually supported |
| **`nei`** | `602` | 17.9% | *Not Enough Information* to conclusively verify |

---

## 4. NLP Preprocessing Methodology

### Component Implementations

1. **Conservative Financial Cleaner** ([cleaner.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/cleaner.py)):
   * Strips web scraper promotional noise (e.g., *"Become a Founding Member to help us hire more fact-checkers..."*).
   * Repairs glued extraction artifacts (e.g., `interview,which` $\rightarrow$ `interview, which`).
   * Preserves currency symbols (`$`, `€`, `£`, `₹`), percentages (`%`, `bps`), and fiscal codes (`401(k)`).

2. **Custom Finance Tokenizer** ([tokenizer.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/tokenizer.py)):
   * Preserves currency-amount pairs (`$11B`, `$8.5`), fiscal year codes (`FY2025`, `Q3FY24`), and hyphenated expressions (`shareholder-capitalism`, `COVID-19`).
   * Tracks absolute character offsets and zero-indexed token sequence positions.

3. **Domain-Aware Stopwords with Polarity Protection** ([stopwords.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/stopwords.py)):
   * Protects **90+ high-value financial terms** (`tax`, `debt`, `bank`, `loss`, `profit`, `rate`, `pension`, `bond`, `fund`, `dividend`, `capital`, `inflation`, `recession`).
   * Enforces **Strict Polarity Protection**: Words like `not`, `no`, `never`, `without`, `against`, and `cannot` are **never stripped**, maintaining truth value integrity.

4. **Financial POS Tagger & Syntactic Analyzer** ([pos_tagger.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/pos_tagger.py)):
   * Penn Treebank POS tagger using Averaged Perceptron with domain token overrides (`401(k)`, `COVID-19`, currencies).
   * Automatically extracts multi-word Noun Phrases (`[JJ]* [NN]+`).

5. **Morphological Lemmatizer (Pipeline A & A+)** ([lemmatizer.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/lemmatizer.py)):
   * **Baseline Mode (Without POS):** Heuristic suffix rules and irregular dictionaries.
   * **Enhanced Mode (With POS):** POS-guided WordNet lemmatizer passing disambiguated grammatical categories (`noun`, `verb`, `adjective`, `adverb`).

6. **Financial Entity Recognizer** ([entities.py](file:///c:/Users/radhe/Downloads/NLP/FinFact_Domain_Analysis/src/preprocessing/entities.py)):
   * Extracts domain entities: `MONEY` (`$11 billion`), `PERCENT` (`5.5%`), `DATE` (`Aug. 31, 2015`), `ORG` (`Amazon`, `Federal Reserve`), `PERSON` (`Joe Biden`, `Dan Patrick`).

---

## 5. Execution Guide & CLI Commands

### 1. Run Unit Tests (22 Tests Passing)
```bash
python -m unittest discover -s tests
```
*Validates cleaning, tokenization, POS tagging, noun phrase extraction, AST parsing, and positional search.*

### 2. Run Comparative Evaluation (Without POS vs With POS)
```bash
python run_pipeline.py --compare
```
*Executes the 15 canonical queries on both engines, generates `pos_comparison_report.json`, and outputs `pos_comparison_table.csv`.*

### 3. Launch Web Application (Serves POS-Guided Engine)
```bash
python run_pipeline.py --serve
```
*Starts the FastAPI server at `http://127.0.0.1:8000` with the Glassmorphic Dark Dashboard.*

### 4. Interactive Browser Endpoints
* **Web GUI Dashboard:** `http://127.0.0.1:8000/`
* **Swagger OpenAPI Docs:** `http://127.0.0.1:8000/docs`
* **Comparative Report API:** `http://127.0.0.1:8000/api/comparison`
* **Benchmark Report API:** `http://127.0.0.1:8000/api/benchmark`
* **Live POS Analysis API:** `POST http://127.0.0.1:8000/api/analyze`
