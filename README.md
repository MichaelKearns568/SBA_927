# SBA 927: Business Text Analytics — Natural Language Processing Techniques

---

## Project Overview

This project applies foundational Natural Language Processing (NLP) techniques using Python and the `nltk` library to analyze business text data (`SBA927.txt`). Designed to simulate a real-world data analyst workflow, the script processes unstructured text to extract actionable insights through text normalization, morphological root extraction, grammatical tagging, and entity classification. Environment setup and dependency management are handled using modern `uv` workflows.

---

## Key Concepts

* **Tokenization & Preprocessing:** Breaking continuous text strings into individual words or sentences (`word_tokenize`, `sent_tokenize`), normalizing casing, and filtering out punctuation, non-alphabetic characters, and common English stop words (`stopwords`) to reduce noise.
* **Stemming (`PorterStemmer`):** A heuristic rule-based process that strips word suffixes to produce a rough root form (the stem) for fast normalization.
* **Lemmatization (`WordNetLemmatizer`):** A morphological and vocabulary-based analysis that uses WordNet dictionary definitions to reduce words to their valid base dictionary form (the lemma).
* **Named Entity Recognition (NER):** An information extraction pipeline (`ne_chunk`) that locates and classifies proper nouns in text—such as organizations, people, and geopolitical entities (GPE)—into predefined categories.
* **Part-of-Speech (POS) Tagging:** The grammatical labeling process (`pos_tag`) that assigns each token its corresponding syntactic role (e.g., noun, verb, adjective) based on sentence context.

---

## Project Structure

```text
SBA_927/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── SBA927.txt
└── src/
    └── sba_927/
        ├── __init__.py
        └── app.py
```

---

## Setup and Execution Instructions

1. **Initialize the `uv` project and add dependencies:**
   If you are setting up the repository from scratch, initialize the project and install `nltk`:

   ```bash
   uv init
   uv add nltk
   ```

2. **Place the dataset in the project root:**
   Ensure `SBA927.txt` is located at the root directory (`../../SBA927.txt` relative to `src/sba_927/app.py`).

3. **Sync the environment and run the script using `uv`:**

   ```bash
   uv sync
   uv run python src/sba_927/app.py
   ```

   > **Note:** On first execution, `app.py` automatically downloads the required NLTK corpora and taggers (`punkt`, `stopwords`, `averaged_perceptron_tagger`, `maxent_ne_chunker`, and `words`). Ensure `wordnet` is also available when running `WordNetLemmatizer` in a fresh environment.

---

## Summary of Findings & Learnings

* **Text Cleaning & Normalization:** Implemented a custom `tokenize_and_preprocess` pipeline combining NLTK's `word_tokenize` with alphabetical, stop-word, and punctuation filtering to isolate high-signal business terms.
* **Stemming vs. Lemmatization:** Compared `PorterStemmer` and `WordNetLemmatizer` side-by-side across preprocessed tokens, demonstrating how stemming prioritizes speed while lemmatization preserves human-readable dictionary roots.
* **Syntactic & Entity Extraction:** Chained sentence tokenization, `pos_tag`, and `ne_chunk` tree traversal to extract structured `(entity, label)` tuples and grammatical tags directly from raw business records.
* **Environment Reproducibility:** Managed Python runtime versions and `nltk` dependencies cleanly via `uv` and `pyproject.toml`.