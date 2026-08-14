# HR Policy Assistant

A retrieval-augmented generation (RAG) chatbot that answers employee questions
grounded strictly in a company's own HR policy handbook (PDF). Built with
Google Gemini (embeddings + generation), ChromaDB as the local vector store,
and a hybrid dense + BM25 sparse retriever — evaluated with a golden Q&A set
and standard information-retrieval metrics (precision@k, recall@k, MRR).

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Sample questions & live demo](#sample-questions--live-demo)

## Features

- **Folder-based ingestion** — drop any number of PDF policy documents into
  `data/`; the loader scans the whole folder instead of a hardcoded filename.
- **Hybrid retrieval** — combines dense embedding search (semantic) with
  BM25 keyword search, fusing normalized scores per chunk.
- **Grounded generation** — the system prompt instructs the model to answer
  only from retrieved context and say so explicitly when it can't.
- **Source attribution** — every answer is returned with the source
  document(s) it was grounded in, shown in the Streamlit UI.
- **Idempotent indexing** — the vector store is only (re)built when empty,
  so restarting the app doesn't re-embed everything on every run.
- **Dependency-injected core** — `RagPipeline` accepts any retriever/generator
  with the right interface, so tests run against fakes with zero network
  calls or API cost.
- **Quantitative evaluation** — a golden Q&A set is scored for both answer
  correctness and retrieval quality (precision@k, recall@k, MRR), so changes
  to chunking or retrieval logic can be validated with numbers instead of
  vibes.
- **CI** — lint (`ruff`) and unit tests run automatically on every push/PR.

## Architecture

```
data/*.pdf  (any number of policy documents)
    │  pypdf extracts raw text per file
    ▼
document_loader.py  →  chunking.py
    │  fixed-size overlapping word chunks (chunk_size / chunk_overlap)
    ▼
embeddings.py  (Gemini gemini-embedding-001 — batched)
    │
    ▼
vector_store.py  (ChromaDB, persisted to disk under CHROMA_PERSIST_DIR)
    │
    ▼
retriever.py  (hybrid retrieval)
    │   dense: cosine similarity search over Chroma
    │   sparse: BM25Okapi keyword search over all chunks
    │   → per-chunk scores normalized to [0,1] and summed, re-ranked
    ▼
generator.py  (Gemini generation model, grounded system prompt)
    │   prompt = retrieved chunks (with source) + question
    │   instructed to refuse if the answer isn't in the context
    ▼
pipeline.py  (RagPipeline.answer() — single entry point)
    │   ties retriever + generator together; returns {answer, sources}
    ▼
app.py  (Streamlit chat UI)
    displays the answer and an expandable list of source documents
```

Indexing is idempotent: on startup, the app scans `data/` for PDFs and only 
embeds documents if the vector store is empty. Dropping a new PDF into
`data/` and restarting the app is enough for it to be picked up on the next
build (delete the `CHROMA_PERSIST_DIR` folder to force a full rebuild).

## Project structure

```
src/hr_rag/
  config.py                 # env-driven settings (Settings class, loaded once as `settings`)
  ingestion/
    document_loader.py      # scans data/ for *.pdf, extracts text with pypdf
  rag/
    chunking.py              # splits text into overlapping word-count chunks
    embeddings.py             # Gemini embedding client (batched embed_documents / embed_query)
    vector_store.py            # Chroma persistence, add_chunks / dense query
    retriever.py                 # HybridRetriever: dense + BM25 fused ranking
    generator.py                  # grounded prompt construction + Gemini generation
    pipeline.py                    # RagPipeline: retriever + generator → answer()
app.py                                # Streamlit chat UI
requirements.txt                       # runtime dependencies
pyproject.toml                          # packaging (src layout, editable install), ruff/pytest config
.env.example                             # documented environment variables
tests/                                    # unit tests — fakes only, no network calls
  test_chunking.py
  test_pipeline.py
eval/                                      # quality evaluation (requires a real API key)
  golden_qa.jsonl                           # hand-written question/keyword/source ground truth
  evaluate.py                                # runs the real pipeline, reports answer + retrieval metrics
data/                                         # your HR policy PDF(s) go here (gitignored contents)
.github/workflows/ci.yml                       # lint + test on every push/PR
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env             # then edit .env and add your GEMINI_API_KEY
```

Drop one or more HR policy PDFs into `data/`.

## Usage

```bash
streamlit run app.py
```

Or you can try live demo [here.](https://hr-rag-chatbot.streamlit.app)

## Configuration

All tunables live in `.env` (documented in `.env.example`):

| Variable                  | Default                  | Purpose                                                   |
| -------------------------- | ------------------------- | ---------------------------------------------------------- |
| `GEMINI_API_KEY`            | —                          | Required. Your Gemini API key.                              |
| `GEMINI_GENERATION_MODEL`    | `gemini-3.5-flash-lite`     | Model used for answer generation.                            |
| `GEMINI_EMBEDDING_MODEL`      | `gemini-embedding-001`       | Model used for chunk/query embeddings.                        |
| `CHROMA_PERSIST_DIR`           | `.chroma`                     | Where the vector store is persisted on disk.                   |
| `DOCS_DIR`                       | `data`                          | Folder scanned for `*.pdf` documents.                            |
| `CHUNK_SIZE`                      | `800`                             | Words per chunk.                                                  |
| `CHUNK_OVERLAP`                    | `120`                               | Words shared between consecutive chunks.                            |
| `TOP_K`                              | `5`                                   | Number of chunks retrieved per question.                              |

`chunk_size`/`chunk_overlap` defaults are a reasonable starting point, not a
tuned optimum — see [Evaluation](#evaluation) for how to validate them
against real data instead of guessing.

## Testing

```bash
pytest -v
```

Unit tests use fake retriever/generator/pipeline objects (constructor-based
dependency injection) — no network calls, no API key required, runs in
milliseconds. This is what runs in CI on every push/PR (`.github/workflows/ci.yml`),
alongside `ruff check .` for linting.

## Evaluation

```bash
python eval/evaluate.py
```

This is a **manual quality gate**, not a CI step — it calls the real Gemini
API against the real vector store, so it requires a valid `GEMINI_API_KEY`
and costs a small amount of API usage. Run it before shipping changes to
chunking, prompts, or retrieval logic.

`eval/golden_qa.jsonl` holds a hand-written set of questions, each with:
- `expected_keywords` — terms that should appear in a correct answer (used
  both to score the generated answer and, as a cheap proxy, to judge whether
  a retrieved chunk is relevant)
- `expected_source` — which document the answer should come from

For each question, the script reports:

**Answer quality** — does the generated answer contain all expected
keywords? Reported as `passed/total` and a pass rate, with a `PASS`/`FAIL`
line printed per question so failures are easy to spot.

**Retrieval quality**, computed independently of generation (so retrieval
and generation failures can be told apart):
- **Precision@k** — of the top-k retrieved chunks, what fraction are
  relevant (contain an expected keyword)?
- **Recall@k** — was at least one relevant chunk retrieved in the top-k?
- **MRR (Mean Reciprocal Rank)** — how high up did the first relevant chunk
  rank, averaged across all questions (`1.0` = always first).

Latest run against the 12-question golden set (`TOP_K=5`, single PDF):

| Metric        | Score |
| ------------- | ----- |
| Answer score  | 11/12 (92%) |
| Precision@5   | 0.30  |
| Recall@5      | 1.00  |
| MRR           | 0.96  |

Recall and MRR being near-perfect indicate the retriever almost always
surfaces the right chunk, and almost always ranks it first. Precision@5
looks low in isolation, but with a single source document there is
typically only one truly relevant chunk per question — against `top_k=5`
that caps precision@5 at `0.20` even for a perfect retriever, so `0.30` is
consistent with strong retrieval, not a sign of noise. Precision becomes a
more informative signal once the corpus has more documents and more
genuinely relevant chunks per question to find.

## Sample questions & live demo

Questions tried against the app during development, and how the system
responded. Each answer came with a "Sources" section listing the exact
handbook chunk(s) it was grounded in.

| Question | Answer summary | Notes |
| --- | --- | --- |
| "My cat got sick, can I take 5 days off?" | Explains pet care leave is 2 paid days/month (or convertible to remote days), so 5 consecutive pet care days isn't possible under the policy. | Correctly applies the specific policy limit instead of just confirming leave exists. |
| "I just got married, how many days of leave can I take?" | States that marriage leave is not mentioned anywhere in the handbook, and lists the leave categories that *are* covered (annual, sick, public holidays, menstrual, paternity, birthday, pet care) instead of guessing. | The key grounding test: the model refuses to invent a policy that doesn't exist rather than producing a plausible-sounding but false answer. |
| "How many days of annual leave do employees get?" | 20 days per year, with the notice-period and carry-over rules from the handbook. | Straightforward factual retrieval; part of the golden eval set. |
| "How long is paternity leave and how much is paid?" | 1 year, job-protected; first 8 weeks paid at 100%, remainder unpaid with a return guarantee. | Answer required combining two adjacent facts from the same policy clause. |

The second example is the most important one architecturally: it's direct
evidence that the `SYSTEM_INSTRUCTION` grounding constraint in
`generator.py` works as intended — the model would rather say "this isn't
in the documents" than fabricate a marriage-leave policy — refusing to
guess is treated as more valuable than always sounding confident, which is
the whole point of grounding answers in a single authoritative document
instead of the model's general knowledge.
