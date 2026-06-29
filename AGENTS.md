# AGENTS.md — legal-rag-vietnam

Vietnamese legal RAG pipeline for a competition (2000 questions, F2-scored). Runs on Kaggle GPU.

## Entry point & commands

- **Full pipeline**: `python main.py --input R2AIStage1DATA.json [--resume] [--batch-size 50] [--debug]`
- **Debug pipeline** (per-stage diagnostics): `python debug_pipeline.py --input data/R2AIStage1DATA_50.json --ground-truth data/ground_truth_50.json --num-questions 50 --output-dir debug_output`
- **Normalize manifest**: `python normalize_manifest.py [--dry-run]`
- **Build BM25 standalone**: `python src/index_bm25.py`

## Architecture

Pipeline stages: **Hybrid Retrieval (BM25 + Qdrant dense + RRF)** → **Reranking (BGE Cross-Encoder)** → **LLM Generation (Qwen2.5-7B-Instruct, 4-bit)** → **Self-Verification (5 rules)** → **Post-Processing + submission packaging**

```mermaid
flowchart LR
  A[Question] --> B[HybridRetriever]
  B --> C[LegalReranker]
  C --> D[AnswerGenerator]
  D --> E[SelfVerifier]
  E -- fail --> D
  E -- pass --> F[PostProcessor]
  F --> G[results.json + submission.zip]
```

## Key gotchas

- **Qdrant Cloud only** (no local Qdrant). Requires `.env` with `QDRANT_URL` and `QDRANT_API_KEY`.
- **LLM is Qwen2.5-7B-Instruct** despite comments mentioning "DeepSeek-R1-14B". 4-bit quantized (bitsandbytes nf4), `device_map="auto"`.
- **F2 optimization**: `relevant_docs`/`relevant_articles` selected from **top-2 reranked contexts** (precision-first), NOT from LLM citations. Generator output takes priority; PostProcessor only used as fallback when generator returns empty.
- **Self-Verification rules**: RULE1 (articles in context → hard fail), RULE2 (doc numbers in manifest → hard fail), RULE3 (doc name consistency → warning only), RULE4 (out-of-context claims → warning only), RULE5 (must cite "Điều X" → hard fail). On fail, regenerates with lower temperature (`0.05`), max 1 retry.
- **Codebase is in Vietnamese** (comments, variable names, strings, logging).
- **BM25 tokenization** uses `underthesea.word_tokenize` + domain stopwords + synonym map for legal abbreviations (tncn→thu nhập cá nhân, etc.).

## Config

All settings in `config/settings.py`. Key tunables:
- `RERANKER_THRESHOLD = 0.30` — min score to keep a doc after reranking
- `TOP_K_FINAL = 6` — docs fed to LLM
- `RELEVANT_ARTICLES_MAX = 2`, `RELEVANT_DOCS_MAX = 2` — fields emitted in submission
- `LLM_MAX_NEW_TOKENS = 1024` (actual max is `max(2048, this)` → 2048)

## Data files

- `data/corpus_clean.json` — legal corpus (needed for BM25, exported from Qdrant via `export_corpus.py`)
- `data/law_manifest.json` — doc number → canonical name mapping (BTC format)
- `data/R2AIStage1DATA.json` — 2000 questions (competition input)
- `data/R2AIStage1DATA_50.json` / `data/ground_truth_50.json` — 50-question debug subset with ground truth
- `data/bm25_corpus.pkl` — serialized BM25 index (auto-created, gitignored)

## Output structure

- `output/checkpoint.json` — incremental save every batch
- `output/results.json` — final submission JSON
- `output/submission_<timestamp>.zip` — submission archive
- `logs/pipeline.log` — full run log
- `logs/evaluation_report.md` — auto-generated metrics report
- `logs/detailed_log.json` — per-question breakdown
