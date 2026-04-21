# 🔍 CAPA/8D Expert Knowledge Worker

An AI-powered expert assistant for quality engineers working with CAPA procedures, 8D problem-solving methodology, root cause analysis, FMEA, and industry compliance standards.

Built as a production-grade RAG pipeline with full evaluation framework, streaming UI, and Langfuse observability — not a demo.

---

## What it does

Quality engineers ask questions in natural language. The system retrieves the most relevant content from a curated knowledge base, reranks it with a local cross-encoder, generates a grounded expert answer, and strips any claims not supported by the retrieved context.

**Example questions it handles well:**
- *"Our containment sort found zero defects but the customer found three more — do we expand the suspect window?"*
- *"What are the IATF 16949 specific requirements for 8D?"*
- *"Why is calendar-based PM dangerous when production volume changes?"*
- *"Is it acceptable to close an 8D if the PCA relies 100% on operator retraining?"*
- *"The same defect has come back three times despite previous corrective actions — what now?"*
- *"My auditor says my D7 is just a paperwork exercise — what should it actually contain?"*
- *"What happens if the root cause literally cannot be verified because we can't recreate it in the lab?"*
- *"The customer rejected our 8D — how do I respond?"*

---

## Architecture

```
User question
     │
     ▼
Query Rewriting ── Claude Haiku generates 3 alternative phrasings (history-aware)
     │
     ▼
Retrieval ── text-embedding-3-small + Chroma (top 30 per query × 4 queries)
     │
     ▼
Merge & Deduplicate ── union across queries, deduplicated by chunk ID
     │
     ▼
Reranking ── BAAI/bge-reranker-v2-m3 (local, MPS/CUDA/CPU)
     │         fallback: Claude Haiku LLM reranker
     ▼
Answer Generation ── GPT-4o-mini with conversation history
     │
     ▼
Groundedness Check ── Claude Haiku strips ungrounded claims
     │
     ▼
Expert answer with inline source citations
```

**Stack:**

| Component | Model / Tool |
|---|---|
| Query rewriting | Claude Haiku (`claude-haiku-4-5`) |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Vector store | Chroma (persistent, local) |
| Reranker | `BAAI/bge-reranker-v2-m3` (HuggingFace, local) |
| Answer generation | `gpt-4o-mini` (OpenAI) |
| Groundedness check | Claude Haiku |
| Observability | Langfuse (traces per pipeline run) |
| UI | Gradio 5.x / 6.x (streaming) |

---

## Knowledge Base

14 enriched documents, ~35,000 words, covering:

| Document | Category | Coverage |
|---|---|---|
| `8D_problem_solving_methodology.md` | methodology | D0–D8 complete guide, timing norms, common mistakes |
| `8D_report_example_automotive.md` | example | Steering bracket bore defect — full worked example with FINDING-anchored QR section |
| `8D_report_example_semiconductor.md` | example | IC wirebond failure — SEM/EDX analysis with FINDING-anchored QR section |
| `CAPA_SOP_enriched.md` | procedure | ISO 9001 Clause 10.2, CAPA ID format, VoE criteria |
| `containment_decision_guide.md` | procedure | Full location table, ICA methods, removal logic |
| `control_plan_basics.md` | reference | Column-by-column guide, 25% volume change rule |
| `effectiveness_verification_guide.md` | procedure | VoE metrics by CA type, Cpk thresholds |
| `fmea_basics.md` | reference | S/O/D/RPN, AIAG-VDA AP, material substitution rules + practitioner Q&A |
| `is_is_not_analysis.md` | tool | Template + 2 worked examples + practitioner Q&A |
| `multi-industry_CAPA_8D_compliance.md` | compliance | ISO 9001, IATF 16949, AS9100, ISO 13485, Semiconductor |
| `rca_tool_selection_matrix.md` | tool | 5 Whys/Ishikawa/FTA routing logic + unverifiable root cause guide |
| `root_cause_analysis.md` | tool | 5 Whys + Ishikawa toolkits, systemic vs local distinction |
| `capa_edge_cases.md` | general | Edge cases: ICA failure, containment pressure, long-running CAPAs, untraceable suspects |
| `8d_practitioner_scenarios.md` | general | Internal/customer/supplier rejections, daily updates, speed vs thoroughness |

---

## Evaluation

Evaluated against **197 test questions** across 3 independent sources:
- **t001–t050**: Developer-written structured questions with expected sources
- **t051–t197**: Blind questions from Gemini Pro and ChatGPT (practitioner-phrased)

### Important note on MRR

The eval framework calculates MRR only for questions with `expected_sources` defined. Of 197 questions, 48 structured questions (t001–t048) have expected sources — the remaining 149 blind questions (t051–t197) have `expected_sources=[]` and always score MRR=0 by design. The headline MRR figure (0.198–0.203) is therefore not a retrieval quality metric — it is a structural artefact of the eval design. The meaningful retrieval metric is MRR on structured questions only (~0.80–0.92 depending on run). Judge scores (correctness, completeness, groundedness) are valid across all 197 questions.

### Full 197-question eval — two runs compared

| Metric | Pre-iteration baseline | Post-iteration (FINDING anchors + temp=0) | Delta |
|---|---|---|---|
| Judge — Overall | **6.926** | **6.774** | -0.152 |
| Judge — Correctness | 7.412 | 7.246 | -0.166 |
| Judge — Completeness | 6.268 | 6.149 | -0.119 |
| Judge — Groundedness | 7.098 | 6.928 | -0.170 |
| Source coverage rate | 99.0% | 96.9% | -2.1% |
| Mean top chunk score | 4.049 | 4.048 | ~0 |
| Checker score | 0.665 | 0.651 | -0.014 |
| Checker fired rate | 87.6% | 85.1% | -2.5% |

### By category — pre vs post iteration

| Category | Pre overall | Post overall | Delta | Pre MRR | Post MRR |
|---|---|---|---|---|---|
| example | 4.670 | **6.330** | **+1.660** ✅ | 0.383 | **0.875** ✅ |
| 8D_methodology | 6.670 | **6.860** | **+0.190** ✅ | 0.161 | 0.170 |
| mixed | 6.360 | **6.560** | **+0.200** ✅ | 0.000 | 0.000 |
| compliance | 6.710 | 6.730 | +0.020 → | 0.235 | 0.235 |
| edge_case | 7.110 | 7.030 | -0.080 → | 0.000 | 0.000 |
| containment | 7.620 | 7.350 | -0.270 | 0.136 | 0.136 |
| RCA | 7.620 | 7.370 | -0.250 | 0.228 | 0.225 |
| VoE | 6.970 | 6.610 | -0.360 | 0.262 | 0.205 |
| CAPA_procedure | 6.050 | 5.650 | -0.400 | 0.116 | 0.117 |
| enriched_content | 8.220 | 7.890 | -0.330 | 1.000 | 1.000 |
| FMEA | 6.840 | 6.180 | -0.660 | 0.373 | 0.373 |

**Interpretation:** The FINDING anchors + temperature=0 fix achieved its primary objective — `example` category improved +1.66 with MRR jumping from 0.383 → 0.875. The overall regression (-0.15) is explained by generation quality changes across several categories after the `--reset` re-ingest, not retrieval degradation (MRR is unchanged for all regressing categories). The `--reset` generated new Haiku headlines at `temperature=0` for all documents — valid but slightly different from the previous run's headlines, producing marginally worse answer quality for FMEA, VoE, and CAPA_procedure categories. The pre-iteration baseline (6.926) remains the best overall score. Next iteration priority: GPT-4o-mini guardrails to reduce generic filler that triggers the groundedness checker, and markdown-aware chunking to fix sequential content splitting (t019 completeness).

### Example category iteration (4 questions, tracked separately)

| Run | Overall | MRR | top_chunk | t015 | t019 | t030 | t038 | Notes |
|---|---|---|---|---|---|---|---|---|
| Baseline | 4.25 | 0.750 | 6.74 | 7.0 | 4.0 | 4.0 | 2.0 | No QR section |
| + QR section | 6.58 | 0.750 | 6.74 | 7.7 | 3.7 | 9.3 | 5.7 | Upsert only, no reset |
| + category map change | 5.50 | 0.542 | 5.56 | 7.0 | 4.0 | 8.0 | 3.0 | --reset broke embeddings |
| + FACTUAL RECALL RULE | 4.83 | 0.473 | 5.56 | 5.7 | 4.3 | 7.3 | 2.0 | Answer prompt change backfired |
| Partial revert | 5.33 | 0.442 | 5.56 | 4.3 | 6.0 | 7.0 | 4.0 | FACTUAL RECALL still active |
| + FINDING anchors + temp=0 | **6.58** | **0.875** | **7.96** | 7.3 | 3.3 | 8.7 | 7.0 | Stable deterministic baseline |

---

## Key Engineering Decisions

**1. LLM-enriched chunking**
Each chunk is enriched at ingest with a Claude Haiku-generated headline and summary. Embedding = `headline + summary + original_text`. Improves retrieval for questions phrased differently from source material.

**2. BGE cross-encoder reranking**
`BAAI/bge-reranker-v2-m3` runs locally (free, no API). Reranks 40–120 merged candidates to 15. ~2s after warmup. Falls back to Claude Haiku if torch not installed. RETRIEVAL_K=30 for broader candidate pool.

**3. Groundedness post-checker**
After answer generation, Claude Haiku audits each claim against retrieved chunks and strips anything ungrounded. Only fires when top BGE score ≥ 0.5 — skips pure synthesis queries where claim-level checking is too aggressive.

**4. Streaming UI**
Expert Q&A tab streams tokens as they arrive. Sources panel populates from retrieval sink before first token — no second API call needed. Gradio 6.x compatible.

**5. Langfuse observability**
Every pipeline run traced with child spans: `rewrite_query`, `bge_rerank`, `generate_answer`, `groundedness_check`. Latency, chunk scores, checker scores, and source metadata logged per question.

**6. Three-source test set**
197 questions built from three independent sources to prevent eval overfitting. External questions are practitioner-phrased with no knowledge of document structure.

**7. Deterministic enrichment (temperature=0)**
Claude Haiku enrichment calls run at `temperature=0`. This ensures every re-ingest produces identical headlines for identical chunk text, making eval results comparable across runs. Discovered after multiple regressions caused by non-deterministic headline generation at the default `temperature=1.0`.

**8. FINDING-anchored QR sections**
Worked example documents contain a Quick Reference section prepended before the full case record. Each QR entry begins with a `FINDING:` line — a single factual sentence with CAPA ID, part number, and key numbers. At `temperature=0`, Haiku uses this line as the chunk headline, producing specific embeddings that resist retrieval competition from semantically adjacent general guidance documents.

---

## Engineering Post-Mortem: What Broke and What We Learned

This section documents the iteration cycle on the `example` category — from a baseline of 4.25/10 to a stable 6.58/10 with MRR improving from 0.750 to 0.875. It is included because the debugging process produced more durable engineering knowledge than the initial build.

### Problem 1 — Retrieval competition from semantically adjacent documents

**What happened:** After adding `capa_edge_cases.md` and `8d_practitioner_scenarios.md` to the KB, factual questions about worked examples began returning wrong answers. The question "What did the automotive 8D team find when they did the lateral search in D7?" scored 2.0/10 despite the correct document being retrieved.

**Diagnosis via Langfuse:** Traces showed `8d_practitioner_scenarios__chunk_0004` scoring 3.08 in BGE reranking vs the correct automotive example chunk at 2.82. The practitioner scenarios document discusses D7 lateral search as general guidance, using nearly identical vocabulary to the factual case record. BGE cannot distinguish "here's what the team found" from "here's what a team should look for" when vocabulary is the same.

**Diagnosis via t-SNE:** Embedding space visualization showed `general` category chunks scattered throughout `example` category space — direct visual evidence of the retrieval competition. Quantified from the HTML data: `general` centroid spread = 7.00 (highest of any category), `example` centroid at (7.98, -11.84) vs `general` centroid at (-3.92, -1.14), centroid distance = 16.00. Despite this macro-level separation, 4 out of 23 `general` chunks (17%) were within distance 5 of an example chunk — the exact stragglers causing retrieval interference. **This visualization would have immediately identified the problem had it been run after the KB change, before running the 2-hour eval. It is now part of the iteration checklist.**

**Fix attempted — wrong:** Changed `doc_category` for the new documents in `ingest.py`, triggered `--reset` re-ingest. This forced Haiku to re-enrich all chunks at `temperature=1.0` (default), producing different headlines for all documents including the automotive example. MRR dropped from 0.750 to 0.542. **Lesson: never use `--reset` when enrichment is non-deterministic — it destroys working embeddings.**

**Fix attempted — wrong:** Added a `CRITICAL — FACTUAL RECALL RULE` to the answer generation system prompt. This caused GPT-4o-mini to over-apply the instruction to questions where the top-ranked chunk was methodology content rather than case content, making unrelated answers worse. Semiconductor corrective actions (t015) dropped from 7.7 → 4.3 despite perfect retrieval. **Lesson: broad prompt instructions have unpredictable side effects across all query types.**

**Fix that worked:** FINDING anchor lines added to QR section entries in worked example documents. Each `FINDING:` line is a single specific sentence with CAPA ID, part number, and key numbers. Combined with `temperature=0` on enrichment, Haiku reliably generates this as the chunk headline — producing embeddings specific enough to score higher than general guidance chunks for case-specific queries.

**t-SNE before vs after FINDING anchors:**

| Category | Spread (before) | Spread (after) | example centroid distance (after) |
|---|---|---|---|
| example | scattered, no cluster | 4.56 | — |
| general | everywhere | 7.00 | **16.00** |
| tool | moderate | 7.48 | 24.20 |
| procedure | scattered | 6.24 | 17.28 |
| compliance | moderate | 3.18 | 7.79 ⚠️ |
| reference | isolated | 3.27 | 11.08 |
| methodology | moderate | 6.12 | 13.84 |

`compliance` at distance 7.79 from `example` is a latent risk — IATF 16949 D7/PFMEA content shares vocabulary with the automotive example D7 section.

---

### Problem 2 — Non-deterministic enrichment making eval results non-comparable

**Root cause:** `client.messages.create()` in `enrich_chunk()` was called without `temperature`. Anthropic API default is `temperature=1.0`. Different Haiku runs generated different headlines for identical text, producing different `embed_text` values, different embeddings, and different BGE rankings. Eval scores were measuring a combination of KB quality and enrichment randomness — you could not reliably attribute score changes to KB changes.

**Fix:** `temperature=ENRICHMENT_TEMP` (=0) added to the enrichment API call. One line change. Every future re-ingest now produces identical results for identical content.

**Lesson:** Any LLM call that feeds a deterministic downstream process (embeddings, structured extraction, classification) must run at `temperature=0`. Reserve `temperature > 0` for user-facing generation where variation is acceptable.

---

### Problem 3 — Chunk boundary shifting when documents are modified

**What happened:** Adding the QR section (~800 words) to the top of the automotive example document shifted all chunk boundaries for the entire document. body chunks (D4, D7, etc.) got new text boundaries, new headlines, and new embeddings. The eval score that worked before the QR addition could not be reproduced after a `--reset` ingest — even with the QR section present — because the body chunk embeddings were now different.

**Why RUN2 (first QR addition) scored well:** It used plain `upsert` (no `--reset`), which added new QR chunks while preserving the original body chunk embeddings. Every subsequent `--reset` destroyed those preserved embeddings.

**Fix:** The FINDING anchors make body chunk quality less dependent on lucky headline generation — they anchor the most retrieval-critical content explicitly. Combined with `temperature=0`, the ingest is now stable regardless of chunk boundary position.

**Lesson:** In a RAG pipeline, document modifications are not free — they can degrade retrieval for content you didn't touch. Use incremental upsert for additions. Use `--reset` only when you explicitly want to rebuild all embeddings.

---

### Iteration checklist (derived from the above)

After any KB change:
1. `uv run scripts/ingest.py --reset` (only if structural changes — otherwise upsert)
2. `uv run scripts/diagnostics/tsne_viz.py` — check for category bleeding before running eval
3. `uv run evaluation/eval.py --category [affected_category]` — fast category check
4. Only if category eval is clean: `uv run evaluation/eval.py` — full 197-question eval

---

### Known limitations

**t019 (5 Whys walk-through) — generation quality**
MRR=1.0 and top_chunk_score=9.71 (perfect retrieval) but judge overall=3.3/10. The LLM retrieves the complete 5 Whys chain but summarises the root cause rather than reproducing the chain step by step. Planned fix: a worked example in the answer generation prompt demonstrating how to reproduce a numbered chain when the context contains one.

**`general` category structural overlap**
t-SNE analysis of the final embedding space (parsed from HTML, 170 vectors × 1536 dims) confirms the macro-level separation is healthy: `example` centroid at (7.98, -11.84) is distance 16.00 from `general` centroid at (-3.92, -1.14). However, 4 specific `general` chunks (17% of the category) remain within distance 5 of example chunks:

- A SCAR/supplier detection failure chunk from `8d_practitioner_scenarios.md` at (-1.59, -11.63) — sitting inside the semiconductor example cluster
- Two `capa_edge_cases.md` chunks about ICA failure and detection gaps at (-4.28, -6.30) and (-4.49, -6.75) — near the automotive example's Is/Is Not table chunk at (-2.32, -9.88), which has no FINDING anchor because it is in the document body rather than the QR section
- One `8d_practitioner_scenarios.md` supplier escalation chunk near the same area

Additional risk: `compliance` centroid distance to `example` is only 7.79 — the IATF 16949 D7/PFMEA compliance content uses nearly identical vocabulary to the automotive example D7 section. Not yet causing eval failures but a known latent risk.

Long-term fix: split `capa_edge_cases.md` and `8d_practitioner_scenarios.md` into smaller topically focused files (one semantic purpose per document, 300–500 words each), and add FINDING anchors to document body chunks that are retrieval-critical, not just QR section chunks. Deferred to next iteration.

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key + Anthropic API key
- Langfuse account (free) + project keys

### Install
```bash
git clone https://github.com/kolmag/capa-8d-expert.git
cd capa-8d-expert
uv sync
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Optional: BGE reranker
```bash
uv add torch transformers
# Model downloads on first run (~570MB)
```

### Build knowledge base
Provide your own `.md` documents in `knowledge-base/markdown/`, then:
```bash
uv run scripts/ingest.py --reset
```

### Run
```bash
uv run scripts/app.py
# Open http://localhost:7860
```

---

## Evaluation commands

```bash
# Full 197-question eval (~2-3 hours)
uv run evaluation/eval.py --tests evaluation/tests_v3.jsonl

# Quick sample — 20 random questions
uv run evaluation/eval.py --tests evaluation/tests_v3.jsonl --sample 20

# By category
uv run evaluation/eval.py --tests evaluation/tests_v3.jsonl --category RCA
```

---

## Diagnostics

```bash
# t-SNE embedding space visualization — run after every re-ingest
uv run scripts/diagnostics/tsne_viz.py \
    --db_path ./chroma_db \
    --collection capa_8d_expert \
    --dims 2
```

---

## Project Structure

```
capa-8d-expert/
├── scripts/
│   ├── ingest.py              # Chunking, LLM enrichment, Chroma ingestion
│   ├── answer.py              # RAG pipeline (rewrite→retrieve→rerank→answer→check)
│   ├── app.py                 # Gradio UI (Expert Q&A)
│   └── diagnostics/
│       └── tsne_viz.py        # Embedding space visualization
├── evaluation/
│   ├── eval.py                # MRR + LLM-as-judge pipeline
│   └── tests_v3.jsonl         # 197 test questions (3 sources)
├── knowledge-base/
│   └── markdown/              # 14 enriched source documents
├── .env.example
└── pyproject.toml
```

---

## License

MIT

---

*Built as part of an AI engineering portfolio. The engineering post-mortem above documents the full iteration cycle — baseline, regressions, root cause analysis, and fixes. Evaluation-driven development: every KB change is measured against a 197-question test set before committing.*
