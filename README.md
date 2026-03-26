# 🔍 CAPA/8D Expert Knowledge Worker

An AI-powered expert assistant for quality engineers working with CAPA procedures, 8D problem-solving methodology, root cause analysis, FMEA, and industry compliance standards.

Built as a production-grade RAG pipeline with full evaluation framework — not a demo.

---

## What it does

Quality engineers ask questions in natural language. The system retrieves the most relevant content from a curated knowledge base, reranks it with a local cross-encoder, generates a grounded expert answer, and strips any claims not supported by the retrieved context.

**Example questions it handles well:**
- *"Our containment sort found zero defects but the customer found three more — do we expand the suspect window?"*
- *"What are the IATF 16949 specific requirements for 8D?"*
- *"Why is calendar-based PM dangerous when production volume changes?"*
- *"Is it acceptable to close an 8D if the PCA relies 100% on operator retraining?"*

---

## Architecture

```
User question
     │
     ▼
Query Rewriting ── Claude Haiku generates 3 alternative phrasings
     │
     ▼
Retrieval ── text-embedding-3-small + Chroma (top 20 per query)
     │
     ▼
Merge & Deduplicate ── union across 4 queries, deduplicated by chunk ID
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
| UI | Gradio 5.x |

---

## Knowledge Base

12 enriched documents, ~22,000 words, 96 chunks:

| Document | Coverage |
|---|---|
| `8D_problem_solving_methodology.md` | D0–D8 complete guide, timing norms, common mistakes |
| `8D_report_example_automotive.md` | Steering bracket bore defect — full worked example |
| `8D_report_example_semiconductor.md` | IC wirebond failure — SEM/EDX analysis |
| `CAPA_SOP_enriched.md` | ISO 9001 Clause 10.2, CAPA ID format, VoE criteria |
| `containment_decision_guide.md` | Full location table, ICA methods, removal logic |
| `control_plan_basics.md` | Column-by-column guide, 25% volume change rule |
| `effectiveness_verification_guide.md` | VoE metrics by CA type, Cpk thresholds |
| `fmea_basics.md` | S/O/D/RPN, AIAG-VDA AP, material substitution rules |
| `is_is_not_analysis.md` | Template + 2 worked examples |
| `multi-industry_CAPA_8D_compliance.md` | ISO 9001, IATF 16949, AS9100, ISO 13485, Semiconductor |
| `rca_tool_selection_matrix.md` | 5 Whys/Ishikawa/FTA routing logic |
| `root_cause_analysis.md` | 5 Whys + Ishikawa toolkits, stopping criteria |

---

## Evaluation

Evaluated against **197 test questions** across 3 independent sources:
- **t001–t050**: Developer-written structured questions
- **t051–t197**: Blind questions from Gemini Pro and ChatGPT (practitioner-phrased)

| Metric | Score |
|---|---|
| MRR (source retrieval) | **0.947** |
| Source coverage rate | **100%** |
| Judge — Correctness | **8.74/10** |
| Judge — Completeness | **8.42/10** |
| Judge — Groundedness | **8.26/10** |
| Judge — Overall | **8.47/10** |
| Blind questions overall | **7.55/10** |

**By category:**

| Category | MRR | Overall |
|---|---|---|
| compliance | 1.000 | 8.83 |
| 8D_methodology | 1.000 | 8.67 |
| FMEA | 1.000 | 9.00 |
| RCA | 0.875 | 8.83 |
| enriched_content | 1.000 | 7.89 |
| containment | 1.000 | 7.89 |
| example | 1.000 | 9.50 |

---

## Key Engineering Decisions

**1. LLM-enriched chunking**
Each chunk is enriched at ingest with a Claude Haiku-generated headline and summary. Embedding = `headline + summary + original_text`. Improves retrieval for questions phrased differently from source material.

**2. BGE cross-encoder reranking**
`BAAI/bge-reranker-v2-m3` runs locally (free, no API). Reranks 40–80 merged candidates to 15. ~2s after warmup vs ~40s for LLM reranking. Falls back to Claude Haiku if torch not installed.

**3. Groundedness post-checker**
After answer generation, Claude Haiku audits each claim against retrieved chunks and strips anything ungrounded. Only fires when top BGE score ≥ 0.5 — skips pure synthesis queries where claim-level checking would be too aggressive.

**4. Three-source test set**
197 questions built from three independent sources to prevent eval overfitting. External questions are practitioner-phrased with no knowledge of document structure.

**5. Conversation history threading**
Both query rewriting and answer generation are history-aware for grounded follow-up questions.

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key + Anthropic API key

### Install
```bash
git clone https://github.com/YOUR_USERNAME/capa-8d-expert.git
cd capa-8d-expert
uv sync
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Optional: BGE reranker
```bash
uv add torch transformers
# Model downloads on first run (~570MB)
```

### Build knowledge base
```bash
uv run scripts/ingest.py --reset
```

### Run
```bash
uv run scripts/app.py
# Open http://localhost:7860
```

---

## Evaluation

```bash
# Quick — 20 random questions (~35 min)
uv run evaluation/eval.py --tests evaluation/tests_v3.jsonl --sample 20

# Full 197-question eval (~3-4 hours)
uv run evaluation/eval.py --tests evaluation/tests_v3.jsonl

# By category
uv run evaluation/eval.py --tests evaluation/tests_v3.jsonl --category edge_case
```

---

## Project Structure

```
capa-8d-expert/
├── scripts/
│   ├── ingest.py          # Chunking, LLM enrichment, Chroma ingestion
│   ├── answer.py          # RAG pipeline (rewrite→retrieve→rerank→answer→check)
│   └── app.py             # Gradio UI
├── evaluation/
│   ├── eval.py            # MRR + LLM-as-judge pipeline
│   └── tests_v3.jsonl     # 197 test questions (3 sources)
├── knowledge-base/
│   └── markdown/          # 12 enriched source documents
├── .env.example
└── pyproject.toml
```

---

## License

MIT

---

*Built as part of an AI engineering portfolio. Knowledge base covers CAPA/8D quality management — a domain where answer accuracy matters.*
