# Lessons Learned — CAPA/8D Expert (App 1)

**Project duration:** ~3 weeks active development  
**Final state:** Production-grade RAG pipeline, eval-driven iteration, published to GitHub  
**Applies to:** All future RAG portfolio apps, starting with Auditor Expert (App 2)

---

## 1. Pipeline Architecture — What to Keep

These decisions were correct from the start and should be copied verbatim into every future app.

### LLM-enriched chunking (embed_text = headline + summary + queries + original)
The single highest-leverage architectural decision. Standard RAG embeds raw text. Enriched RAG embeds `headline + summary + practitioner_queries + original_text`. The headline gives BGE a precise anchor. The summary adds context. The practitioner queries bridge formal SOP vocabulary to conversational question phrasing. Do not skip this on any future app.

### BGE cross-encoder reranking
Free, local, ~2s after warmup, meaningfully better than embedding-only retrieval. Keep it. The graceful fallback to LLM reranker when torch is unavailable is the right pattern for M1 8GB constraints.

### Three-source test set (developer + blind external + adversarial)
197 questions from three independent sources prevents eval overfitting. The blind questions (practitioner-phrased, no knowledge of document structure) are the most honest signal. Build the blind question set from day one on every future app — don't add it later.

### Langfuse observability on every pipeline run
Trace-level debugging (which chunks ranked, what BGE scored them, what the checker did) saved hours of guesswork. Without Langfuse traces we would not have diagnosed the 3.08 vs 2.82 BGE score competition that caused the t038 regression. Non-negotiable for production-grade RAG.

### Temperature=0 on ALL enrichment LLM calls
The single most important operational lesson. The Anthropic API default is temperature=1.0. Non-zero temperature on enrichment means every `--reset` re-ingest produces different headlines → different embeddings → different BGE rankings → non-comparable eval scores. Discovered after three consecutive regressions. Add this on day one — one line, zero cost.

### Groundedness checker (Claude Haiku as critic)
The actor/critic pattern is already implemented. GPT-4o-mini generates, Claude Haiku audits and strips hallucinated claims. Keep this. Gemini correctly identified this as a production-grade agentic pattern.

---

## 2. KB Architecture — What to Do Differently

These are the structural mistakes that cost the most time.

### One semantic purpose per document — strictly
The biggest source of retrieval competition was documents that covered multiple semantic domains. `8d_practitioner_scenarios.md` covered internal rejection, customer rejection, supplier rejection, daily updates, and speed vs thoroughness — five distinct topics in one file. The embedding model averages the semantics, producing chunks that land in the middle of the embedding space rather than in a precise location.

**For Auditor Expert:** One document per topic. Maximum 600 words per document. If a topic needs more than 600 words, split it into two documents with distinct names.

### FINDING anchors in worked examples from day one
The `FINDING:` line pattern — a single factual sentence with document ID, case reference, and key numbers at the start of each QR entry — was the fix that took example category MRR from 0.383 to 1.000. It works because it forces Haiku to generate a highly specific headline at temperature=0, making the chunk resistant to retrieval competition from adjacent categories.

**For Auditor Expert:** Every worked example, every case study, every specific audit finding — add FINDING anchors from the start. Don't wait for a regression to tell you it's needed.

### Add practitioner scenario sections to every procedural document
The containment_decision_guide.md scored 6.803 until we added scenario-based content covering line stop urgency, in-transit parts, zero-defect sort results, and field containment. After adding those sections it jumped to 7.697. The formal SOP vocabulary and the practitioner question vocabulary don't overlap — you need both in the same document.

**For Auditor Expert:** Every procedural document gets a `## Practitioner Scenarios` section. Write it at the same time as the formal content, not as a retrofit.

### Synthetic practitioner queries in enrichment (from day one)
The `practitioner_queries` field in the enrichment prompt — 3 conversational questions Haiku generates per chunk — bridges the vocabulary gap between formal SOP language and the way practitioners actually ask questions. Added midway through CAPA/8D Expert. For Auditor Expert, include it in `ENRICHMENT_PROMPT` from the first ingest.

---

## 3. Evaluation — Process Lessons

### The iteration checklist (non-negotiable)
Every KB change must follow this sequence — no shortcuts:
1. `--reset` ingest (or upsert if only adding, not modifying)
2. t-SNE — check for category bleeding
3. Sc heatmap — check cross-category competition risks
4. Category eval (fast) — confirm target categories improved
5. Full eval — only after category eval is clean

Skipping t-SNE after a KB change caused the worst regression in the project — the category map change that destroyed working embeddings. A 30-second t-SNE would have shown the problem immediately.

### Run t-SNE as a diagnostic, not a presentation step
t-SNE is not a pretty picture for the README. It is a retrieval debugger. Run it after every significant KB change and after every `--reset` ingest. The before/after comparison is where the value is — not the final state.

### Sc analysis (intra-category + cross-category heatmap)
t-SNE shows spatial relationships. Sc quantifies them. Both are needed:
- t-SNE: "are my categories well-separated visually?"
- Sc heatmap: "which specific pairs have retrieval competition risk?"
- Sc per-query (with `--queries`): "which categories are appearing in top-K for specific question types?"

For Auditor Expert: run Sc heatmap **before the first ingest** on the designed KB documents. Catch overlap at the design stage, not after 2 hours of eval time.

### Eval score comparability requires deterministic enrichment
Before fixing temperature=0, eval scores were measuring a mixture of KB quality and enrichment randomness. You could not attribute score changes to KB changes. This invalidated weeks of iteration history. With temperature=0, every re-ingest of the same documents produces identical embeddings — scores are now fully comparable across runs.

### Non-deterministic `--reset` is destructive
Every `--reset` re-ingest regenerates all headlines. Even at temperature=0, if chunk boundaries shift (because you added content to a document), all downstream chunks get new text and new embeddings. RUN2 scored 6.58 by using plain upsert (preserving existing embeddings). Every subsequent `--reset` destroyed those embeddings.

**Rule:** Use upsert for additions. Use `--reset` only when you explicitly want all embeddings regenerated. Never `--reset` unless you have a specific reason.

---

## 4. Retrieval Architecture — What to Do Differently

### CHUNK_SIZE = 400, not 500
BGE tokenizer ≈ 1.15× GPT token count. `CHUNK_SIZE=500` GPT tokens ≈ 575 BGE tokens, which combined with headline+summary enrichment (~80 tokens) exceeds BGE's 512-token silent truncation limit. Chunks get silently truncated during reranking. Use 400 from day one.

### Markdown-header-aware chunking from day one
The fixed token splitter cuts sequential content (5 Whys chains, numbered step lists, D-discipline sections) at arbitrary boundaries. The markdown-header splitter respects semantic boundaries — each `##` or `###` section stays intact. t019 (5 Whys walk-through) improved from 3.3 to 5.3 after switching. Build this into `ingest.py` from the start.

### RETRIEVAL_K = 30 (not 20)
More candidates for BGE to rerank = better final ranking. The marginal cost of retrieving 30 vs 20 candidates is negligible. Keep K=30 on all future apps.

### README.md exclusion from KB
The git placeholder `knowledge-base/markdown/README.md` was being ingested as a KB document. Always exclude it: `if f.name != "README.md"` in the file loader. One line, catch it on day one.

---

## 5. Diagnostics — New Tools Built

Both scripts belong in every future RAG app from the start:

**`tsne_viz.py`** — embedding space visualisation. 2D + 3D. Coloured by doc_category. Run after every re-ingest.

**`sc_viz.py`** — cosine similarity analysis. Three modes:
- Intra-category violin: how coherent is each category internally?
- Cross-category heatmap: which category pairs have retrieval competition risk?
- Per-query Sc: which categories appear in top-K for specific question types?

Copy both into `scripts/diagnostics/` of every new app. Add them to the iteration checklist.

---

## 6. Answer Generation — ANSWER_SYSTEM Prompt Lessons

### Anti-padding guardrails are necessary
Without explicit instructions, GPT-4o-mini pads answers with generic quality management advice ("engage cross-functional teams", "ensure management buy-in") that the groundedness checker strips. This produces truncated, incomplete answers that score low on completeness.

The fix: explicit prohibition of introductory phrases, transitional filler, concluding summaries, and generic advice not present in the retrieved context.

**Copy this pattern to every future app's ANSWER_SYSTEM.**

### Sequential process rule
For any app where the KB contains numbered lists, step chains, or sequential procedures: add an explicit instruction to reproduce them in full in order. Without this, the LLM summarises rather than reproduces.

### FACTUAL RECALL RULE — do not add
Tested and failed. The instruction to "reproduce specific factual content" caused the model to over-apply it to questions where the wrong chunk was ranked first, making unrelated answers worse. The FINDING anchors + good retrieval are the correct solution — not prompt engineering workarounds for bad retrieval.

### Temperature=0 for all expert apps
Consistency is more important than variation for a domain expert system. Temperature=0 everywhere except the creative/suggestion tools.

---

## 7. What We Would Do Differently From Day One

If starting CAPA/8D Expert again with what we know now:

1. **CHUNK_SIZE=400** in first commit
2. **temperature=0** on enrichment in first commit
3. **Markdown-header chunking** in first commit
4. **FINDING anchors** in all worked examples before first ingest
5. **Practitioner scenario sections** in all procedural documents before first ingest
6. **Synthetic practitioner queries** in enrichment prompt before first ingest
7. **t-SNE + Sc** run before first eval, not after first regression
8. **`README.md` exclusion** in first commit
9. **Anti-padding ANSWER_SYSTEM** guardrails in first commit
10. **Three-source test set** designed before first eval

Items 1–4 alone would have prevented the three major regression cycles that cost approximately 15 hours of debugging.

---

## 8. Skills to Review Before Auditor Expert

### `python-coding` SKILL.md
- Add RAG-specific patterns: enrichment pipeline, Chroma client, BGE reranker
- Add the iteration checklist as a standard workflow
- Add `uv run scripts/diagnostics/` as a standard project component

### `8d-capa-quality` SKILL.md  
- Add audit-specific domain context for Auditor Expert (ISO 9001 clause structure, IATF 16949 requirements, NCR grading)
- Add the FINDING anchor pattern as a KB design principle
- Add the practitioner scenario section requirement

### Ed Donner course notes
- Chunking strategies section — RecursiveCharacterTextSplitter vs semantic chunking vs header-aware: we now have a clear recommendation (header-aware for structured domain docs)
- BGE 512-token limit: confirmed, fix is CHUNK_SIZE=400
- Multi-source eval design: implemented and validated

### Mike Cohen course notes
- Sc distribution plots: implemented as `sc_viz.py` — already applied
- Cosine similarity heatmap before ingest: apply to Auditor Expert KB design before first ingest
- tokenization module: confirmed BGE vs GPT token count difference, fix applied

---

## 9. Auditor Expert — Start Clean Checklist

Before writing a single line of code for App 2:

- [ ] Design KB documents table (filename | category | target words | must-contain)
- [ ] Run Sc heatmap on designed documents (synthetic, pre-ingest) to check for overlap
- [ ] Confirm all documents have one semantic purpose each
- [ ] Plan FINDING anchor locations in all worked examples
- [ ] Plan practitioner scenario sections for all procedural documents
- [ ] Set CHUNK_SIZE=400, temperature=0, markdown chunking in `ingest.py` config
- [ ] Add synthetic practitioner queries to `ENRICHMENT_PROMPT`
- [ ] Copy anti-padding guardrails to `ANSWER_SYSTEM`
- [ ] Build three-source test set before first eval (developer + blind Gemini + blind ChatGPT)
- [ ] Multi-model benchmark plan: GPT-4o-mini, Claude Haiku 4.5, Llama 3.3 70B (Groq), Qwen2.5-72B (OpenRouter)
- [ ] Copy `tsne_viz.py` and `sc_viz.py` to `scripts/diagnostics/`
- [ ] Add iteration checklist to README from day one

---

*Document written at CAPA/8D Expert project completion. Apply all lessons above before starting Auditor Expert. The goal is to reach the equivalent of the final CAPA/8D Expert state — eval-driven, diagnostics in place, production-grade — in half the iteration cycles.*
