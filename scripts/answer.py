"""
answer.py — CAPA/8D Expert Query Pipeline
Pattern: rewrite → retrieve → merge → rerank → answer

Pipeline:
  1. rewrite_query    — generate N alternative phrasings with Claude Haiku
  2. fetch_context    — retrieve K=20 candidates per query (original + rewritten) from Chroma
  3. merge            — deduplicate by chunk_id, union all results
  4. rerank           — BAAI/bge-reranker-v2-m3 (local HF cross-encoder), fallback to LLM
  5. answer           — generate expert answer with GPT-4o-mini + source citations

Reranker strategy:
  Primary:  BAAI/bge-reranker-v2-m3 (HuggingFace, runs locally, free, fast)
  Fallback: Claude Haiku LLM scoring (used if torch/transformers not installed)

Usage:
    uv run scripts/answer.py "What should I do first when a customer reports a defect?"
    uv run scripts/answer.py --no-rewrite "What is D3 ICA?"
    uv run scripts/answer.py --debug "How do I update the FMEA after a CAPA?"
    uv run scripts/answer.py --reranker llm "..."   # force LLM reranker
    uv run scripts/answer.py --reranker bge "..."   # force BGE reranker
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env if _env.exists() else None)
except ImportError:
    pass

import chromadb
from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Config ─────────────────────────────────────────────────────────────────────

CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "capa_8d_expert"

REWRITE_MODEL   = "claude-haiku-4-5"    # fast query rewriting
ANSWER_MODEL    = "gpt-4o-mini"         # answer generation
LLM_RERANK_MODEL = "claude-haiku-4-5"  # fallback LLM reranker

BGE_MODEL_NAME  = "BAAI/bge-reranker-v2-m3"  # HuggingFace cross-encoder

N_REWRITES      = 3     # number of alternative query phrasings
RETRIEVAL_K     = 20    # candidates retrieved per query
FINAL_K         = 15    # chunks kept after reranking for answer context
ANSWER_TEMP     = 0     # deterministic expert answers


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:      str
    source_file:   str
    doc_category:  str
    headline:      str
    original_text: str
    distance:      float


@dataclass
class RankedChunk:
    chunk_id:        str
    source_file:     str
    doc_category:    str
    headline:        str
    original_text:   str
    relevance_score: float   # 0–10 scale for UI display
    reranker:        str     # "bge" or "llm"


@dataclass
class AnswerResult:
    question:          str
    rewritten_queries: list[str]
    ranked_chunks:     list[RankedChunk]
    answer:            str
    sources:           list[str]
    reranker_used:     str
    checker_score:     float = 1.0   # Option 3 groundedness score (0-1); 1.0 = skipped or perfect


# ── BGE reranker (HuggingFace) ─────────────────────────────────────────────────

_bge_model = None   # lazy-loaded singleton

def _load_bge() -> Optional[object]:
    """
    Lazy-load BAAI/bge-reranker-v2-m3.
    Returns the model if available, None if torch/transformers not installed.
    First call downloads the model (~570MB) to HuggingFace cache.
    """
    global _bge_model
    if _bge_model is not None:
        return _bge_model

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        print(f"  Loading {BGE_MODEL_NAME} (first run downloads ~570MB)...")
        tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
        model     = AutoModelForSequenceClassification.from_pretrained(BGE_MODEL_NAME)
        model.eval()

        # Use MPS on Apple Silicon, CUDA on GPU, else CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        _bge_model = (tokenizer, model, device)
        print(f"  BGE reranker loaded on {device}")
        return _bge_model

    except ImportError:
        return None


def bge_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    final_k: int = FINAL_K,
) -> tuple[list[RankedChunk], bool]:
    """
    Rerank chunks using BAAI/bge-reranker-v2-m3.
    Returns (ranked_chunks, success).
    If model unavailable, returns ([], False) to trigger fallback.
    """
    bge = _load_bge()
    if bge is None:
        return [], False

    try:
        import torch
        tokenizer, model, device = bge

        # Build (query, passage) pairs
        pairs = [[question, chunk.original_text[:512]] for chunk in chunks]

        # Tokenise in one batch
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        # Score
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()

        # Map raw sigmoid score (0–1) to 0–10 scale for UI consistency
        ranked = []
        for chunk, raw_score in zip(chunks, scores):
            ranked.append(RankedChunk(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                doc_category=chunk.doc_category,
                headline=chunk.headline,
                original_text=chunk.original_text,
                relevance_score=round(float(raw_score) * 10, 2),
                reranker="bge",
            ))

        ranked.sort(key=lambda c: c.relevance_score, reverse=True)
        return ranked[:final_k], True

    except Exception as e:
        print(f"  BGE reranker error: {e} — falling back to LLM reranker")
        return [], False


# ── LLM reranker (Claude Haiku fallback) ───────────────────────────────────────

RERANK_SYSTEM = """You are a relevance scoring expert for a CAPA/8D quality management knowledge base.

Score how relevant a text chunk is to answering the given question.
Return ONLY a JSON object: {"score": <number 0-10>, "reason": "<one sentence>"}

Scoring guide:
10 = Directly answers the question with specific detail
7-9 = Highly relevant, contains key information needed
4-6 = Partially relevant, touches on the topic
1-3 = Tangentially related
0 = Not relevant"""

RERANK_PROMPT = """Question: {question}

Chunk (from {source_file}):
{text}

Score this chunk's relevance to the question. Return JSON only."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def _llm_score_chunk(question: str, chunk: RetrievedChunk, client: Anthropic) -> float:
    response = client.messages.create(
        model=LLM_RERANK_MODEL,
        max_tokens=100,
        system=RERANK_SYSTEM,
        messages=[{"role": "user", "content": RERANK_PROMPT.format(
            question=question,
            source_file=chunk.source_file,
            text=chunk.original_text[:800],
        )}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return float(json.loads(raw).get("score", 0))


def llm_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    final_k: int = FINAL_K,
) -> list[RankedChunk]:
    """LLM-based reranking — one Haiku call per chunk."""
    client = Anthropic()
    ranked = []
    for chunk in chunks:
        score = _llm_score_chunk(question, chunk, client)
        ranked.append(RankedChunk(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            doc_category=chunk.doc_category,
            headline=chunk.headline,
            original_text=chunk.original_text,
            relevance_score=score,
            reranker="llm",
        ))
    ranked.sort(key=lambda c: c.relevance_score, reverse=True)
    return ranked[:final_k]


def rerank(
    question: str,
    chunks: list[RetrievedChunk],
    final_k: int = FINAL_K,
    mode: str = "auto",   # "auto" | "bge" | "llm"
) -> tuple[list[RankedChunk], str]:
    """
    Rerank with the best available method.
    mode="auto": try BGE first, fall back to LLM
    mode="bge":  force BGE (error if unavailable)
    mode="llm":  force LLM reranker
    Returns (ranked_chunks, reranker_name)
    """
    if mode == "llm":
        return llm_rerank(question, chunks, final_k), "llm"

    if mode == "bge":
        ranked, ok = bge_rerank(question, chunks, final_k)
        if not ok:
            raise RuntimeError(
                "BGE reranker unavailable. Install: uv add torch transformers"
            )
        return ranked, "bge"

    # auto: try BGE, fall back to LLM
    ranked, ok = bge_rerank(question, chunks, final_k)
    if ok:
        return ranked, "bge"
    return llm_rerank(question, chunks, final_k), "llm"


# ── Chroma client ──────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma DB not found at '{CHROMA_DIR}'. "
            "Run: uv run scripts/ingest.py --reset"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


# ── Step 1: Query rewriting ────────────────────────────────────────────────────

REWRITE_SYSTEM = """You are a query expansion specialist for a CAPA and 8D problem-solving knowledge base.

Given a user question, generate alternative phrasings that will help retrieve relevant chunks.
Think about: synonyms, related concepts, different levels of specificity, and domain terminology.

Return a JSON array of strings — alternative query phrasings only. No explanation, no markdown."""

REWRITE_PROMPT = """Original question: {question}

Generate {n} alternative phrasings that would retrieve relevant content from a CAPA/8D knowledge base.
Cover: different terminology, more specific versions, related concepts.

Return JSON array only: ["phrasing 1", "phrasing 2", ...]"""

REWRITE_PROMPT_WITH_HISTORY = """Conversation so far:
{history_summary}

Current question: {question}

The current question may be a follow-up to the conversation above.
Generate {n} alternative phrasings that would retrieve relevant content from a CAPA/8D knowledge base.
If the question is a follow-up (e.g. "give me examples", "what about X"), expand it using the conversation context.

Return JSON array only: ["phrasing 1", "phrasing 2", ...]"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def rewrite_query(
    question: str,
    n: int = N_REWRITES,
    history: list[dict] | None = None,
) -> list[str]:
    """Generate N alternative query phrasings, optionally grounded in conversation history."""
    client = Anthropic()

    # Build history summary for context (last 3 turns max)
    if history and len(history) >= 2:
        turns = history[-6:]  # last 3 user+assistant pairs
        history_summary = ""
        for msg in turns:
            role = "User" if msg["role"] == "user" else "Assistant"
            text = msg["content"][:300]  # truncate long assistant answers
            history_summary += f"{role}: {text}\n"
        prompt = REWRITE_PROMPT_WITH_HISTORY.format(
            history_summary=history_summary.strip(),
            question=question,
            n=n,
        )
    else:
        prompt = REWRITE_PROMPT.format(question=question, n=n)

    response = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=300,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return json.loads(raw)[:n]


# ── Step 2: Retrieval ──────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding


def retrieve(
    query: str,
    collection: chromadb.Collection,
    k: int = RETRIEVAL_K,
) -> list[RetrievedChunk]:
    embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunks.append(RetrievedChunk(
            chunk_id=results["ids"][0][i],
            source_file=meta.get("source_file", ""),
            doc_category=meta.get("doc_category", ""),
            headline=meta.get("headline", ""),
            original_text=results["documents"][0][i],
            distance=results["distances"][0][i],
        ))
    return chunks


# ── Step 3: Merge + deduplicate ────────────────────────────────────────────────

def merge_results(all_results: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    seen: dict[str, RetrievedChunk] = {}
    for result_list in all_results:
        for chunk in result_list:
            if chunk.chunk_id not in seen or chunk.distance < seen[chunk.chunk_id].distance:
                seen[chunk.chunk_id] = chunk
    return sorted(seen.values(), key=lambda c: c.distance)


# ── Step 5: Answer generation ──────────────────────────────────────────────────

ANSWER_SYSTEM = """You are an expert CAPA and 8D problem-solving consultant with deep knowledge of:
- 8D methodology (D0–D8) and CAPA procedures
- Root cause analysis tools (5 Whys, Ishikawa, Is/Is Not, FTA)
- FMEA, Control Plans, and quality management systems
- ISO 9001:2015, IATF 16949, and industry-specific standards

Answer questions precisely and practically, as an expert advising a quality engineer.
- Give specific, actionable guidance — not generic descriptions
- Reference specific disciplines (D3, D4, etc.) and tools by name when relevant
- If the context contains worked examples, reproduce the specific facts, numbers, and findings from those examples directly — do not paraphrase or generalise
- If the question asks about a step or decision, explain both what to do AND common mistakes to avoid
- Be direct — quality engineers need clear answers, not hedged summaries

CRITICAL — GROUNDEDNESS RULE (read carefully, every point matters):
- Base your answer ONLY on the provided knowledge base context chunks
- Every claim must be directly traceable to a specific chunk in the context
- Do NOT add introductory phrases like "Great question", "In quality management...", or "It is important to note that..."
- Do NOT add concluding summaries or transitional filler — end when the answer is complete
- Do NOT add generic quality management advice (e.g. "engage cross-functional teams", "ensure management buy-in", "conduct training", "foster a culture of quality") unless those exact concepts appear in the retrieved context with specific guidance
- Do NOT substitute from general knowledge when a chunk is incomplete — omit the missing detail entirely
- If a sequential process (like 5 Whys steps or a numbered checklist) is in the context, reproduce it in full in the correct order — do not summarise or skip steps
- Shorter, fully-grounded answers are better than longer answers that mix grounded and ungrounded content
- If the question cannot be fully answered from the context, say: "Based on the available documentation, I can cover [topics]. For [missing topic], consult [relevant standard] directly."

At the end of your answer, list the sources you drew from as: [Source: filename]"""

ANSWER_PROMPT = """Question: {question}

Relevant knowledge base context:
{context}

Provide a precise, expert answer based on the context above."""

ANSWER_PROMPT_FOLLOWUP = """The user is continuing an ongoing conversation. Use the conversation history to understand any follow-up questions or references.

Current question: {question}

Relevant knowledge base context:
{context}

Provide a precise, expert answer based on the context above. If the question references something from earlier in the conversation, incorporate that context naturally."""


def build_context(chunks: list[RankedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk.headline}\n"
            f"Source: {chunk.source_file} | Score: {chunk.relevance_score:.2f}\n"
            f"{chunk.original_text}"
        )
    return "\n\n---\n\n".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def generate_answer(
    question: str,
    chunks: list[RankedChunk],
    history: list[dict] | None = None,
) -> str:
    """Generate an expert answer, optionally with conversation history for follow-ups."""
    client = OpenAI()
    context = build_context(chunks)

    if history and len(history) >= 2:
        # Build messages: system + last N turns + current question with context
        messages = [{"role": "system", "content": ANSWER_SYSTEM}]

        # Include last 4 turns (2 user + 2 assistant) for context
        for msg in history[-4:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"][:800],  # truncate long turns
            })

        # Add current question with retrieved context
        messages.append({
            "role": "user",
            "content": ANSWER_PROMPT_FOLLOWUP.format(
                question=question,
                context=context,
            ),
        })
    else:
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",   "content": ANSWER_PROMPT.format(
                question=question,
                context=context,
            )},
        ]

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=ANSWER_TEMP,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# ── Step 5b: Groundedness post-check ──────────────────────────────────────────

GROUNDEDNESS_SYSTEM = """You are a groundedness auditor for a RAG system.

Your job: check whether each claim in an answer is supported by the provided context chunks.

Return a JSON object with this structure:
{
  "grounded_claims": ["claim 1", "claim 2"],
  "ungrounded_claims": ["claim X", "claim Y"],
  "grounded_answer": "<the answer rewritten with only grounded claims, keeping original tone and format>",
  "groundedness_score": <0.0-1.0, fraction of claims that are grounded>,
  "removals": <number of claims removed>
}

Rules:
- A claim is grounded if its core fact or guidance appears in the context, even if phrased differently
- A claim is ungrounded if it adds general knowledge not present in any context chunk
- The grounded_answer must be a natural, readable response — not a list of bullet points unless the original was
- Keep all grounded claims intact, including their inline [Source: ...] citations
- If all claims are grounded, return the original answer unchanged in grounded_answer
- Never add new information — only remove ungrounded claims
- Return ONLY valid JSON, no markdown fences"""

GROUNDEDNESS_PROMPT = """QUESTION: {question}

CONTEXT CHUNKS (numbered, these are the only valid sources):
{context_summary}

ANSWER TO CHECK:
{answer}

Check each claim. Return JSON only."""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def check_groundedness(
    question: str,
    chunks: list[RankedChunk],
    answer_text: str,
    threshold: float = 0.75,
) -> tuple[str, float]:
    """
    Post-process answer to remove ungrounded claims.
    Returns (cleaned_answer, groundedness_score).
    Only runs the checker if answer is long enough to be worth checking (>200 chars).
    Skips if all BGE scores are very low — synthesis answers are expected to draw
    across many low-scored chunks and don't benefit from claim-level checking.
    """
    # Skip for very short answers — nothing to trim
    if len(answer_text) < 200:
        return answer_text, 1.0

    # Skip if top chunk score is extremely low — pure synthesis query
    # where claim-level grounding check would be too aggressive
    top_score = max((c.relevance_score for c in chunks), default=0)
    if top_score < 0.5:
        return answer_text, 1.0

    try:
        client = Anthropic()

        # Build compact context summary (headline + first 150 chars per chunk)
        context_parts = []
        for i, chunk in enumerate(chunks[:10], 1):
            context_parts.append(
                f"[{i}] {chunk.headline} | {chunk.original_text[:150]}..."
            )
        context_summary = "\n".join(context_parts)

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            system=GROUNDEDNESS_SYSTEM,
            messages=[{"role": "user", "content": GROUNDEDNESS_PROMPT.format(
                question=question,
                context_summary=context_summary,
                answer=answer_text[:2000],
            )}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        parsed = json.loads(raw)
        score = float(parsed.get("groundedness_score", 1.0))
        removals = int(parsed.get("removals", 0))
        cleaned_raw = parsed.get("grounded_answer", answer_text)
        # Defensively handle cases where model returns list instead of string
        if isinstance(cleaned_raw, list):
            cleaned = " ".join(str(x) for x in cleaned_raw).strip()
        else:
            cleaned = str(cleaned_raw).strip()

        # Only use cleaned answer if meaningful removals were made
        # and the cleaned answer is substantive
        if removals > 0 and len(cleaned) > 100:
            return cleaned, score
        return answer_text, score

    except Exception:
        # Groundedness check failure is non-fatal — return original answer
        return answer_text, 1.0


# ── Full pipeline ──────────────────────────────────────────────────────────────

def answer(
    question: str,
    use_rewrite: bool = True,
    debug: bool = False,
    reranker_mode: str = "auto",
    history: list[dict] | None = None,
) -> AnswerResult:
    """Full RAG pipeline: rewrite → retrieve → merge → rerank → answer.

    history: list of {"role": "user"|"assistant", "content": str} dicts
             from previous turns. Used to ground query rewrites and answer
             generation for follow-up questions.
    """
    collection = get_collection()

    # Step 1: Query rewriting (history-aware for follow-ups)
    if use_rewrite:
        rewrites = rewrite_query(question, history=history)
        if debug:
            print(f"\n[Rewrites]")
            for r in rewrites:
                print(f"  • {r}")
    else:
        rewrites = []

    all_queries = [question] + rewrites

    # Step 2: Retrieve
    all_results = []
    for q in all_queries:
        results = retrieve(q, collection, k=RETRIEVAL_K)
        all_results.append(results)
        if debug:
            print(f"\n[Retrieved {len(results)} for: '{q[:60]}...']")

    # Step 3: Merge
    merged = merge_results(all_results)
    if debug:
        print(f"\n[Merged: {len(merged)} unique chunks]")

    # Step 4: Rerank
    ranked, reranker_used = rerank(question, merged, final_k=FINAL_K, mode=reranker_mode)
    if debug:
        print(f"\n[Top {len(ranked)} after reranking — {reranker_used}]")
        for c in ranked:
            print(f"  {c.relevance_score:.2f} | {c.source_file} | {c.headline[:60]}")

    # Step 5: Generate answer
    answer_text = generate_answer(question, ranked, history=history)

    # Step 5b: Groundedness post-check — strip ungrounded claims
    answer_text, groundedness_score = check_groundedness(
        question, ranked, answer_text
    )
    if debug:
        print(f"\n[Groundedness score: {groundedness_score:.2f}]")

    sources = list(dict.fromkeys(c.source_file for c in ranked))

    return AnswerResult(
        question=question,
        rewritten_queries=rewrites,
        ranked_chunks=ranked,
        answer=answer_text,
        sources=sources,
        reranker_used=reranker_used,
        checker_score=groundedness_score,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAPA/8D Expert — Query Pipeline")
    parser.add_argument("question", type=str, help="Question to answer")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="Skip query rewriting")
    parser.add_argument("--debug", action="store_true",
                        help="Show retrieval and reranking details")
    parser.add_argument("--reranker", choices=["auto", "bge", "llm"], default="auto",
                        help="Reranker to use (default: auto)")
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"Question: {args.question}")
    print(f"{'─'*60}\n")

    result = answer(
        question=args.question,
        use_rewrite=not args.no_rewrite,
        debug=args.debug,
        reranker_mode=args.reranker,
    )

    print(result.answer)
    print(f"\n{'─'*60}")
    print(f"Sources   : {', '.join(result.sources)}")
    print(f"Chunks    : {len(result.ranked_chunks)}")
    print(f"Reranker  : {result.reranker_used}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
