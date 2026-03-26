"""
ingest.py — CAPA/8D Expert Knowledge Base Ingestion Pipeline
Follows Ed Donner's day5 pattern: LLM semantic chunking + OpenAI embeddings + Chroma

Architecture:
  1. Load all .md files from knowledge-base/markdown/
  2. Split into raw text chunks (RecursiveCharacterTextSplitter)
  3. LLM enrichment: each chunk → {headline, summary, original_text} via Claude Haiku
  4. Embed enriched text (headline + summary + original_text) with text-embedding-3-small
  5. Store in Chroma with rich metadata for retrieval

Usage:
    cd capa_expert/
    python scripts/ingest.py
    python scripts/ingest.py --reset        # wipe and rebuild from scratch
    python scripts/ingest.py --dry-run      # show chunks without embedding
"""

import argparse
import json
import multiprocessing
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Load .env from project root (works when running from any subdirectory)
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env if _env.exists() else None)
except ImportError:
    pass  # dotenv not installed — falls back to environment variables

import chromadb
from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE_DIR = Path("knowledge-base/markdown")
CHROMA_DIR         = Path("chroma_db")
COLLECTION_NAME    = "capa_8d_expert"

# Chunking params (tuned from evaluation experiments)
CHUNK_SIZE         = 500    # tokens
CHUNK_OVERLAP      = 200    # tokens

# LLM for semantic enrichment
ENRICHMENT_MODEL   = "claude-haiku-4-5"   # fast + cheap + excellent JSON

# Embedding model
EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536

# Multiprocessing
NUM_WORKERS        = max(1, multiprocessing.cpu_count() - 1)

# Rate limiting
EMBED_BATCH_SIZE   = 100   # OpenAI embeds up to 2048 at once; keep conservative
ENRICH_DELAY_S     = 0.05  # small delay between Haiku calls


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class RawChunk:
    """A raw text chunk before LLM enrichment."""
    chunk_id:    str
    source_file: str
    doc_category: str   # derived from filename prefix
    text:        str
    token_count: int
    chunk_index: int
    total_chunks: int


@dataclass
class EnrichedChunk:
    """A chunk after LLM semantic enrichment — ready for embedding."""
    chunk_id:     str
    source_file:  str
    doc_category: str
    headline:     str   # LLM-generated: 1 sentence capturing the main concept
    summary:      str   # LLM-generated: 2-3 sentence contextual summary
    original_text: str  # the raw chunk text
    embed_text:   str   # headline + summary + original_text (what gets embedded)
    token_count:  int
    chunk_index:  int
    total_chunks: int


# ── Token utilities ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count: ~0.75 tokens per word (good enough for chunking)."""
    return int(len(text.split()) * 1.33)

def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Paragraph-aware recursive splitter producing meaningful semantic chunks.
    Targets chunk_size words; tries to break on paragraph or sentence boundaries.
    chunk_size=500 tokens → ~375 words; chunk_overlap=200 tokens → ~150 words.
    """
    word_size    = int(chunk_size / 1.33)    # ~375 words per chunk
    word_overlap = int(chunk_overlap / 1.33) # ~150 word overlap

    # First split on double-newlines to preserve paragraph semantics
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph would exceed chunk size, flush current buffer
        if len(current_words) + len(para_words) > word_size and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap from end of current buffer
            current_words = current_words[-word_overlap:] if word_overlap else []

        # If a single paragraph is itself larger than chunk size, split it
        if len(para_words) > word_size:
            # Flush anything in buffer first
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            # Split long paragraph at sentence boundaries
            sentences = para.replace("\n", " ").split(". ")
            sent_buffer: list[str] = []
            for sent in sentences:
                sw = sent.split()
                if len(sent_buffer) + len(sw) > word_size and sent_buffer:
                    chunks.append(". ".join(sent_buffer) + ".")
                    sent_buffer = sent_buffer[-word_overlap//10:] if word_overlap else []
                sent_buffer.extend(sw)
            if sent_buffer:
                current_words = sent_buffer
        else:
            current_words.extend(para_words)

    # Flush remaining buffer
    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if len(c.split()) > 10]  # drop micro-chunks


# ── Document loading ───────────────────────────────────────────────────────────

CATEGORY_MAP = {
    "8d_problem":       "methodology",
    "8d_report":        "example",
    "capa_sop":         "procedure",
    "containment":      "procedure",
    "control_plan":     "reference",
    "effectiveness":    "procedure",
    "fmea":             "reference",
    "is_is_not":        "tool",
    "multi-industry":   "compliance",
    "rca_tool":         "tool",
    "root_cause":       "tool",
}

def get_category(filename: str) -> str:
    stem = filename.lower()
    for prefix, category in CATEGORY_MAP.items():
        if stem.startswith(prefix):
            return category
    return "general"


def load_documents(kb_dir: Path) -> list[RawChunk]:
    """Load all .md files and split into raw token-aware chunks."""
    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")

    all_chunks: list[RawChunk] = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        raw_chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        total = len(raw_chunks)

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{md_path.stem}__chunk_{i:04d}"
            all_chunks.append(RawChunk(
                chunk_id=chunk_id,
                source_file=md_path.name,
                doc_category=get_category(md_path.stem),
                text=chunk_text,
                token_count=count_tokens(chunk_text),
                chunk_index=i,
                total_chunks=total,
            ))

    return all_chunks


# ── LLM semantic enrichment ────────────────────────────────────────────────────

ENRICHMENT_SYSTEM = """You are a quality management expert helping build a RAG knowledge base for CAPA and 8D problem-solving.

For each text chunk provided, return a JSON object with exactly these fields:
- "headline": A single precise sentence (max 20 words) capturing the main concept or rule in this chunk. Write it as a standalone statement a quality engineer could search for.
- "summary": 2-3 sentences of contextual explanation that would help retrieve this chunk for relevant questions. Include key terms, discipline names (D0-D8), standards (ISO 9001, IATF 16949), and tool names.

Return ONLY valid JSON. No preamble, no markdown fences, no explanation."""

ENRICHMENT_PROMPT = """Chunk from document: {source_file}
Category: {doc_category}
Position: chunk {chunk_index} of {total_chunks}

TEXT:
{text}

Return JSON with "headline" and "summary" fields only."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def enrich_chunk(chunk: RawChunk, client: Anthropic) -> EnrichedChunk:
    """Call Claude Haiku to generate headline + summary for a chunk."""
    prompt = ENRICHMENT_PROMPT.format(
        source_file=chunk.source_file,
        doc_category=chunk.doc_category,
        chunk_index=chunk.chunk_index,
        total_chunks=chunk.total_chunks,
        text=chunk.text,
    )

    response = client.messages.create(
        model=ENRICHMENT_MODEL,
        max_tokens=300,
        system=ENRICHMENT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    headline = parsed.get("headline", "").strip()
    summary  = parsed.get("summary", "").strip()

    # Build the embed text: headline + summary + original (order matters for retrieval)
    embed_text = f"{headline}\n\n{summary}\n\n{chunk.text}"

    return EnrichedChunk(
        chunk_id=chunk.chunk_id,
        source_file=chunk.source_file,
        doc_category=chunk.doc_category,
        headline=headline,
        summary=summary,
        original_text=chunk.text,
        embed_text=embed_text,
        token_count=count_tokens(embed_text),
        chunk_index=chunk.chunk_index,
        total_chunks=chunk.total_chunks,
    )


def enrich_all_chunks(
    raw_chunks: list[RawChunk],
    dry_run: bool = False,
) -> list[EnrichedChunk]:
    """Enrich all chunks with LLM-generated headline + summary."""
    if dry_run:
        print("  [dry-run] Skipping LLM enrichment")
        return [
            EnrichedChunk(
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                doc_category=c.doc_category,
                headline=f"[DRY RUN] {c.source_file} chunk {c.chunk_index}",
                summary="[DRY RUN] No enrichment in dry-run mode.",
                original_text=c.text,
                embed_text=c.text,
                token_count=c.token_count,
                chunk_index=c.chunk_index,
                total_chunks=c.total_chunks,
            )
            for c in raw_chunks
        ]

    client = Anthropic()
    enriched = []

    print(f"\n  Enriching {len(raw_chunks)} chunks with {ENRICHMENT_MODEL}...")
    for chunk in tqdm(raw_chunks, desc="  LLM enrichment", unit="chunk"):
        enriched.append(enrich_chunk(chunk, client))
        time.sleep(ENRICH_DELAY_S)

    return enriched


# ── Embedding ──────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed a batch of texts using OpenAI text-embedding-3-small."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_all_chunks(
    enriched_chunks: list[EnrichedChunk],
    dry_run: bool = False,
) -> list[list[float]]:
    """Embed all enriched chunks in batches."""
    if dry_run:
        print("  [dry-run] Skipping embedding")
        return [[0.0] * EMBEDDING_DIM] * len(enriched_chunks)

    client = OpenAI()
    texts = [c.embed_text for c in enriched_chunks]
    embeddings = []

    print(f"\n  Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="  Embedding", unit="batch"):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        embeddings.extend(embed_batch(batch, client))

    return embeddings


# ── Chroma storage ─────────────────────────────────────────────────────────────

def build_metadata(chunk: EnrichedChunk) -> dict:
    """Build Chroma metadata dict — all values must be str/int/float/bool."""
    return {
        "source_file":   chunk.source_file,
        "doc_category":  chunk.doc_category,
        "headline":      chunk.headline,
        "summary":       chunk.summary,
        "chunk_index":   chunk.chunk_index,
        "total_chunks":  chunk.total_chunks,
        "token_count":   chunk.token_count,
    }


def store_in_chroma(
    enriched_chunks: list[EnrichedChunk],
    embeddings: list[list[float]],
    chroma_dir: Path,
    collection_name: str,
    reset: bool = False,
) -> chromadb.Collection:
    """Store enriched chunks and embeddings in Chroma."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"  Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches
    CHROMA_BATCH = 100
    print(f"\n  Storing {len(enriched_chunks)} chunks in Chroma...")
    for i in tqdm(range(0, len(enriched_chunks), CHROMA_BATCH), desc="  Chroma upsert", unit="batch"):
        batch_chunks = enriched_chunks[i : i + CHROMA_BATCH]
        batch_embeds = embeddings[i : i + CHROMA_BATCH]

        collection.upsert(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeds,
            documents=[c.original_text for c in batch_chunks],  # stored text for retrieval
            metadatas=[build_metadata(c) for c in batch_chunks],
        )

    return collection


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CAPA/8D Expert — Knowledge Base Ingestion Pipeline"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing Chroma collection and rebuild from scratch"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and chunk documents without calling LLM or embedding APIs"
    )
    parser.add_argument(
        "--kb-dir", type=Path, default=KNOWLEDGE_BASE_DIR,
        help=f"Knowledge base directory (default: {KNOWLEDGE_BASE_DIR})"
    )
    parser.add_argument(
        "--chroma-dir", type=Path, default=CHROMA_DIR,
        help=f"Chroma persistence directory (default: {CHROMA_DIR})"
    )
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"CAPA/8D Expert — Ingestion Pipeline")
    print(f"{'─'*60}")
    print(f"  KB dir     : {args.kb_dir}")
    print(f"  Chroma dir : {args.chroma_dir}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Chunk size : {CHUNK_SIZE} tokens / {CHUNK_OVERLAP} overlap")
    print(f"  Enrich LLM : {ENRICHMENT_MODEL}")
    print(f"  Embed model: {EMBEDDING_MODEL}")
    print(f"  Mode       : {'DRY RUN' if args.dry_run else 'FULL'}")
    print(f"  Reset      : {args.reset}")
    print(f"{'─'*60}")

    # ── Step 1: Load + chunk ──────────────────────────────────────────────────
    print("\n[1/4] Loading and chunking documents...")
    raw_chunks = load_documents(args.kb_dir)

    # Print per-document stats
    from collections import Counter
    doc_counts = Counter(c.source_file for c in raw_chunks)
    for doc, count in sorted(doc_counts.items()):
        tokens = sum(c.token_count for c in raw_chunks if c.source_file == doc)
        print(f"  {doc:<55} {count:>3} chunks  {tokens:>6} tokens")

    print(f"\n  Total: {len(raw_chunks)} chunks across {len(doc_counts)} documents")
    avg_tokens = sum(c.token_count for c in raw_chunks) / len(raw_chunks)
    print(f"  Avg chunk size: {avg_tokens:.0f} tokens")

    if args.dry_run:
        print("\n[DRY RUN] Showing first 3 chunks:")
        for c in raw_chunks[:3]:
            print(f"\n  --- {c.chunk_id} ---")
            print(f"  {c.text[:200]}...")
        print("\n[DRY RUN] Complete. No API calls made.")
        return

    # ── Step 2: LLM enrichment ────────────────────────────────────────────────
    print("\n[2/4] LLM semantic enrichment (headline + summary per chunk)...")
    enriched_chunks = enrich_all_chunks(raw_chunks, dry_run=args.dry_run)

    # Spot check
    print(f"\n  Sample enrichment (chunk 0):")
    print(f"  Headline : {enriched_chunks[0].headline}")
    print(f"  Summary  : {enriched_chunks[0].summary[:150]}...")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print("\n[3/4] Generating embeddings...")
    embeddings = embed_all_chunks(enriched_chunks, dry_run=args.dry_run)

    # ── Step 4: Store ─────────────────────────────────────────────────────────
    print("\n[4/4] Storing in Chroma...")
    collection = store_in_chroma(
        enriched_chunks,
        embeddings,
        args.chroma_dir,
        COLLECTION_NAME,
        reset=args.reset,
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    final_count = collection.count()
    print(f"\n{'─'*60}")
    print(f"Ingestion complete")
    print(f"  Chunks stored    : {final_count}")
    print(f"  Collection       : {COLLECTION_NAME}")
    print(f"  Chroma path      : {args.chroma_dir}")
    print(f"{'─'*60}\n")

    # Save enrichment manifest for debugging / eval
    manifest_path = args.chroma_dir / "ingest_manifest.json"
    manifest = {
        "collection": COLLECTION_NAME,
        "total_chunks": final_count,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "enrichment_model": ENRICHMENT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "documents": [
            {
                "file": doc,
                "chunks": count,
                "tokens": sum(c.token_count for c in raw_chunks if c.source_file == doc),
            }
            for doc, count in sorted(doc_counts.items())
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved: {manifest_path}\n")


if __name__ == "__main__":
    main()
