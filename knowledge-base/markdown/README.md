# Knowledge Base

This directory contains the domain-specific documents that power the RAG pipeline.

## Structure
Each document should be a `.md` file with clear section headings.
The ingest pipeline will chunk, enrich with LLM-generated headlines, and embed them automatically.

## To use your own knowledge base
1. Add your `.md` files to this directory
2. Run: `uv run scripts/ingest.py --reset`

## Documents used in evaluation
The eval results in this repo were produced against a 12-document CAPA/8D
quality management knowledge base (~22,000 words) covering:
- 8D methodology (D0–D8)
- CAPA procedures (ISO 9001, IATF 16949)
- Root cause analysis tools
- FMEA and Control Plans
- Industry compliance standards
```

**Update `.gitignore`:**
```
# Knowledge base documents — provide your own
knowledge-base/markdown/*.md