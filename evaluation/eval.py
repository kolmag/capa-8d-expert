"""
eval.py — CAPA/8D Expert RAG Evaluation Pipeline
Metrics: MRR (retrieval), LLM-as-judge (answer quality), source coverage

Pipeline per test case:
  1. Run answer() pipeline → get ranked_chunks + answer text
  2. MRR: check if expected_sources appear in top-k retrieved chunks
  3. LLM judge: score answer on correctness, completeness, groundedness (0-10 each)
  4. Aggregate: mean MRR, mean judge scores, per-category breakdown

Usage:
    uv run evaluation/eval.py
    uv run evaluation/eval.py --tests evaluation/tests.jsonl
    uv run evaluation/eval.py --sample 5        # quick run on 5 random tests
    uv run evaluation/eval.py --no-rewrite      # skip query rewriting for speed
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean

# Load .env
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env if _env.exists() else None)
except ImportError:
    pass

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from answer import answer as run_answer

# ── Config ─────────────────────────────────────────────────────────────────────

JUDGE_MODEL   = "claude-sonnet-4-5"   # stronger model for evaluation
RESULTS_DIR   = Path("evaluation/results")
MRR_K         = 10   # check top-K chunks for source coverage

# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id:               str
    question:         str
    expected_topics:  list[str]
    expected_sources: list[str]
    difficulty:       str
    category:         str
    question_type:    str = "general"   # procedural|conceptual|decision|criteria|factual|regulatory|adversarial
    source:           str = "internal"  # internal | external


@dataclass
class JudgeScores:
    correctness:   float   # 0-10: factually correct against knowledge base
    completeness:  float   # 0-10: covers all key expected topics
    groundedness:  float   # 0-10: answer is supported by retrieved context
    overall:       float   # mean of the three


@dataclass
class EvalResult:
    test_id:          str
    question:         str
    difficulty:       str
    category:         str
    # Retrieval
    mrr_score:        float   # 1/rank of first expected source, 0 if not found
    sources_found:    list[str]
    sources_missing:  list[str]
    top_chunk_score:  float   # reranker score of top chunk
    # Answer quality
    judge:            JudgeScores
    answer_preview:   str     # first 200 chars
    # Meta
    latency_s:        float
    # Optional fields with defaults last
    question_type:    str = "general"
    checker_score:    float = 1.0   # Option 3 groundedness score; 1.0 = skipped/perfect
    error:            str = ""


# ── MRR calculation ────────────────────────────────────────────────────────────

def compute_mrr(
    ranked_chunks: list,
    expected_sources: list[str],
    k: int = MRR_K,
) -> tuple[float, list[str], list[str]]:
    """
    Mean Reciprocal Rank for source coverage.
    Returns (mrr_score, sources_found, sources_missing)
    """
    retrieved_sources = [c.source_file for c in ranked_chunks[:k]]

    found = []
    first_rank = None

    for i, src in enumerate(retrieved_sources, 1):
        for expected in expected_sources:
            if expected in src and expected not in found:
                found.append(expected)
                if first_rank is None:
                    first_rank = i

    missing = [s for s in expected_sources if s not in found]
    mrr = (1.0 / first_rank) if first_rank else 0.0

    return mrr, found, missing


# ── LLM judge ─────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator for a CAPA/8D quality management RAG system.

Score the assistant's answer on three dimensions (0-10 each):

1. CORRECTNESS: Is the answer factually accurate? Does it align with established quality management practices (ISO 9001, IATF 16949, 8D methodology)?
   - 9-10: Completely accurate, no errors
   - 7-8: Mostly accurate, minor imprecisions
   - 5-6: Partially accurate, some errors
   - 3-4: Significant errors present
   - 0-2: Mostly wrong or misleading

2. COMPLETENESS: Does the answer cover the key topics expected for this question?
   - 9-10: Covers all expected topics thoroughly
   - 7-8: Covers most expected topics
   - 5-6: Covers some but misses important topics
   - 3-4: Superficial, misses most key points
   - 0-2: Barely addresses the question

3. GROUNDEDNESS: Is the answer supported by the retrieved context provided? Does it stay within what the context supports?
   - 9-10: Fully grounded, all claims traceable to context
   - 7-8: Mostly grounded, minor extrapolations
   - 5-6: Partially grounded, some unsupported claims
   - 3-4: Many claims not in context
   - 0-2: Answer ignores or contradicts context

Return ONLY valid JSON: {"correctness": N, "completeness": N, "groundedness": N, "reasoning": "one sentence"}"""

JUDGE_PROMPT = """QUESTION: {question}

EXPECTED TOPICS (what a good answer should cover): {expected_topics}

RETRIEVED CONTEXT (what the system had access to):
{context}

ASSISTANT'S ANSWER:
{answer}

Score this answer. Return JSON only."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def judge_answer(
    question: str,
    expected_topics: list[str],
    context_chunks: list,
    answer_text: str,
    client: Anthropic,
    is_adversarial: bool = False,
) -> JudgeScores:
    """Score an answer using Claude Sonnet as judge."""

    # Build context summary for judge (first 300 chars per chunk)
    context_parts = []
    for i, chunk in enumerate(context_chunks[:5], 1):  # top 5 chunks only
        context_parts.append(f"[{i}] {chunk.headline}: {chunk.original_text[:300]}...")
    context_str = "\n\n".join(context_parts)

    if is_adversarial:
        prompt = ADVERSARIAL_JUDGE_PROMPT.format(
            question=question,
            answer=answer_text[:1500],
        )
    else:
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected_topics=", ".join(expected_topics),
            context=context_str,
            answer=answer_text[:1500],
        )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    parsed = json.loads(raw)
    c = float(parsed.get("correctness", 0))
    co = float(parsed.get("completeness", 0))
    g = float(parsed.get("groundedness", 0))

    return JudgeScores(
        correctness=c,
        completeness=co,
        groundedness=g,
        overall=mean([c, co, g]),
    )


# ── Single test evaluation ─────────────────────────────────────────────────────

def evaluate_one(
    test: TestCase,
    use_rewrite: bool,
    judge_client: Anthropic,
) -> EvalResult:
    """Run a single test case through the full eval pipeline."""
    start = time.time()

    try:
        result = run_answer(
            question=test.question,
            use_rewrite=use_rewrite,
            debug=False,
        )

        mrr, found, missing = compute_mrr(
            result.ranked_chunks,
            test.expected_sources,
        )

        top_score = result.ranked_chunks[0].relevance_score if result.ranked_chunks else 0.0

        is_adv = test.question_type == "adversarial" or test.category == "out_of_scope"
        judge = judge_answer(
            question=test.question,
            expected_topics=test.expected_topics,
            context_chunks=result.ranked_chunks,
            answer_text=result.answer,
            client=judge_client,
            is_adversarial=is_adv,
        )

        return EvalResult(
            test_id=test.id,
            question=test.question,
            difficulty=test.difficulty,
            category=test.category,
            mrr_score=mrr,
            sources_found=found,
            sources_missing=missing,
            top_chunk_score=top_score,
            judge=judge,
            answer_preview=result.answer[:200],
            latency_s=time.time() - start,
            question_type=getattr(test, 'question_type', 'general'),
            checker_score=getattr(result, 'checker_score', 1.0),
        )

    except Exception as e:
        return EvalResult(
            test_id=test.id,
            question=test.question,
            difficulty=test.difficulty,
            category=test.category,
            mrr_score=0.0,
            sources_found=[],
            sources_missing=test.expected_sources,
            top_chunk_score=0.0,
            judge=JudgeScores(0, 0, 0, 0),
            answer_preview="",
            latency_s=time.time() - start,
            question_type=getattr(test, 'question_type', 'general'),
            error=str(e),
        )


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def compute_aggregates(results: list[EvalResult]) -> dict:
    """Compute aggregate metrics across all results."""
    valid = [r for r in results if not r.error]

    if not valid:
        return {"error": "No valid results"}

    # Overall metrics
    agg = {
        "total": len(results),
        "valid": len(valid),
        "failed": len(results) - len(valid),
        "mean_mrr": mean(r.mrr_score for r in valid),
        "mean_correctness": mean(r.judge.correctness for r in valid),
        "mean_completeness": mean(r.judge.completeness for r in valid),
        "mean_groundedness": mean(r.judge.groundedness for r in valid),
        "mean_overall": mean(r.judge.overall for r in valid),
        "mean_top_chunk_score": mean(r.top_chunk_score for r in valid),
        "mean_latency_s": mean(r.latency_s for r in valid),
        "source_coverage_rate": mean(1.0 if not r.sources_missing else 0.0 for r in valid),
        # Option 3 groundedness checker metrics
        "mean_checker_score": mean(r.checker_score for r in valid),
        "checker_fired_rate": mean(1.0 if r.checker_score < 1.0 else 0.0 for r in valid),
        "checker_fired_count": sum(1 for r in valid if r.checker_score < 1.0),
    }

    # By difficulty
    for diff in ["basic", "intermediate", "advanced"]:
        subset = [r for r in valid if r.difficulty == diff]
        if subset:
            agg[f"mrr_{diff}"] = mean(r.mrr_score for r in subset)
            agg[f"overall_{diff}"] = mean(r.judge.overall for r in subset)

    # By category
    categories = list(set(r.category for r in valid))
    agg["by_category"] = {}
    for cat in sorted(categories):
        subset = [r for r in valid if r.category == cat]
        agg["by_category"][cat] = {
            "n": len(subset),
            "mrr": round(mean(r.mrr_score for r in subset), 3),
            "overall": round(mean(r.judge.overall for r in subset), 2),
        }

    # By question type
    qtypes = list(set(getattr(r, 'question_type', 'general') for r in valid))
    agg["by_question_type"] = {}
    for qt in sorted(qtypes):
        subset = [r for r in valid if getattr(r, 'question_type', 'general') == qt]
        if subset:
            agg["by_question_type"][qt] = {
                "n": len(subset),
                "mrr": round(mean(r.mrr_score for r in subset), 3),
                "overall": round(mean(r.judge.overall for r in subset), 2),
            }

    return agg


# ── Report printing ────────────────────────────────────────────────────────────

def print_report(results: list[EvalResult], agg: dict):
    """Print a formatted evaluation report to stdout."""
    print(f"\n{'═'*65}")
    print(f"  CAPA/8D Expert — Evaluation Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*65}")

    print(f"\n{'─'*65}")
    print(f"  OVERALL METRICS  ({agg['valid']}/{agg['total']} tests passed)")
    print(f"{'─'*65}")
    print(f"  MRR (source retrieval)   : {agg['mean_mrr']:.3f}")
    print(f"  Source coverage rate     : {agg['source_coverage_rate']:.1%}")
    print(f"  Judge — Correctness      : {agg['mean_correctness']:.2f}/10")
    print(f"  Judge — Completeness     : {agg['mean_completeness']:.2f}/10")
    print(f"  Judge — Groundedness     : {agg['mean_groundedness']:.2f}/10")
    print(f"  Judge — Overall          : {agg['mean_overall']:.2f}/10")
    print(f"  Mean top chunk score     : {agg['mean_top_chunk_score']:.2f}/10")
    print(f"  Mean latency             : {agg['mean_latency_s']:.1f}s")
    print(f"  Checker mean score       : {agg['mean_checker_score']:.3f}  (1.0 = skipped)")
    print(f"  Checker fired            : {agg['checker_fired_count']}/{agg['total']} tests ({agg['checker_fired_rate']:.0%})")

    print(f"\n{'─'*65}")
    print(f"  BY DIFFICULTY")
    print(f"{'─'*65}")
    for diff in ["basic", "intermediate", "advanced"]:
        if f"mrr_{diff}" in agg:
            print(f"  {diff:<14} MRR={agg[f'mrr_{diff}']:.3f}  Overall={agg[f'overall_{diff}']:.2f}/10")

    print(f"\n{'─'*65}")
    print(f"  BY CATEGORY")
    print(f"{'─'*65}")
    for cat, metrics in agg["by_category"].items():
        print(f"  {cat:<22} n={metrics['n']}  MRR={metrics['mrr']:.3f}  Overall={metrics['overall']:.2f}/10")

    if "by_question_type" in agg:
        print(f"\n{'─'*65}")
        print(f"  BY QUESTION TYPE")
        print(f"{'─'*65}")
        for qt, metrics in agg["by_question_type"].items():
            print(f"  {qt:<18} n={metrics['n']}  MRR={metrics['mrr']:.3f}  Overall={metrics['overall']:.2f}/10")

    print(f"\n{'─'*65}")
    print(f"  PER-TEST RESULTS")
    print(f"{'─'*65}")
    for r in results:
        status = "✓" if not r.error else "✗"
        missing = f" [missing: {', '.join(r.sources_missing)}]" if r.sources_missing else ""
        print(
            f"  {status} {r.test_id} | MRR={r.mrr_score:.2f} | "
            f"Overall={r.judge.overall:.1f} | "
            f"{r.difficulty:<14} {r.category}{missing}"
        )
        if r.error:
            print(f"    ERROR: {r.error}")

    print(f"\n{'═'*65}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAPA/8D Expert — Evaluation Pipeline")
    parser.add_argument(
        "--tests", type=Path,
        default=Path("evaluation/tests_v2.jsonl"),
        help="Path to tests jsonl (default: tests_v2.jsonl)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Run only N random tests (for quick checks)",
    )
    parser.add_argument(
        "--no-rewrite", action="store_true",
        help="Skip query rewriting (faster, lower quality)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Save results JSON to this path",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Run only tests in this category",
    )
    args = parser.parse_args()

    # Load tests
    if not args.tests.exists():
        print(f"✗ Test file not found: {args.tests}")
        sys.exit(1)

    tests = []
    with args.tests.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                tests.append(TestCase(**d))

    # Filter by category
    if args.category:
        tests = [t for t in tests if t.category == args.category]
        print(f"Filtered to category '{args.category}': {len(tests)} tests")

    # Sample
    if args.sample:
        tests = random.sample(tests, min(args.sample, len(tests)))
        print(f"Sampled {len(tests)} tests")

    print(f"\n{'─'*65}")
    print(f"  CAPA/8D Expert — Running Evaluation")
    print(f"  Tests     : {len(tests)}")
    print(f"  Judge     : {JUDGE_MODEL}")
    print(f"  Rewrites  : {'off' if args.no_rewrite else 'on'}")
    print(f"{'─'*65}\n")

    judge_client = Anthropic()
    results = []

    for test in tqdm(tests, desc="Evaluating", unit="test"):
        result = evaluate_one(
            test=test,
            use_rewrite=not args.no_rewrite,
            judge_client=judge_client,
        )
        results.append(result)

        # Print quick result inline
        status = "✓" if not result.error else "✗"
        qtype = f"[{test.question_type}]" if hasattr(test, 'question_type') else ""
        print(
            f"  {status} {test.id} | MRR={result.mrr_score:.2f} | "
            f"Overall={result.judge.overall:.1f}/10 | "
            f"{test.difficulty:<14} {qtype} {result.latency_s:.1f}s"
        )

    # Aggregate
    agg = compute_aggregates(results)
    print_report(results, agg)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or RESULTS_DIR / f"eval_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "judge_model": JUDGE_MODEL,
            "use_rewrite": not args.no_rewrite,
            "n_tests": len(tests),
        },
        "aggregates": agg,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
