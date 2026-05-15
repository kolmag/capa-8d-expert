import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_answer_stream_api_exists():
    sys.path.insert(0, str(SCRIPTS))
    answer = importlib.import_module("answer")

    assert hasattr(answer, "answer")
    assert hasattr(answer, "answer_stream")


def test_eval_defaults_to_tracked_v3_tests():
    sys.path.insert(0, str(ROOT / "evaluation"))
    eval_mod = importlib.import_module("eval")

    assert "ADVERSARIAL_JUDGE_PROMPT" in eval_mod.__dict__
    assert Path("evaluation/tests_v3.jsonl").exists()


def test_ingest_skips_markdown_readme():
    sys.path.insert(0, str(SCRIPTS))
    ingest = importlib.import_module("ingest")

    docs = ingest.load_documents(ROOT / "knowledge-base" / "markdown")
    assert docs
    assert "README.md" not in {doc.source_file for doc in docs}
