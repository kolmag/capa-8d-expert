from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def test_answer_stream_api_exists():
    from capa_8d_expert import answer, answer_stream

    assert callable(answer)
    assert callable(answer_stream)


def test_eval_defaults_to_tracked_v3_tests():
    spec = importlib.util.spec_from_file_location("eval_mod", ROOT / "evaluation" / "eval.py")
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    assert "ADVERSARIAL_JUDGE_PROMPT" in eval_mod.__dict__
    assert Path("evaluation/tests_v3.jsonl").exists()


def test_ingest_skips_markdown_readme():
    from capa_8d_expert import ingest

    docs = ingest.load_documents(ROOT / "knowledge-base" / "markdown")
    assert docs
    assert "README.md" not in {doc.source_file for doc in docs}


def test_stale_standalone_rag_scripts_do_not_reappear():
    assert not (ROOT / "scripts" / "answer_groq.py").exists()
    assert not (ROOT / "scripts" / "answer_original.py").exists()


def test_browser_style_citation_artifacts_are_removed():
    from capa_8d_expert.answer import clean_citation_artifacts

    text = "Is/Is Not narrows the cause space \u30101\u2020L1-L4\u3011 before RCA ."
    assert clean_citation_artifacts(text) == "Is/Is Not narrows the cause space before RCA."
