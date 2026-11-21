"""Full pipeline: ingest -> judge runner -> scorecard, on one real fixture.

Marked integration_test: needs a real provider key to actually run.
Skips (not fails) when none is configured, since CI's integration job
currently has no secrets set up; this must not turn it red.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from aixeval.ingest.langfuse import ingest_langfuse
from aixeval.ingest.load import load_json_object
from aixeval.judge.config import build_client_from_env
from aixeval.judge.evidence import extract_evidence_langfuse
from aixeval.judge.runner import run_grounding_judge
from aixeval.scorecard.grounding import score_grounding


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
_FIXTURE = Path(__file__).parent.parent / "fixtures" / "langfuse_gist.json"


@pytest.mark.integration_test
def test_ingest_then_judge_then_score_on_one_fixture() -> None:
    """A real Langfuse-shaped fixture: ingested, judged live, then scored.

    Not asserting a specific score: an LLM judge's exact number can
    vary by model and run. What this proves is that ingest's output and
    the judge runner's input actually compose, end to end, on real data.
    """
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        pytest.skip("No provider API key configured; set one in .env to run this live.")

    record = ingest_langfuse(_FIXTURE)
    payload = load_json_object(_FIXTURE)
    client = build_client_from_env()

    new_record, attachment = run_grounding_judge(payload, record, client, extract_evidence_langfuse)

    assert attachment["status"] == "scored", attachment.get("reason")
    assert attachment["model"] == client.model
    assert attachment["raw_reply"]
    assert 0.0 <= new_record["grounding_judge"] <= 1.0

    result = score_grounding(new_record)
    assert result["status"] == "scored"
    assert 0 <= result["grade"] <= 4
