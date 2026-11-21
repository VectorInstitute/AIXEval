"""run_cross_modal_hallucination_judge with a real image_path, against a real provider.

Marked integration_test: needs a real provider key to actually run.
Skips (not fails) when none is configured, same discipline as
test_judge_integration.py. Unlike that test, this one does assert a
score direction: the whole point of image_path mode is that the judge
reads actual pixels via complete_with_image, not a text proxy, so a
faithful vs. hallucinated description of the SAME image diverging is
the one thing worth proving live. Thresholds are loose (<=0.3/>=0.7),
not exact, to tolerate ordinary judge-to-judge variance.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from aixeval.judge.config import build_client_from_env
from aixeval.judge.runner import run_cross_modal_hallucination_judge


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
_FIXTURE_IMAGE = Path(__file__).parent.parent / "fixtures" / "cross_modal_hallucination_test_image.jpg"

# The fixture is a synthetic scene: light-blue background, a brown
# ellipse, a tan strip along the bottom, a yellow-gold ellipse above
# the brown one. Deliberately abstract shapes, not a real photo --
# keeps the fixture tiny and avoids needing any real image rights.
_FAITHFUL_OUTPUT = (
    "A simple scene with a brown shape and a yellow-gold circle above a tan strip, against a light blue background."
)
_HALLUCINATED_OUTPUT = "A golden retriever and a small cat are playing together on a sandy beach under a palm tree."


def _skip_if_no_provider_key() -> None:
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        pytest.skip("No provider API key configured; set one in .env to run this live.")


@pytest.mark.integration_test
def test_faithful_description_of_the_real_image_scores_low() -> None:
    _skip_if_no_provider_key()
    client = build_client_from_env()
    record = {"image_path": str(_FIXTURE_IMAGE), "output": _FAITHFUL_OUTPUT}

    new_record, attachment = run_cross_modal_hallucination_judge(record, client)

    assert attachment["status"] == "scored", attachment.get("reason")
    assert attachment["model"] == client.model
    assert attachment["raw_reply"]
    assert new_record["cross_modal_hallucination_judge"] <= 0.3


@pytest.mark.integration_test
def test_hallucinated_description_of_the_same_real_image_scores_high() -> None:
    _skip_if_no_provider_key()
    client = build_client_from_env()
    record = {"image_path": str(_FIXTURE_IMAGE), "output": _HALLUCINATED_OUTPUT}

    new_record, attachment = run_cross_modal_hallucination_judge(record, client)

    assert attachment["status"] == "scored", attachment.get("reason")
    assert new_record["cross_modal_hallucination_judge"] >= 0.7
