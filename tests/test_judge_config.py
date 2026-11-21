"""build_client_from_env: JUDGE_PROVIDER + JUDGE_MODEL_* pick a client."""

import pytest

from aixeval.judge.client import ClaudeClient
from aixeval.judge.config import (
    available_providers,
    build_client,
    build_client_from_env,
    default_model_for,
    provider_from_env,
)
from aixeval.judge.gemini_client import GeminiClient
from aixeval.judge.openai_client import OpenAIClient
from aixeval.judge.vllm_client import VLLMClient


@pytest.fixture(autouse=True)
def _no_dotenv_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: never read the repo's real .env."""
    monkeypatch.setattr("aixeval.judge.config.load_dotenv", lambda *a, **kw: False)
    env_vars = (
        "JUDGE_PROVIDER",
        "JUDGE_MODEL_ANTHROPIC",
        "JUDGE_MODEL_OPENAI",
        "JUDGE_MODEL_GEMINI",
        "JUDGE_MODEL_VLLM",
        "VLLM_BASE_URL",
        "VLLM_API_KEY",
    )
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


def test_defaults_to_anthropic_with_default_model() -> None:
    client = build_client_from_env()
    assert isinstance(client, ClaudeClient)
    assert client.model == "claude-sonnet-5"


def test_reads_anthropic_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_MODEL_ANTHROPIC", "claude-opus-5")
    client = build_client_from_env()
    assert isinstance(client, ClaudeClient)
    assert client.model == "claude-opus-5"


def test_explicit_anthropic_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    assert isinstance(build_client_from_env(), ClaudeClient)


def test_unknown_provider_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown JUDGE_PROVIDER"):
        build_client_from_env()


def test_selects_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructing openai.OpenAI() reads the key immediately (unlike
    # anthropic.Anthropic()), so a dummy value is enough: no network
    # call happens until .complete() runs.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    client = build_client_from_env()
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-5.5"


def test_selects_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy")
    monkeypatch.setenv("JUDGE_PROVIDER", "gemini")
    client = build_client_from_env()
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-flash-latest"


def test_provider_name_is_case_and_space_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", " Anthropic ")
    assert isinstance(build_client_from_env(), ClaudeClient)


def test_selects_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "vllm")
    monkeypatch.setenv("JUDGE_MODEL_VLLM", "Qwen2.5-7B-Instruct")
    client = build_client_from_env()
    assert isinstance(client, VLLMClient)
    assert client.model == "Qwen2.5-7B-Instruct"


def test_vllm_without_a_model_set_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unlike every other provider, vllm has no sensible default model:
    # a self-hosted server could be running anything.
    monkeypatch.setenv("JUDGE_PROVIDER", "vllm")
    with pytest.raises(ValueError, match="JUDGE_MODEL_VLLM"):
        build_client_from_env()


def test_vllm_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Construction must not require an API key:
    # vLLM's own server needs one only if started with --api-key.
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_PROVIDER", "vllm")
    monkeypatch.setenv("JUDGE_MODEL_VLLM", "Qwen2.5-7B-Instruct")
    assert isinstance(build_client_from_env(), VLLMClient)


def test_build_client_uses_the_explicit_choice_not_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    client = build_client("vllm", "gemma-3-12b-it")
    assert isinstance(client, VLLMClient)
    assert client.model == "gemma-3-12b-it"


def test_build_client_blank_model_falls_back_to_the_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    client = build_client(" OpenAI ", "  ")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-5.5"


def test_build_client_vllm_without_any_model_raises() -> None:
    with pytest.raises(ValueError, match="JUDGE_MODEL_VLLM"):
        build_client("vllm")


def test_build_client_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Known providers"):
        build_client("not-a-real-provider", "x")


def test_default_model_for_prefers_the_env_var_then_the_builtin_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert default_model_for("gemini") == "gemini-flash-latest"
    assert default_model_for("vllm") is None
    assert default_model_for("not-a-real-provider") is None
    monkeypatch.setenv("JUDGE_MODEL_VLLM", "gemma-3-12b-it")
    assert default_model_for("vllm") == "gemma-3-12b-it"


def test_an_empty_model_env_var_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_MODEL_GEMINI", "")
    assert default_model_for("gemini") == "gemini-flash-latest"


def test_provider_from_env_is_normalized_and_defaults_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    assert provider_from_env() == "anthropic"
    monkeypatch.setenv("JUDGE_PROVIDER", " VLLM ")
    assert provider_from_env() == "vllm"


def test_available_providers_lists_every_wired_provider() -> None:
    assert available_providers() == ("anthropic", "gemini", "openai", "vllm")
