"""Pick a judge provider + model from the environment (and a local .env)."""

from __future__ import annotations

import os
from collections.abc import Callable

from dotenv import load_dotenv

from aixeval.judge.client import ClaudeClient
from aixeval.judge.gemini_client import GeminiClient
from aixeval.judge.openai_client import OpenAIClient
from aixeval.judge.provider import Provider
from aixeval.judge.vllm_client import VLLMClient


PROVIDER_ENV_VAR = "JUDGE_PROVIDER"
DEFAULT_PROVIDER = "anthropic"

_MODEL_ENV_VARS: dict[str, str] = {
    "anthropic": "JUDGE_MODEL_ANTHROPIC",
    "openai": "JUDGE_MODEL_OPENAI",
    "gemini": "JUDGE_MODEL_GEMINI",
    "vllm": "JUDGE_MODEL_VLLM",
}
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.5",
    "gemini": "gemini-flash-latest",
    # "vllm" deliberately has no entry here: a self-hosted server could be
    # running anything, so JUDGE_MODEL_VLLM must always be set explicitly
    # rather than silently defaulting to a model that likely isn't deployed.
}
_BUILDERS: dict[str, Callable[[str], Provider]] = {
    "anthropic": lambda model: ClaudeClient(model=model),
    "openai": lambda model: OpenAIClient(model=model),
    "gemini": lambda model: GeminiClient(model=model),
    "vllm": lambda model: VLLMClient(model=model),
}


def available_providers() -> tuple[str, ...]:
    """Every provider name ``build_client`` accepts, in a stable order."""
    return tuple(sorted(_BUILDERS))


def provider_from_env() -> str:
    """Return the provider ``JUDGE_PROVIDER`` names after loading ``.env``.

    Not validated: an unknown name is returned as-is so the caller's
    ``build_client`` raises the usual "Unknown JUDGE_PROVIDER" error.
    """
    load_dotenv()
    return os.environ.get(PROVIDER_ENV_VAR, DEFAULT_PROVIDER).strip().lower()


def default_model_for(provider: str) -> str | None:
    """Return the model this provider uses when none is given.

    Reads its ``JUDGE_MODEL_*`` env var, else the built-in default.
    ``None`` for a provider with no default (``vllm``) whose env var is
    unset, or for an unknown provider.
    """
    load_dotenv()
    env_var = _MODEL_ENV_VARS.get(provider)
    if env_var is None:
        return None
    return os.environ.get(env_var) or _DEFAULT_MODELS.get(provider)


def build_client(provider: str, model: str | None = None) -> Provider:
    """Build a judge client for an explicitly chosen provider and model.

    Parameters
    ----------
    provider
        One of ``available_providers()``; case and surrounding spaces
        are ignored.
    model
        The model id to send. ``None`` or blank falls back to
        ``default_model_for(provider)``.

    Raises
    ------
    ValueError
        ``provider`` is not one this toolkit knows, or it has no
        default model (``vllm``) and none was given or set in its env var.
    NotImplementedError
        The named provider is known but has no client wired up
        (defensive only; every declared provider currently is).
    """
    provider = provider.strip().lower()
    if provider not in _MODEL_ENV_VARS:
        known = ", ".join(sorted(_MODEL_ENV_VARS))
        msg = f"Unknown {PROVIDER_ENV_VAR}={provider!r}. Known providers: {known}."
        raise ValueError(msg)
    build = _BUILDERS.get(provider)
    if build is None:
        msg = f"Provider {provider!r} is declared but has no client yet."
        raise NotImplementedError(msg)
    chosen = (model or "").strip() or default_model_for(provider)
    if not chosen:
        model_env_var = _MODEL_ENV_VARS[provider]
        msg = f"Set {model_env_var} to name the model your {provider} server is serving; there is no default."
        raise ValueError(msg)
    return build(chosen)


def build_client_from_env() -> Provider:
    """Build a judge client from ``JUDGE_PROVIDER`` + ``JUDGE_MODEL_*``.

    Loads a local ``.env`` file first if one is present (does not
    overwrite variables already set in the real environment), then reads
    ``JUDGE_PROVIDER`` (default ``anthropic``) and that provider's model
    variable (default per provider if unset).

    Raises
    ------
    ValueError
        ``JUDGE_PROVIDER`` names a provider this toolkit does not know,
        or names ``vllm`` without a ``JUDGE_MODEL_VLLM`` set (the one
        provider with no sensible default model).
    NotImplementedError
        The named provider is known but has no client wired up
        (defensive only; anthropic, openai, gemini, and vllm all are).
    """
    return build_client(provider_from_env())


def print_config_examples() -> None:
    """Print which provider/model would be selected under a few env states."""
    print(f"{PROVIDER_ENV_VAR} unset -> default provider {DEFAULT_PROVIDER!r}.")
    print("Each provider's model comes from its own env var, else a default:")
    for name, env_var in sorted(_MODEL_ENV_VARS.items()):
        wired = "wired up" if name in _BUILDERS else "not wired up yet"
        default = f"default {_DEFAULT_MODELS[name]!r}" if name in _DEFAULT_MODELS else "no default, must be set"
        print(f"  {name}: {env_var} ({default}) - {wired}")


if __name__ == "__main__":
    print_config_examples()
