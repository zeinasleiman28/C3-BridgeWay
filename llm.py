"""
llm.py
------
Creates the LLM client from the settings in .env.

    LLM_PROVIDER=gemini   -> Google Gemini through its OpenAI-compatible endpoint (default)
    LLM_PROVIDER=openai   -> OpenAI
    LLM_MODEL=...         -> optional, overrides the default model

Both providers speak the same OpenAI "chat completions" API, so the rest of the
code uses the openai SDK and doesn't care which one is behind it.

API keys are only ever read from the environment (.env), never written in code.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

PROVIDERS = {
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": None,  # the SDK's default OpenAI URL
        "default_model": "gpt-4o-mini",
    },
}


class LLMConfigError(Exception):
    """Raised when .env is missing a provider setting or an API key."""


def get_llm_settings():
    """Read and check the settings. Returns (provider, model, api_key, base_url)."""
    load_dotenv()  # reads .env into os.environ (does nothing if there is no .env)

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in PROVIDERS:
        raise LLMConfigError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)} (got '{provider}').")
    settings = PROVIDERS[provider]

    api_key = os.getenv(settings["key_env"], "").strip()
    if not api_key or api_key.startswith("your-"):
        raise LLMConfigError(
            f"No API key found for {provider}. Copy .env.example to .env and set {settings['key_env']}."
        )

    model = os.getenv("LLM_MODEL", "").strip() or settings["default_model"]
    return provider, model, api_key, settings["base_url"]


def create_client():
    """Return (client, model) ready for client.chat.completions.create(model=model, ...)."""
    provider, model, api_key, base_url = get_llm_settings()
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=2)
    return client, model
