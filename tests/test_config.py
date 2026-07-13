"""
Sanity tests for config/settings.py.

These don't call any real API or need a real key. They exist to prove three
things about our config-loading mechanism (Phase 1's actual "Definition of
Done"):
  1. Secrets come from the environment/.env, never from a hard-coded string.
  2. When a variable is missing, we get a safe blank/default instead of a
     crash - later phases decide what "missing key" should mean for them.
  3. Defaults (like the random seed) are stable across calls, which matters
     for reproducibility later in Phase 2.

Run with:  ./venv/bin/pytest   (from the project root)
"""

from config.settings import get_settings


def test_get_settings_reads_env_var(monkeypatch):
    """If OPENAI_API_KEY is set in the environment, get_settings() must pick it up verbatim."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    settings = get_settings()
    assert settings.openai_api_key == "test-key-123"


def test_get_settings_has_no_hardcoded_key(monkeypatch):
    """With no env var set at all, the key must come back blank - never a real-looking fallback."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = get_settings()
    assert settings.openai_api_key == ""


def test_get_settings_default_seed_is_stable(monkeypatch):
    """Two calls with no RANDOM_SEED override must agree, so data generation stays reproducible."""
    monkeypatch.delenv("RANDOM_SEED", raising=False)
    first = get_settings()
    second = get_settings()
    assert first.random_seed == second.random_seed


def test_get_settings_seed_is_overridable(monkeypatch):
    """Setting RANDOM_SEED in the environment must change the parsed integer seed."""
    monkeypatch.setenv("RANDOM_SEED", "7")
    settings = get_settings()
    assert settings.random_seed == 7
