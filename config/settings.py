"""
Central place for loading configuration and secrets for the whole project.

WHY THIS FILE EXISTS:
Every phase from here on (data generation's random seed, GraphRAG's API
calls, the inference layer's model choice) needs some piece of configuration.
Instead of every module calling `os.getenv(...)` on its own (easy to typo a
variable name, easy to accidentally hard-code a real key while testing),
every other module should import `get_settings()` from here. That keeps all
secret/config handling in exactly one place.

HOW IT WORKS:
`load_dotenv()` reads the `.env` file at the project root (if present) and
copies its key=value pairs into the process environment. `get_settings()`
then reads that environment into a typed `Settings` object. If `.env` is
missing entirely (e.g. a fresh clone of the repo before anyone has copied
`.env.example`), nothing crashes here — values just come back as empty
strings/defaults, and it's up to whichever phase actually needs the key
(Phase 5 indexing, Phase 7 inference) to check for that and fail with a
clear message at the point of use.
"""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# This file lives at <project_root>/config/settings.py, so its grandparent
# directory is the project root. Computing it this way (instead of assuming
# the current working directory) means get_settings() works correctly no
# matter which folder you run a script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load key=value pairs from .env into the process environment.
# override=False means: if a variable is already set in the real shell
# environment (e.g. exported before running, or set by CI), that real value
# wins over whatever is in the .env file.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """
    Read-only bundle of every configuration value the project needs.

    Using a dataclass instead of scattering `os.getenv()` calls everywhere
    means: every available setting is documented in one place, a typo'd
    environment variable name fails fast here instead of quietly returning
    None deep inside Phase 7, and tests can build a Settings object by hand
    without touching real environment variables at all.
    """

    openai_api_key: str       # secret - never print/log this value
    openai_api_base: str      # lets us point at Azure OpenAI or another provider later
    llm_model: str            # chat model used for extraction (Phase 5) and reasoning (Phase 7)
    embedding_model: str      # embedding model used for GraphRAG's vector store
    random_seed: int          # fixed seed so synthetic data (Phase 2) is reproducible
    project_root: Path        # absolute path to the repo root, handy for building file paths


def get_settings() -> Settings:
    """
    Read environment variables (populated from .env by the load_dotenv()
    call above) and return them as a Settings object.

    This intentionally does NOT raise an error if OPENAI_API_KEY is blank.
    Phase 1's job is only to prove the *loading mechanism* works end to end
    without any hard-coded secret. Later phases that actually call the LLM
    are responsible for checking `settings.openai_api_key` themselves and
    failing with a clear, specific error message at that point.
    """
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        project_root=PROJECT_ROOT,
    )
