"""Load environment variables from a local .env file (secrets stay out of source).

Import and call ``load_env()`` once at every process entry point (app, CLI, job
runner). Values already set in the real environment win over the .env file, so shell
exports and secret managers keep priority. If python-dotenv is not installed the call
is a harmless no-op and the app falls back to the process environment.
"""
from __future__ import annotations

_LOADED = False


def load_env() -> bool:
    """Load .env into os.environ (without overriding existing vars). Idempotent."""
    global _LOADED
    if _LOADED:
        return True
    try:
        from dotenv import load_dotenv, find_dotenv
    except Exception:
        _LOADED = True
        return False
    load_dotenv(find_dotenv(usecwd=True), override=False)
    _LOADED = True
    return True
