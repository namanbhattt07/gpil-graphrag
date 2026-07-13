"""
config/ package.

Holds everything related to configuration and secrets:
- settings.py   -> loads API keys and app settings from the .env file
- settings.yaml -> (added in Phase 5) GraphRAG's own indexing/query config

Nothing else in the project should read `os.environ` directly for secrets —
always go through `config.settings.get_settings()` so there is exactly one
place that knows how configuration is loaded.
"""
