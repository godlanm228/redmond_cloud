"""config.json must pass its own JSON schema.

The loader treats a schema failure as a warning and carries on, so a field
added to AppConfig and config.json but not to the schema does not break the
start - it just prints a 60-line "Schema validation failed" dump on every
restart (Sep 28, 2026: gemini_thinking_level). This keeps the two in sync.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_config_json_matches_its_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "config" / "schema" / "config_schema.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config" / "config.json").read_text(encoding="utf-8"))
    jsonschema.validate(instance=config, schema=schema)


def test_every_config_field_is_known_to_the_schema():
    """A new AppConfig field must be declared in the schema too."""
    from config.config import AppConfig
    schema = json.loads((ROOT / "config" / "schema" / "config_schema.json").read_text(encoding="utf-8"))
    missing = set(AppConfig.model_fields) - set(schema["properties"])
    # Secrets come from env only and are deliberately kept out of config.json.
    allowed_outside = {name for name in missing if name.endswith(("_api_key", "_token"))}
    assert missing - allowed_outside == set(), f"not in the schema: {sorted(missing - allowed_outside)}"
