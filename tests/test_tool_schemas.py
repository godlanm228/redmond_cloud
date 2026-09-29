"""Tool schemas as the model sees them, and the mute's end date.

Sep 29, 2026 (scenario run): «В больнице лежу , мут до первого числа» → Groq
400 «/hours: expected number, but got null». The model filled every optional
field, with null where it had nothing; mode and scope allowed null, days and
hours did not, and the model also had to turn «до первого числа» into hours.
"""

from datetime import datetime

import pytest

from logic import coach_storage, toolbox
from logic.tools import execute_tool
from utils.time import OWNER_TZ, set_clock


def _walk(obj, path=""):
    required = set(obj.get("required") or [])
    for name, prop in (obj.get("properties") or {}).items():
        yield f"{path}{name}", prop, name in required
        if prop.get("type") == "object" or "object" in (prop.get("type") or []):
            yield from _walk(prop, f"{path}{name}.")


def test_every_optional_parameter_accepts_null():
    offenders = []
    for schema in toolbox.model_tools(None):
        fn = schema["function"]
        for path, prop, required in _walk(fn.get("parameters") or {}):
            t = prop.get("type")
            types = t if isinstance(t, list) else [t]
            if not required and t is not None and "null" not in types:
                offenders.append(f"{fn['name']}.{path}")
    assert not offenders, offenders


@pytest.fixture
def at_1644():
    set_clock(lambda: datetime(2026, 9, 29, 16, 44, tzinfo=OWNER_TZ))
    yield
    set_clock(None)


def test_the_run_call_with_nulls_now_works(at_1644):
    execute_tool("mute_notifications", {"mode": None, "days": None, "hours": None,
                                        "scope": "all", "until": "2026-10-01"})
    info = coach_storage.mute_info()
    assert info["scope"] == "all"
    assert info["until"].startswith("2026-10-01T00:00")


def test_an_end_time_is_kept_exactly(at_1644):
    out = execute_tool("mute_notifications", {"until": "2026-10-01T16:45", "scope": "all"})
    assert coach_storage.mute_info()["until"].startswith("2026-10-01T16:45")
    assert "01.10 16:45" in out


def test_a_past_end_date_is_questioned_not_applied(at_1644):
    out = execute_tool("mute_notifications", {"until": "2026-09-01"})
    assert "прошла" in out
    assert coach_storage.mute_info() is None


def test_nothing_given_is_the_old_short_snooze(at_1644):
    execute_tool("mute_notifications", {"mode": None, "days": None, "hours": None, "scope": None})
    assert coach_storage.mute_info()["until"].startswith("2026-09-29T18:44")
