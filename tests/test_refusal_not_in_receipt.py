"""A guard's refusal is an instruction to the model, not an action for the owner.

30.09.2026 the receipt under Iris's answer read «🗑 Не трогаю #110, #111 — …»
right above «🗑 Удалила — #110, #111»: the model had simply done the step the
guard asked for (read the diary first).
"""

from logic.response_generator import _receipt, _tool_result_failed
from logic.tools import Refused


def test_a_refusal_counts_as_nothing_done():
    assert _tool_result_failed(Refused("Не трогаю #110 — этой записи я не видел"))
    assert not _tool_result_failed("Удалила — #110: Heute 2 h für Statistik gelernt")


def test_the_receipt_shows_only_what_was_done():
    done = [("delete_diary_entry", {"entry_ids": [110]}, "Удалила — #110: Heute 2 h")]
    assert _receipt(done) == "🗑 Удалила — #110: Heute 2 h"


def test_the_rules_say_who_writes_the_diary():
    """30.09.2026 «зачем ты записал это в дневник?» got an invented purpose."""
    from logic import system_facts
    assert "written by CODE" in system_facts.RULES
    assert "never tie it to him by guess" in system_facts.RULES
