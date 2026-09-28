"""A diary entry must rest on what the owner actually said.

Cases are real: the entries on the left were written to the production diary
next to the messages on the right, found by the Sep 28, 2026 audit.
"""

import pytest

from logic import response_generator as rgm
from logic.intent_recognizer import Intent


def ctx(user_text, history=()):
    return rgm.GenerationContext(intent=Intent(name="chat", slots={}), user_text=user_text,
                                 history=[{"user": u, "bot": ""} for u in history])


@pytest.mark.parametrize("said, entry", [
    ("Мут на 7 дней", "Поел"),                                          # Sep 14
    ("Я УЖЕ ПОТОЕНИРОВАЛСЯ. КАКОЙ НАХУЙ КОДИНГ", "Поел."),              # Jun 15
    ("привет", "Напомнила о необходимости поесть до 17:40"),            # Jun 12: bot's own action
])
def test_entries_the_owner_never_said_are_refused(said, entry):
    refusal = rgm._ungrounded_write("add_diary_entry", {"text": entry}, ctx(said))
    assert refusal and "его же словами" in refusal


@pytest.mark.parametrize("said, entry", [
    ("Работаю", "На работе / работает"),
    ("Проснулся в 12 Поел овсянку Снова болит живот", "Поел овсянку"),
    ("Да С 17 до 23", "Смена с 17 до 23"),
    ("Курю кальян с Настей Чилю", "Курит кальян с Настей, отдыхает"),
])
def test_entries_grounded_in_his_words_pass(said, entry):
    assert rgm._ungrounded_write("add_diary_entry", {"text": entry}, ctx(said)) == ""


def test_the_previous_turn_counts():
    """«Что записала?» after the fact: the fact is one turn back."""
    c = ctx("что записала?", history=["Поел гречку с курицей"])
    assert rgm._ungrounded_write("add_diary_entry", {"text": "Поел гречку"}, c) == ""


def test_code_written_prompts_are_not_the_owners_words():
    """Sep 10: the day-ping prompt made Iris log the ping itself."""
    c = ctx("(scheduled, пинг дня) Влад проснулся недавно (в 12:00) и на связи.")
    assert rgm._ungrounded_write("add_diary_entry", {"text": "Проснулся в 12:00"}, c)


def test_other_tools_are_not_affected():
    assert rgm._ungrounded_write("read_diary", {"last_n": 5}, ctx("привет")) == ""
