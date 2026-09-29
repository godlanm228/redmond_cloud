"""logic/understanding: what code accepts from the model's reading of a message.

The model reads; code checks. A fact, a command or an urgency is accepted only
with a quote that really is in the owner's message.
"""

import json

from logic import understanding as und

PLAN = "Довести до ума и презентабельности вид редмонд клауда , тоесть вас и подготовиться к собеседованию"
HOSPITAL = "В больнице лежу , мут до первого числа"


def _raw(**kw):
    base = {"addressee": "Iris", "research": False, "about": "", "refers_to": "",
            "facts": [], "commands": [], "urgency": "none", "urgency_quote": ""}
    base.update(kw)
    return "```json\n" + json.dumps(base, ensure_ascii=False) + "\n```"


def test_quotes_survive_case_punctuation_and_line_breaks():
    msg = "Курю кальян с Настей\nЧилю\nНичего не планирую кроме чила"
    assert und.quoted("курю кальян с настей, чилю", msg)
    assert und.quoted("Ничего не планирую", msg)
    assert not und.quoted("курю кальян с Машей", msg)
    assert not und.quoted("", msg)
    assert und.quoted("всё ок", "Всё ок!")


def test_a_fact_the_owner_did_not_write_is_dropped():
    u = und.parse(_raw(facts=[
        {"quote": "Довести до ума и презентабельности вид редмонд клауда", "when": "plan",
         "fact": "Планирует довести Redmond Cloud до презентабельного вида", "topic": "работа"},
        {"quote": "позавтракал", "fact": "Позавтракал", "when": "done", "topic": "еда"},
    ]), PLAN)
    assert [f.fact for f in u.facts] == ["Планирует довести Redmond Cloud до презентабельного вида"]
    assert u.facts[0].when == "plan"
    assert any("Позавтракал" in d for d in u.dropped)


def test_hospital_with_a_mute_reads_as_a_fact_a_command_and_attention():
    u = und.parse(_raw(
        facts=[{"quote": "В больнице лежу", "fact": "Лежит в больнице", "when": "now", "topic": "здоровье"}],
        commands=[{"quote": "мут до первого числа", "type": "mute", "until": "2026-10-01",
                   "hours": None, "scope": "all"}],
        urgency="attention", urgency_quote="В больнице лежу"), HOSPITAL)
    assert u.urgency == "attention"
    assert u.commands[0].until == "2026-10-01" and u.commands[0].scope == "all"
    assert u.facts[0].when == "now"


def test_a_crisis_needs_its_quote():
    u = und.parse(_raw(urgency="crisis", urgency_quote="хочу умереть"), "Устал сегодня")
    assert u.urgency == "attention", "an unquoted crisis must not drive the reply"
    u = und.parse(_raw(urgency="crisis", urgency_quote="Суисайд"), "Суисайд")
    assert u.urgency == "crisis"


def test_nobody_and_unknown_names():
    assert und.parse(_raw(addressee="никто"), "хм").addressee == und.NOBODY
    assert und.parse(_raw(addressee="Alice"), "хм").addressee == ""
    assert und.parse("не JSON", "хм") is None


def test_a_useless_answer_moves_to_the_next_model():
    from utils import llm_gate
    llm_gate.configure(pools={"understand": ["model-a", "model-b"]})
    calls = []

    def complete(model, prompt, **kw):
        calls.append(model)
        return "извини, не понял" if model == "model-a" else _raw(addressee="Redmond")

    u = und.understand("что такое TPM?", now="2026-09-29 20:00, вторник", complete=complete)
    assert calls == ["model-a", "model-b"]
    assert u.addressee == "Redmond" and u.model == "model-b"


def test_the_input_marks_scheduled_messages_as_the_bots():
    text = und.build_input("Это как?", [{"who": "плановое сообщение бота", "text": "Добрый день!"}],
                           "2026-09-29 12:02, вторник")
    assert "плановое сообщение бота: Добрый день!" in text
    assert "НОВОЕ сообщение Влада: «Это как?»" in text
