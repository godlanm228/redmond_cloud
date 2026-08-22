"""Целостность контекста и честность действий.

Каждый тест здесь воспроизводит происшествие из боевого лога 15–20.08.2026,
а не гипотезу. Все они КРАСНЫЕ на коде от 21.08.2026 — это зафиксированный
разрыв между тем, что хаб сообщает, и тем, что он делает.

Общий класс дефекта у всех: действие рапортует об успехе, и никто не сверяет,
что оно совпало с намерением.

Незакрытые помечены xfail(strict=True) (по мере починки метки снимаются —
22.08 сняты четыре: сбои памяти стали слышны и отказ перестал
выдавать минутный лимит за суточный). Это не «отключено», а зафиксировано:
тест выполняется, его падение ожидаемо и не красит прогон, но как только
дефект починят — XPASS уронит набор и заставит снять метку. Забыть про них
нельзя по построению.
"""

import logging

import pytest

from logic import coach_storage, response_generator as rg_mod, tools
from logic.intent_recognizer import Intent
from logic.response_generator import GenerationContext, ResponseGenerator
from utils import db

CHAT = -1001234567890


def _ctx(user_text="", history=None, docs=None):
    return GenerationContext(
        intent=Intent(name="chat", slots={}),
        user_text=user_text,
        history=history or [],
        retrieved_docs=docs or [],
    )


def _bare_rg(**attrs):
    """ResponseGenerator без __init__: тестируем метод, а не сборку объекта."""
    import threading

    rg = object.__new__(ResponseGenerator)
    rg.mem = None
    rg.top_k = 3
    rg.max_history = 6
    rg.history_by_chat = {}
    rg._history_guard = threading.RLock()
    rg._history_loaded = set()
    for k, v in attrs.items():
        setattr(rg, k, v)
    return rg


# --------------------------------------------------------------------------
# 1. Идентификаторы обязаны переживать сжатие выдачи инструмента.
#    Инцидент 17.08 10:03: список дневника был срезан до 400 символов,
#    id пропали, модель назвала порядковый номер.
# --------------------------------------------------------------------------

def test_compression_keeps_record_ids():
    lines = ["Последние записи (10):"]
    for i in range(80, 90):
        lines.append(f"  #{i} 2026-08-17T12:0{i % 10}+02:00 [спорт]: "
                     f"Запись номер {i}, достаточно длинная чтобы выдача "
                     f"инструмента заведомо перевалила лимит сжатия.")
    text = "\n".join(lines)
    assert len(text) > 400, "заготовка должна быть длиннее порога сжатия"

    # Как в бою: сжатие спрашивает у инструмента, что существенно.
    compressed = rg_mod._compress_tool_content(
        text, essentials=tools.OUTPUT_ESSENTIALS["read_diary"])

    missing = [i for i in range(80, 90) if f"#{i}" not in compressed]
    assert not missing, (
        f"сжатие потеряло id {missing}. Модель не сможет сослаться на запись, "
        f"которую только что прочитала, и назовёт порядковый номер"
    )


# --------------------------------------------------------------------------
# 2. Деструктивное действие обязано отчитываться содержимым, а не номером.
#    17.08 Айрис сказала «Удалила записи: #3» — правду о неверном действии.
# --------------------------------------------------------------------------

def test_delete_reports_what_was_actually_removed():
    entry = coach_storage.add_diary_entry("Позанимался спортом, начал креатин")
    assert entry, "запись должна была создаться"

    result = tools._tool_delete_diary_entry({"entry_ids": [entry["id"]]})

    assert "спорт" in result.lower(), (
        f"отчёт об удалении не содержит текста удалённого: {result!r}. "
        f"Владелец не может заметить, что снесли не то"
    )


# --------------------------------------------------------------------------
# 3. Удалять можно только то, что модель действительно видела.
#    Точное воспроизведение 17.08: потеряв листинг, модель называет «3»,
#    и уходит непричастная запись месячной давности.
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="К2: деструктивный инструмент принимает id, который модель не могла узнать")
def test_stale_index_does_not_destroy_unrelated_entry():
    ids = []
    for n in range(1, 6):
        e = coach_storage.add_diary_entry(f"Старая запись номер {n}")
        ids.append(e["id"])
    victim = ids[2]  # то, что модель назовёт «3»

    tools._tool_delete_diary_entry({"entry_ids": [3]})

    survived = coach_storage.read_diary(last_n=50)
    assert any(e["id"] == victim for e in survived), (
        "запись, которую модель не читала в этом ходу, была удалена по "
        "порядковому номеру. Деструктивный инструмент обязан отклонять id, "
        "не подтверждённый свежим чтением"
    )


# --------------------------------------------------------------------------
# 4. Собственная скедулерная выдача — не реплика владельца.
#    13 из 24 записей боевой истории — пинги и дайджесты самого бота.
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="К4: джобы бота пишутся в историю как реплики владельца")
def test_scheduled_output_is_not_stored_as_user_turn():
    rg = _bare_rg()
    rg._save_interaction(
        "(scheduled, пинг дня) Влад сегодня ещё не на связи",
        "Как день задался?",
        CHAT,
    )

    rows = db.history_load(CHAT, 10)
    intruders = [r for r in rows if str(r["user"]).startswith("(scheduled")]
    assert not intruders, (
        "джоб бота записан в историю как то, что сказал владелец. "
        "Через несколько таких пингов его настоящие слова вытесняются из окна"
    )


# --------------------------------------------------------------------------
# 5. Слова владельца не вытесняются из окна собственными пингами бота.
#    Сценарий 20.08: перед сообщением Влада — семь подряд джобов.
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="К4: реплики владельца вытесняются из окна пингами бота")
def test_user_words_survive_a_run_of_bot_pings():
    history = [{"user": "У меня сейчас семестрфериен, учебы нету",
                "bot": "Записала: каникулы"}]
    for n in range(4):
        history.append({"user": f"(scheduled, пинг дня) день {n}",
                        "bot": f"Как дела, день {n}?"})

    rendered = _bare_rg()._build_user_message(
        _ctx(user_text="что у меня сегодня?", history=history)
    )

    assert "семестрфериен" in rendered.lower(), (
        "реплика владельца вытеснена пингами бота: в промпт попали только "
        "последние ходы, и все они — сообщения самого бота"
    )


# --------------------------------------------------------------------------
# 6. Сбой памяти обязан быть слышен.
#    Сейчас logger.debug при уровне INFO: память может умереть насовсем,
#    и в логе не будет ни строки.
# --------------------------------------------------------------------------

def test_memory_search_failure_is_audible(caplog):
    class DeadMemory:
        def search(self, *a, **kw):
            raise RuntimeError("FTS5 index corrupted")

    rg = _bare_rg(mem=DeadMemory())
    with caplog.at_level(logging.WARNING):
        rg._enhance_context(_ctx(user_text="что я говорил про креатин?"))

    assert caplog.records, (
        "поиск по памяти упал молча. Владелец получит ответ без контекста и "
        "не узнает, что память вообще не участвовала"
    )


def test_memory_persist_failure_is_audible(caplog):
    class UnwritableMemory:
        def add(self, *a, **kw):
            raise RuntimeError("disk full")

    rg = _bare_rg(mem=UnwritableMemory())
    with caplog.at_level(logging.WARNING):
        rg._save_interaction("важный факт про учёбу", "принято", CHAT)

    assert caplog.records, (
        "запись в долгую память провалилась молча — факт потерян навсегда, "
        "и никакой сигнал об этом не поднялся"
    )


# --------------------------------------------------------------------------
# 7. Поминутный лимит — не суточный.
#    В чат трижды ушло «Дневной лимит Groq исчерпан» там, где Groq просил
#    подождать 16 секунд.
# --------------------------------------------------------------------------

TPM_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000, Used 4145, "
    "Requested 6049. Please try again in 16.454999999s'}}"
)
TPD_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 200000. "
    "Please try again in 3h21m'}}"
)


def test_minute_limit_is_not_reported_as_daily():
    classify = getattr(rg_mod, "_is_daily_limit_error", None)
    assert classify is not None, (
        "нет функции, отличающей суточный лимит от поминутного — поэтому "
        "любой 429 объявляется владельцу приговором на сутки"
    )
    assert classify(TPD_ERROR) is True
    assert classify(TPM_ERROR) is False, (
        "поминутный лимит классифицирован как суточный: владельцу сказали "
        "ждать до полуночи вместо 17 секунд"
    )


def test_retry_delay_is_extracted_from_the_error():
    extract = getattr(rg_mod, "_retry_after_seconds", None)
    assert extract is not None, (
        "провайдер присылает точное время ожидания, а прочитать его нечем"
    )
    assert 16 <= extract(TPM_ERROR) <= 17


# --------------------------------------------------------------------------
# 8. Новый факт обязан обесценивать противоречащий ему старый план.
#    17.08 12:01 записаны каникулы; 13:30 бот зовёт на лекцию в 14:05,
#    18.08 и 19.08 продолжает опираться на план от 12.08.
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="К6: нет сверки плана недели с более свежими фактами")
def test_contradicting_fact_invalidates_the_stale_week_plan():
    coach_storage.save_week_plan(
        "**Понедельник, 17 августа:**\n"
        "*   14:05 — 15:45: Лекция Grundlagen der Ingenieurmathematik (Ботроп).\n"
        "**Вторник, 18 августа:**\n"
        "*   12:20 — 15:45: Лекция и практика Ingenieurmathematik (Ботроп)."
    )
    coach_storage.add_diary_entry(
        "Семестрфериен (каникулы), учёбы сейчас нет", tags=["учёба"]
    )

    conflicts = getattr(coach_storage, "week_plan_conflicts", None)
    assert conflicts is not None, (
        "нет механизма сверки плана с более свежими фактами — план живёт "
        "вечно и переживает любое опровержение"
    )
    assert conflicts(), (
        "план с лекциями не помечен устаревшим после записи о каникулах: "
        "пинг дня продолжит звать владельца на пары"
    )


# --------------------------------------------------------------------------
# 9. Каждая реплика в истории обязана знать, кто её произнёс.
#    В боевой базе колонка agent пуста во всех 24 строках: четыре бота
#    делят одно ведро контекста, хотя код обещает изоляцию.
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="К7: колонка agent не заполняется ни одним вызовом")
def test_history_row_knows_which_agent_spoke():
    db.history_add(CHAT, "что по расписанию?", "вот план", agent="Iris")
    db.history_add(CHAT, "а новости?", "вот новости", agent="Newser")

    rows = db.history_load(CHAT, 10)
    unattributed = [r for r in rows if not str(r["agent"] or "").strip()]
    assert not unattributed, "запись истории без автора"

    rg = _bare_rg()
    rg._save_interaction("обычный вопрос", "обычный ответ", CHAT)

    rows = db.history_load(CHAT, 10)
    latest = rows[-1]
    assert str(latest["agent"] or "").strip(), (
        "основной путь генерации пишет историю без имени агента — "
        "изоляция контекстов Iris/Newser/Redmond существует только в комментарии"
    )
