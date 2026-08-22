"""Отказ называет фактическую причину, а не худшее предположение.

Строки ошибок ниже — дословно из боевого лога 15–17.08.2026.

Что было: `_is_rate_limit_error` считал любой 429 исчерпанием лимита, и в чат
уходило «Дневной лимит Groq исчерпан (бесплатный тариф, сброс в полночь по
UTC)» — ровно 108 символов. В логе эта заглушка встречается трижды как
`Scheduled job done: Iris → chat (108 chars)`, то есть вместо вечернего итога
владелец получал приговор на сутки. Провайдер в том же теле писал, сколько
ждать на самом деле: 16 секунд.
"""

from logic import response_generator as rg


TPM = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` on "
    "tokens per minute (TPM): Limit 8000, Used 4145, Requested 6049. "
    "Please try again in 16.454999999s.'}}"
)
TPD = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 200000. "
    "Please try again in 3h21m.'}}"
)
NO_HINT = "Error code: 429 - {'error': {'message': 'rate_limit_exceeded'}}"


def test_minute_limit_is_not_daily():
    assert rg._is_daily_limit_error(TPM) is False
    assert rg._is_daily_limit_error(TPD) is True


def test_wait_is_read_from_the_provider():
    assert 16 <= rg._retry_after_seconds(TPM) <= 17
    assert rg._retry_after_seconds(TPD) == 3 * 3600 + 21 * 60
    assert rg._retry_after_seconds(NO_HINT) is None


def test_reply_to_minute_limit_does_not_sentence_the_owner_to_a_day():
    reply = rg._rate_limit_reply(TPM)
    assert "Дневной" not in reply, f"минутный лимит выдан за суточный: {reply!r}"
    assert "полночь" not in reply
    assert "17 сек" in reply, f"владельцу не сказано реальное время ожидания: {reply!r}"


def test_reply_to_daily_limit_still_says_daily():
    reply = rg._rate_limit_reply(TPD)
    assert "Дневной" in reply
    assert "3 ч" in reply


def test_reply_without_a_hint_stays_honest():
    """Провайдер не сказал сколько ждать — не выдумываем ни сутки, ни секунды."""
    reply = rg._rate_limit_reply(NO_HINT)
    assert "Дневной" not in reply
    assert "полминуты" in reply


def test_chain_of_errors_is_classified_by_the_worst_real_cause():
    """В цепочке моделей ошибки копятся; минутный лимит рядом с 413 остаётся минутным."""
    chain = TPM + " | Error code: 413 - request too large"
    assert rg._is_daily_limit_error(chain) is False
    assert "Дневной" not in rg._rate_limit_reply(chain)
