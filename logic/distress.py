"""Острая ситуация у владельца: сначала спросить, что случилось.

04.09.2026 Влад написал Iris одно слово в момент, когда ехал в больницу с
сильной болью в животе. Iris сама решила, что это «суицидальные мысли»,
записала это толкование в дневник и ответила заглушкой «Готово: записала в
дневник.» — модель в ту минуту не ответила. Человеческий ответ появился только
после двух упрёков.

Правило владельца (28.09.2026): в критической ситуации бот сначала спрашивает,
что случилось и что нужно, а не угадывает причину и не выдаёт готовые рецепты.

Поэтому здесь две вещи:
  • `detect()` — дешёвое распознавание сигнала по тексту, без модели;
  • `DIRECTIVE` — что модель обязана сделать в этом ответе;
  • `FALLBACK_REPLY` — что уходит, если модели не ответили: вопрос не должен
    зависеть от того, есть ли сейчас лимит у провайдера.

Ложное срабатывание стоит одного вопроса «что случилось?». Пропуск стоит
того, что было 04.09. Поэтому список скорее шире, чем уже, но без голых
«плохо» и «болит»: «плохо спал» и «болит голова немного» сюда не относятся.
"""

from __future__ import annotations

import re

_PATTERNS = [
    r"су[иі]?цид", r"суисайд", r"самоуби", r"покончить с собой", r"не хочу жить",
    r"(?<!со )(?<!от )умира(ю|ет)(?! со смеху| от смеха)",
    r"вызв\w* скор", r"\bскорую\b", r"скор(ая|ую|ой) помощ", r"\b112\b",
    r"не могу дышать", r"задыха", r"теря\w* сознани", r"потерял\w* сознани",
    r"мне (очень |совсем |дико |пиздец как |хуево и |так )?плохо\b",
    r"(очень|дико|адски|сильно|пиздец как|ебан\w*|безумно) (болит|больно)",
    r"(адск|дик|ебан\w*|сильн|невыносим|жутк)\w* бол[ьи]",
    r"\bв больниц", r"\bв реанимаци", r"\bnotarzt\b", r"\bkrankenhaus\b", r"\bnotaufnahme\b",
]
_RX = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def detect(text: str) -> bool:
    """Есть ли в сообщении сигнал острой ситуации."""
    if not text:
        return False
    return bool(_RX.search(text.lower().replace("ё", "е")))


DIRECTIVE = (
    "ACUTE SITUATION SIGNAL. The owner's message may mean an emergency or an acute "
    "state (strong pain, feeling very bad, danger). Your ONLY job in this reply: "
    "calmly and briefly ask what happened and what he needs right now. One or two "
    "short sentences in his language, warm and direct, like a close friend. "
    "Do NOT guess the cause, do NOT diagnose, do NOT assume self-harm or mental "
    "illness, do NOT give advice lists or hotline numbers unless he asks. Do NOT write "
    "anything to the diary in this reply. If he gives an explicit instruction (e.g. mute "
    "until a date), carry it out and then ask. Only if he describes "
    "an immediate danger to life, tell him to call 112 right now."
)

FALLBACK_REPLY = "Влад, что случилось? Что тебе сейчас нужно?"
