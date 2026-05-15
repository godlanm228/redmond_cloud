# Redmond — гид для Claude Code

Этот файл автоматически читается Claude Code при работе с этим репо.
Полная история и архитектура — в `~/.claude/projects/.../memory/project_redmond.md`.

## TL;DR что это

Личный AI-ассистент Влада с **двумя режимами**:
- **Cloud (production):** `@redmond_hub_bot` в Telegram на Oracle Cloud Free VM (`203.0.113.10`)
- **Home (local):** голос через Whisper + edge-tts (запуск `python main.py`)

**Multi-agent в Telegram:**
- 🦞 **Redmond** — повседневный (погода, факты, поиск, болтовня)
- 🎯 **Iris** — личный коуч (цели/дедлайны/дневник/дисциплина, женский character)

Routing: smart (без префиксов работает через `llama-3.1-8b-instant`), либо явный `@redmond` / `@iris` / `@i`.

## Стек

- **LLM:** Groq + `openai/gpt-oss-120b` (function calling), fallback → Gemini → Transformers
- **Router:** Groq `llama-3.1-8b-instant` (быстрый, бесплатный)
- **TTS (home):** edge-tts `en-US-BrianMultilingualNeural`
- **ASR (home):** Whisper `large-v3` на CUDA
- **Memory:** SQLite + FAISS (home) / SQLite-only (cloud, 1 GB RAM)
- **Web search:** Google CSE → DuckDuckGo fallback
- **Weather:** wttr.in (бесплатно, без ключа)

## Ключевые файлы

| Файл | Что |
|---|---|
| `cloud_main.py` | Entry для VPS — Telegram bot + router |
| `main.py` | Entry для home — голосовой режим |
| `core/engine.py` | RedmondEngine (home mode) |
| `core/dispatcher.py` | Auth + safety + intent + дефолтный routing |
| `logic/agents.py` | AgentConfig (Redmond, Iris) + триггеры |
| `logic/agent_router.py` | Smart routing без @-меншинов |
| `logic/response_generator.py` | LLM + function calling tool-loop |
| `logic/tools.py` | TOOL_SCHEMAS + executors (weather/search/coach-tools) |
| `logic/coach_storage.py` | JSON storage для целей/дневника/дедлайнов |
| `config/personality_profile.json` | Принципы поведения Redmond v2 |
| `config/owner_profile.json` | Структурированный профиль владельца |
| `data/owner_dossier.md` | Полное AI-сгенерированное досье (не цитировать дословно) |

## Деплой на VPS

```bash
# SSH
ssh -i "C:/Users/Vlad/Downloads/oracle-key.key" ubuntu@203.0.113.10

# Залить файлы
scp -i "C:/Users/Vlad/Downloads/oracle-key.key" PATH ubuntu@203.0.113.10:~/redmond-hub/app/PATH

# Restart service
ssh -i "C:/Users/Vlad/Downloads/oracle-key.key" ubuntu@203.0.113.10 "sudo systemctl restart redmond-hub"

# Логи
ssh -i "C:/Users/Vlad/Downloads/oracle-key.key" ubuntu@203.0.113.10 "sudo journalctl -u redmond-hub -n 50 --no-pager"
```

## Important context

- **Личность Redmond ≠ Jarvis-калька.** Не использовать «сэр», «к Вашим услугам». Свой стиль.
- **Iris женский character.** Греческая богиня-вестница. Жёсткая, без воды, не утешает.
- **Никаких pep-talk** («Вы справитесь», «всё получится»). Влад не выносит.
- **Не выдумывать цифры/погоду** — обязательно вызывать tools (`get_weather`, `web_search`).
- **Не цитировать AI-литературщину** из досье («бухгалтерия усталости» и т.п.). Это не язык владельца.
- **Не начинать ответ с «Влад, ...»** — обращение по имени только когда уместно.
- **Длина ответа по ситуации** — короткий вопрос → одна фраза, инструкция → разворачиваться.

## Pending / TODO

1. Voice messages из TG → Groq Whisper API
2. News-агент + scheduler с утренними дайджестами
3. Cipher — Claude Code as 3rd agent через Agent SDK + Pro
4. rbry crypto bot интеграция (план в memory: `project_redmond_rbry_integration.md`)
5. Google API: ждём активацию (Vorauszahlung прошла, проект-level pending)

## Знакомство с владельцем

Vladyslav Kulahin («Энди»), 25, Эссен (Германия). RU/DE(C1)/EN. ENTJ (AI-инференс).
Проекты: rbry crypto bot (Python/Binance/FinBERT), Redmond v0.2 (этот), Billiard Club (Unity 6).
Перегрузка: учёба + 2 работы + 3 проекта.

Подробности — `config/owner_profile.json` + `data/owner_dossier.md`.

## Tests

```
.venv/Scripts/python.exe -m pytest tests/
```

29/29 passed на момент 2026-05-14 до Iris/router рефакторинга. После — могут быть устаревшие, нужен прогон.
