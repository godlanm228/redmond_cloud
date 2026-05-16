# Redmond Cloud v2 — гид для Claude Code

> **Ты в CLOUD-инфре. Не PC.**
> PC voice-версия живёт в `C:\Users\Vlad\IdeaProjects\redmond\` — это **отдельная независимая инфра**, не legacy этого репо. Изменения отсюда туда не транслируются без явного запроса. Если «Редмонд» сказано без уточнения — спросить.

История и состояние — в memory: `project_redmond_v2_state.md` (читать первым), `project_redmond_agents.md`, `project_redmond.md` (про PC-инфру).

## TL;DR

Multi-agent Telegram-хаб на Oracle Cloud Free VM (`203.0.113.10`). 4 бота в TG-группе Redberry HUB (`chat_id=-1001234567890`), все слушают всё (Privacy OFF).

| Бот | TG | Роль | Модель |
|---|---|---|---|
| 🦞 Redmond | `@redmond_hub_bot` | Listener + router + общий ассистент | Groq `gpt-oss-120b` |
| 🎯 Iris | `@iris_redberry_bot` | Коуч/трекер (goals/diary/deadlines) | то же |
| 📰 Newser | `@newser_redmond_bot` | Searcher + новости с источниками | то же |
| 🧠 Cipher | `@cipher_redberry_bot` | Claude Code через Pro (stub) | subprocess |

Fallback при rate_limit / tool_use_failed: `qwen/qwen3-32b`.

## Структура local vs VM

| | Local (`redmond_cloud/`) | VM (`~/redmond-hub/`) |
|---|---|---|
| Layout | Всё в корне (`cloud_main_v2.py`, `config/`, `core/`, ...) | Слоистый: `.env` в корне, остальное в `app/` |
| `.env` source of truth | `.env.example` (template) | **Корневой `~/redmond-hub/.env`** (НЕ `app/.env`) |
| Entrypoint | `cloud_main_v2.py` (в корне) | `app/cloud_main_v2.py` |

Расхождение в layout — артефакт scp-deploy. Унификация — Фаза 3 (git pull deploy).

## Ключевые файлы

| Файл | Что |
|---|---|
| `cloud_main_v2.py` | **Единственный** entry point. Создаёт Dispatcher → передаёт в multi_bot_runner |
| `core/multi_bot_runner.py` | 4 Application в asyncio.gather, shared Dispatcher + Coordinator |
| `core/coordinator.py` | Bot registry + HTML output + typing indicator + safe chunking |
| `core/dispatcher.py` | **Тонкая обёртка** для shared services (safety / intent_recognizer / response_generator). Auth-логики НЕТ — `AuthManager` снесён 2026-05-16, пароли не нужны (auth через Telegram whitelist). Возможен дальнейший рефактор: см. backlog «Dispatcher refactor pass» |
| `handlers/multi_bot.py` | `_gate()` (auth по `chat.id == MAIN_CHAT_ID` + `user.id in ALLOWED_USER_IDS`), `redmond_handler` (router), `slim_agent_handler` |
| `logic/agents.py` | AgentConfig (4 агента) + triggers + name_aliases |
| `logic/agent_router.py` | Routing без явного @-mention (llama-3.1-8b-instant) |
| `logic/response_generator.py` | Groq tool-loop + multi-model fallback + per-chat history |
| `logic/tools.py` | TOOL_SCHEMAS (en) + execute_tool + source ranking |
| `config/config.json` | Override defaults из `config.py` (например `llm_provider_order`) |
| `config/owner_profile.json` | **gitignored** — мутируется Iris в runtime |

## Auth — как реально работает в v2

- Бот отвечает ТОЛЬКО если `chat.id == MAIN_CHAT_ID` (Redberry HUB) И `user.id in ALLOWED_USER_IDS`.
- DM → one-time multilingual notice + silent.
- **Паролей нет.** `basic_password` / `super_password` / `AuthManager` снесены в Фазе 1 чистки 2026-05-16. Не возвращать.

## Env vars (живут в корневом `~/redmond-hub/.env` на VM)

```
TELEGRAM_BOT_TOKEN, IRIS_BOT_TOKEN, CIPHER_BOT_TOKEN, NEWSER_BOT_TOKEN
MAIN_CHAT_ID, ALLOWED_USER_IDS
REDMOND_GROQ_API_KEY, REDMOND_GEMINI_API_KEY
REDMOND_GOOGLE_API_KEY, REDMOND_GOOGLE_SEARCH_ENGINE_ID
```

Префиксированные имена с `REDMOND_` — то что реально читает `config/config.py`. Без префикса (`GROQ_API_KEY=`) — НЕ читается, не добавлять.

Template — `.env.example`.

## Deploy на VM (текущий — Фаза 3 заменит на git pull)

```bash
# SSH
ssh -i "C:/Users/Vlad/Downloads/oracle-key.key" ubuntu@203.0.113.10

# Залить файл (scp поверх — временно, до git deploy)
scp -i "C:/Users/Vlad/Downloads/oracle-key.key" PATH ubuntu@203.0.113.10:~/redmond-hub/app/PATH

# Рестарт (manual nohup, systemd ещё не обновлён под v2)
ssh -i "..." ubuntu@203.0.113.10 "pkill -f cloud_main_v2"
ssh -i "..." ubuntu@203.0.113.10 \
  "cd ~/redmond-hub && set -a && . .env && set +a && \
   nohup ./venv/bin/python app/cloud_main_v2.py > /tmp/v2.log 2>&1 & disown"

# Логи
ssh -i "..." ubuntu@203.0.113.10 "tail -50 /tmp/v2.log"
```

**Важно (из глобального CLAUDE.md):**
- Не scp `.bak` файлы и не оставлять их на VM.
- После любой заливки — sweep на VM (`ls ~/redmond-hub` + `ls ~/redmond-hub/app/`) на предмет дублей entrypoint и забытых файлов. **Не только в локальном repo.**
- При cutover (замена файла): сначала `rm` старого на VM, потом scp нового. Не лить поверх.

## Правила поведения LLM (НЕ хардкодить в код — это для меня)

- Personality v2 в `config/personality_profile.json` — стиль НЕ хардкодить в промпте.
- Длина ответа по ситуации: короткий вопрос → одна фраза; инструкция → разворачиваться.
- Не выдумывать факты — вызывать tools (`get_weather`, `web_search`).
- Не цитировать AI-литературщину из owner_dossier (фразы вроде «бухгалтерия усталости»).
- System prompts: CORE на английском (экономия токенов) + VOICE на русском.
- Newser source ranking: trusted (Reuters/Bloomberg/Unity/GitHub) наверх, low-quality (lenta/ria/bcs-express/finam) вниз.

## Pending / TODO

См. memory `project_redmond_v2_state.md` для актуального списка с приоритетами.

## Знакомство с владельцем

Vladyslav Kulahin («Энди»), 25, родом из Киева, живёт в Эссене. RU/UA(native)/DE(C1)/EN.
3 активных проекта: rbry crypto bot, Redmond Cloud (этот), Billiard Club (Unity 6).
Подробности — `config/owner_profile.json` (gitignored) + `data/owner_dossier.md`.
