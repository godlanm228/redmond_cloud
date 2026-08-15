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
| 🧠 Cipher | `@cipher_redberry_bot` | Claude Code через Pro | subprocess + сессии |

Fallback при rate_limit / tool_use_failed: `qwen/qwen3.6-27b` (предыдущий
`qwen/qwen3-32b` Groq снёс — 404 `model_not_found`, чинили 13.08.2026).
Gemini — `gemini-3.6-flash`, роутер — `gemini-3.1-flash-lite`.

## Структура

Local repo (`redmond_cloud/`) и VM (`~/redmond-hub/`) — **идентичная flat layout** (унифицировано 2026-05-16 в Фазе 3 миграции).

```
redmond-hub/
├── .env              ← runtime (gitignored), source of truth
├── .env.example
├── cloud_main_v2.py  ← единственный entry point
├── config/           ← config.py, config.json, owner_profile.json (gitignored), ...
├── core/             ← dispatcher, coordinator, multi_bot_runner
├── data/             ← memory.sqlite, search_usage.sqlite, coach/, owner_dossier.md
├── handlers/
├── logic/
├── safety/
├── utils/
├── hooks/
├── venv/             ← gitignored
└── requirements-cloud.txt
```

Никаких `app/` подпапок. Никаких legacy entry points. Дерево на VM = `git clone github.com/godlanm228/redmond_cloud` + восстановление gitignored runtime (.env, venv, data/, config/owner_profile.json).

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

## Env vars (`~/redmond-hub/.env` на VM, не коммитится)

```
TELEGRAM_BOT_TOKEN, IRIS_BOT_TOKEN, CIPHER_BOT_TOKEN, NEWSER_BOT_TOKEN
MAIN_CHAT_ID, ALLOWED_USER_IDS
REDMOND_GROQ_API_KEY, REDMOND_GEMINI_API_KEY
REDMOND_GOOGLE_API_KEY, REDMOND_GOOGLE_SEARCH_ENGINE_ID
```

Префиксированные имена с `REDMOND_` — то что реально читает `config/config.py`. Без префикса (`GROQ_API_KEY=`) — НЕ читается, не добавлять.

Template — `.env.example`.

## Deploy на VM — через git pull (с 2026-05-16)

**Workflow:**

1. Локально: правка → `git commit` → `git push origin main`.
2. На VM: `cd ~/redmond-hub && git pull` → рестарт v2.

**НЕ использовать `scp`** для деплоя кода. Только git. Исключение — заливка ключей в `.env` (одноразово) или больших не-git артефактов.

> **Если ты Cipher — ты УЖЕ на этой VM.** Команды ниже с `ssh` предназначены
> для запуска с ноутбука. Изнутри VM никакой ssh не нужен: работай локально.
> До 15.08.2026 здесь был раздел про `nohup` и `/tmp/v2.log`, и Cipher по нему
> честно пытался заssh-иться сам в себя с виндовым путём к ключу.

```bash
# SSH с ноутбука (ключ: C:\Users\you\.ssh\oracle-key.key)
ssh -i "C:/Users/you/.ssh/oracle-key.key" ubuntu@203.0.113.10

# Деплой: git pull + рестарт systemd-юнита
ssh -i "..." ubuntu@203.0.113.10 "cd ~/redmond-hub && git pull && \
   sudo systemctl restart redmond-hub.service"

# Логи (пишутся сервисом, файл под root:ubuntu)
ssh -i "..." ubuntu@203.0.113.10 "sudo tail -50 ~/redmond-hub/logs/v2.log"

# Тесты (pytest стоит в venv)
ssh -i "..." ubuntu@203.0.113.10 "cd ~/redmond-hub && ./venv/bin/python -m pytest tests/ -q"
```

**Сервис:** `redmond-hub.service` (systemd, enabled — автостарт после ребута).
Ручной `nohup` больше не используется.

**Данные:** `data/memory.sqlite` — единая база (coach-данные, история чата,
память с FTS5, сессии Cipher). `data/coach/*.json` — замороженный архив до
миграции 15.08.2026, живыми данными НЕ является. Бэкап — ежедневно в 04:30
через `deploy/backup_db.sh` (VACUUM INTO, 14 снимков в `~/backups/db/`).

**Github SSH:** VM имеет deploy key (`~/.ssh/github_deploy` + `~/.ssh/config` маппит github.com на этот ключ). Read-only. Добавлен в `repo Settings → Deploy keys → "Oracle VM redmond-hub"`.

**Что НЕ коммитится (gitignored, живёт только на VM):**
- `.env` — секреты
- `venv/` — Python venv
- `data/` — runtime SQLite (memory.sqlite: coach, история, память, сессии Cipher) + owner_dossier
- `config/owner_profile.json` — мутируется Iris в runtime

При первичной инициализации VM или восстановлении из backup — эти файлы перенести руками поверх `git clone`.


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
