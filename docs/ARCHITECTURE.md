# Architecture

Technical reference for Redmond Cloud — the multi-agent Telegram hub. For the project
overview see [README.md](../README.md).

## Runtime shape

One Python process on an Oracle Cloud Free Tier VM runs four `python-telegram-bot`
`Application` instances concurrently via `asyncio.gather`. They share a `Dispatcher`
(safety, intent recognition, response generation), a `Coordinator` (bot registry, HTML
output, typing indicators, safe message chunking) and a single SQLite database.

Telegram privacy mode is off, so every bot receives every group message. Which one replies
is decided by the router, not by the mention syntax.

```
redmond-hub/
├── .env                 runtime secrets (gitignored, source of truth)
├── cloud_main_v2.py     the only entry point
├── config/              settings, JSON schema, personality profile
├── core/                dispatcher, coordinator, multi_bot_runner, scheduler
├── data/                SQLite database and runtime state (gitignored)
├── handlers/            Telegram handlers, auth gate
├── logic/               agents, router, response generator, tools, vision
├── safety/              goal manager
├── utils/               db layer, Gemini client, memory, search, migrations
└── venv/                gitignored
```

The local checkout and the VM use an identical flat layout — no `app/` nesting, no legacy
entry points. The deployed tree is a `git clone` plus the gitignored runtime files.

## Key modules

| File | Responsibility |
|---|---|
| `cloud_main_v2.py` | Entry point. Builds the dispatcher, hands it to the multi-bot runner |
| `core/multi_bot_runner.py` | Four Applications in `asyncio.gather`, shared dispatcher and coordinator |
| `core/coordinator.py` | Bot registry for cross-agent sends, HTML output, typing indicator, chunking |
| `core/dispatcher.py` | Thin wrapper around shared services (safety, intent, response generation) |
| `core/scheduler.py` | APScheduler cron jobs, all in `Europe/Berlin` |
| `handlers/multi_bot.py` | Auth gate, router handler, slim per-agent handler |
| `logic/agents.py` | Agent configuration, triggers, name aliases |
| `logic/agent_router.py` | Routing without an explicit mention |
| `logic/response_generator.py` | Tool loop, multi-model fallback, per-chat history |
| `logic/tools.py` | Tool schemas, executors, source ranking |
| `utils/db.py` | SQLite layer, schema, migrations |

## Model chain

| Position | Model |
|---|---|
| Primary | Groq `openai/gpt-oss-120b` |
| Fallback | Groq `qwen/qwen3.6-27b` |
| Provider fallback | Google `gemini-3.6-flash` |
| Router | `gemini-3.1-flash-lite`, falling back to Groq `llama-3.1-8b-instant` |

Provider order is configurable through `config/config.json` (`llm_provider_order`). Errors
are accumulated along the chain rather than overwritten, so a failure surfaces its real
cause instead of the last generic message.

Free-tier quotas drive several design decisions: system prompts are split into an English
core (token economy) and a localised voice layer, the owner dossier is read by section
rather than whole, tools are filtered before the request leaves the process, and the dossier
cache is scoped per conversation.

## Authorization

A message is answered only when `chat.id` matches the configured group **and** `user.id` is
in the allowlist. Direct messages get a one-time multilingual notice and are then ignored.
There is no password layer — Telegram identity is the credential.

## Storage

Everything lives in `data/memory.sqlite`: long-term memory with FTS5 full-text search, chat
history that survives restarts, coach data (goals, deadlines, diary, meals, shifts) and
Cipher's session state. Schema changes go through a migration path, and a JSON→SQLite
migration exists for the pre-August 2026 flat files; compatibility with those files is
covered by tests.

Backups run nightly at 04:30 through `deploy/backup_db.sh`, which uses `VACUUM INTO` for a
consistent snapshot without stopping the service, keeping 14 rolling copies.

## Scheduled jobs

| Time (Europe/Berlin) | Job |
|---|---|
| 09:00 | Morning digest |
| 09:03 | Deadline check — silent when nothing is urgent |
| 09:10 | Cipher authorization watch |
| 10:00–22:00, every 30 min | Day ticker with backoff |
| 22:30 | Evening summary |

The ticker respects a two-level mute (`pings` silences only the ticker, `all` silences
everything) and backs off on its own: two unanswered pings and it stays quiet until the
user writes again.

## Deployment

Code ships via `git pull` — never `scp`. The hub runs as the systemd unit
`redmond-hub.service` with autostart enabled; logs go to `logs/v2.log`.

```bash
ssh ubuntu@your-vm "cd ~/redmond-hub && git pull && sudo systemctl restart redmond-hub.service"
ssh ubuntu@your-vm "sudo tail -50 ~/redmond-hub/logs/v2.log"
ssh ubuntu@your-vm "cd ~/redmond-hub && ./venv/bin/python -m pytest tests/ -q"
```

The VM authenticates to GitHub with a read-only deploy key.

## What is never committed

`.env` (secrets), `venv/`, `data/` (runtime database and dossier) and
`config/owner_profile.json`, which the coach agent mutates at runtime. On a fresh VM these
are restored by hand on top of the clone.

## Environment variables

```
TELEGRAM_BOT_TOKEN, IRIS_BOT_TOKEN, CIPHER_BOT_TOKEN, NEWSER_BOT_TOKEN
MAIN_CHAT_ID, ALLOWED_USER_IDS
REDMOND_GROQ_API_KEY, REDMOND_GEMINI_API_KEY
REDMOND_GOOGLE_API_KEY, REDMOND_GOOGLE_SEARCH_ENGINE_ID
```

Only the `REDMOND_`-prefixed names are read by `config/config.py`; an unprefixed
`GROQ_API_KEY` is ignored. See [`.env.example`](../.env.example).
