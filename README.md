# Redmond Cloud — multi-agent Telegram-хаб

4 отдельных Telegram-бота, живущих в одном Python-процессе в одной группе.
Облачная часть Redmond. Локальный голосовой режим — в отдельном репо.

## Состав

| Бот | TG @username | Роль | Модель |
|---|---|---|---|
| 🦞 Redmond | `@redmond_hub_bot` | Listener + router + общий ассистент | Groq `openai/gpt-oss-120b` |
| 🎯 Iris | `@iris_redberry_bot` | Личный коуч и трекер (цели/дедлайны/дневник) | то же |
| 📰 Newser | `@newser_redmond_bot` | Searcher и новости с источниками | то же |
| 🧠 Cipher | `@cipher_redberry_bot` | Claude Code через Pro подписку | subprocess (`claude` CLI) |

Fallback модель при rate-limit / tool-use-failed: `qwen/qwen3-32b`.

## Архитектура

```
TG group (Redberry HUB)
   │
   ▼
Hub process (один Python, на Oracle Cloud Free VM)
   ├── 4 параллельных Application через asyncio.gather
   ├── Все боты слушают всё (Privacy OFF)
   ├── Coordinator — реестр Bot instances для cross-agent отправки
   ├── Auth gate — chat.id == MAIN_CHAT_ID обязательно
   └── Shared state: SQLite + JSON storage
```

## Деплой на Oracle VM

```bash
# Обновить код на VM
ssh -i oracle-key.key ubuntu@VM_IP "cd ~/redmond-hub && git pull"

# .env (см. .env.example)
ssh -i oracle-key.key ubuntu@VM_IP "nano ~/redmond-hub/.env"

# Рестарт systemd-сервиса
ssh -i oracle-key.key ubuntu@VM_IP "sudo systemctl restart redmond-hub"

# Логи
ssh -i oracle-key.key ubuntu@VM_IP "tail -f ~/redmond-hub/logs/v2.log"
```

Код деплоится через `git pull`; runtime-файлы (`.env`, `data/`, `venv/`,
`config/owner_profile.json`) живут только на VM и не коммитятся.

## Ключевые файлы

| Файл | Что |
|---|---|
| `cloud_main_v2.py` | Entry point (multi-bot) |
| `core/multi_bot_runner.py` | Запуск 4 Application'ов через asyncio.gather |
| `core/coordinator.py` | Реестр Bot instances + HTML output + typing indicator |
| `handlers/multi_bot.py` | Auth gate + redmond_handler (с router) + slim_agent_handler |
| `logic/agents.py` | AgentConfig (Redmond/Iris/Newser/Cipher) + триггеры/aliases |
| `logic/agent_router.py` | Routing когда нет явного @-меншина |
| `logic/response_generator.py` | Groq LLM + tool-loop + multi-model fallback |
| `logic/tools.py` | TOOL_SCHEMAS (en) + executors + source ranking |

## Лимиты Groq (free tier)

| Модель | TPM | TPD |
|---|---|---|
| `openai/gpt-oss-120b` (primary) | 8K | 200K |
| `qwen/qwen3-32b` (fallback) | 8K | 500K |
| `llama-3.1-8b-instant` (router) | 30K | 500K |

Чтобы не упираться:
- System prompts: CORE на английском (-50% токенов) + VOICE на русском
- `read_dossier_section(core/strengths/thinking/directives)` вместо полного файла
- Tool filter ДО отправки в Groq
- Dossier cache per-conversation
- Pre-search router отключён (legacy дублирование с tool calling)

## Безопасность

- Auth gate: `chat.id == MAIN_CHAT_ID` обязательно — вне группы боты молчат (one-time multilingual notice + silent)
- Newser source ranking — Russian aggregators (lenta, ria, bcs-express…) отрезаются или вниз
- Prompt-injection defense в system prompts всех агентов

## Локальная разработка

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements-cloud.txt

cp .env.example .env  # заполнить значениями

python cloud_main_v2.py
```

Личные данные владельца (`owner_profile.json`, `data/owner_dossier.md`, `data/coach/`)
**не коммитятся** — они мутируются Iris в runtime и живут только локально + на VM.
