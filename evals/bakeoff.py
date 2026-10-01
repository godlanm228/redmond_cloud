"""Model bake-off: the owner's real messages, the bot's real prompt, several models.

Why. On Oct 1, 2026 the same code and the same real conversation gave
«Валик — уменьшительное от Влад, то есть ты сам» on gpt-oss-120b and
«уменьшительное от Валентин, кто он для тебя — не знаю» on gemini-3.6-flash.
The model the owner talks to is chosen on his own messages, side by side,
not on impressions.

Each message is replayed through the real pipeline in a sandbox cut to the
moment it was written (real history, real profile): understanding, routing,
the diary entries code makes. The first prompt the answering agent sends to
its model (system + user message with history and the reading) is captured,
and the same prompt then goes to every model under test without tools - the
answer only. The live pipeline's own reply is kept as "production".

Claude models are called through the Claude Code CLI (`claude -p`), the way
Cipher calls it: the owner's subscription, no API key. Keep the count small,
the subscription's limits are shared with his own work.

    ./venv/bin/python -m evals.bakeoff --ids 611,615 \
        --models groq:openai/gpt-oss-120b,gemini:gemini-3.6-flash,claude:sonnet --out /tmp/bake
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("evals.bakeoff")

# Same for every model: the answer is compared, not the tool calls.
ANSWER_ONLY = ("\n\n(Model comparison run: no tools are available in this call. Answer the "
               "owner exactly as you would in the chat; what code already did for this "
               "message is listed in the user message.)")


class Capture:
    """The first prompt the answering agent sends to a model in one turn."""

    def __init__(self) -> None:
        self.system = ""
        self.user = ""
        self.agent_model = ""

    def reset(self) -> None:
        self.system = self.user = self.agent_model = ""

    def install(self) -> None:
        from logic.response_generator import ResponseGenerator
        from utils import gemini

        cap = self
        original_groq = ResponseGenerator._groq_chat

        def groq_chat(rg, api_key, model, messages, tools, *a, **kw):
            if not cap.user and tools is not None:
                cap.system = next((m["content"] for m in messages if m.get("role") == "system"), "")
                cap.user = next((m["content"] for m in messages if m.get("role") == "user"), "")
                cap.agent_model = model
            return original_groq(rg, api_key, model, messages, tools, *a, **kw)

        ResponseGenerator._groq_chat = groq_chat

        original_contents = gemini.generate_contents

        def generate_contents(contents, *a, **kw):
            if not cap.user and kw.get("tools") and contents:
                cap.system = kw.get("system", "")
                parts = contents[0].get("parts") or []
                cap.user = "".join(p.get("text", "") for p in parts)
                cap.agent_model = kw.get("model", "")
            return original_contents(contents, *a, **kw)

        gemini.generate_contents = generate_contents


def ask(spec: str, system: str, user: str) -> Dict[str, Any]:
    """spec = provider:model. {"text", "seconds", "error"}."""
    provider, _, model = spec.partition(":")
    started = time.time()
    text, error = "", ""
    try:
        if provider == "groq":
            from utils import groq
            extra = {"reasoning_effort": "medium"} if "gpt-oss" in model else None
            completion, error = groq.chat(model, [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}],
                                          max_tokens=1500 if extra else 900, extra=extra)
            text = groq.text_of(completion)
        elif provider == "gemini":
            from utils import gemini
            data = gemini.generate_contents([{"role": "user", "parts": [{"text": user}]}],
                                            system=system, model=model, max_tokens=1100,
                                            thinking_level="medium")
            text = gemini.extract_text(data)
            if not text:
                error = "no answer (limit or error, see log)"
        elif provider == "claude":
            proc = subprocess.run(
                ["claude", "-p", user, "--system-prompt", system, "--model", model,
                 "--tools", "", "--output-format", "json", "--no-session-persistence"],
                cwd="/tmp", capture_output=True, text=True, timeout=180)
            try:
                out = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError:
                out = {}
            text = str(out.get("result") or "").strip()
            if out.get("is_error") or not text:
                error = (proc.stderr or proc.stdout or "no output")[-400:]
        else:
            error = f"unknown provider {provider!r}"
    except Exception as e:  # noqa: BLE001 — one model failing must not end the run
        error = f"{e.__class__.__name__}: {e}"
    return {"text": text, "seconds": round(time.time() - started, 1), "error": error}


def _messages(db: Path, ids: List[int]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        out = []
        for mid in ids:
            row = conn.execute("SELECT user, bot, timestamp FROM memory WHERE id=?", (mid,)).fetchone()
            if row is None:
                raise ValueError(f"memory #{mid} not found")
            at = datetime.fromtimestamp(row[2], timezone.utc).astimezone()
            out.append({"id": mid, "text": row[0], "live_reply": row[1], "at": at.isoformat()})
        return out
    finally:
        conn.close()


async def run(ids: List[int], models: List[str], pause: float, out_dir: Path) -> Path:
    from evals.harness import Runner, Sandbox, Scenario, Turn, _local

    hub = Path.cwd()
    items = _messages(hub / "data" / "memory.sqlite", ids)
    sandbox = Sandbox(hub)
    sandbox.enter()
    cap = Capture()
    cap.install()
    results: List[Dict[str, Any]] = []
    try:
        runner = Runner(sandbox, pause=0, use_judge=False)
        try:
            for item in items:
                at = _local(datetime.fromisoformat(item["at"]))
                turn = Turn(kind="owner", text=item["text"], at=at.isoformat(timespec="seconds"))
                sc = Scenario(name=f"m{item['id']}", turns=[turn])
                cap.reset()
                runner.start(sc)
                res = await runner.run_turn(sc, 0, turn)
                entry = {**item, "agent": res.agent, "production": res.replies,
                         "diary_added": res.diary_added,
                         "understanding": res.understanding,
                         "prompt_model": cap.agent_model,
                         "system": cap.system, "user": cap.user, "answers": {}}
                if cap.user:
                    for spec in models:
                        entry["answers"][spec] = await asyncio.to_thread(
                            ask, spec, cap.system + ANSWER_ONLY, cap.user)
                        logger.info("#%d %s: %s", item["id"], spec,
                                    entry["answers"][spec]["error"] or "ok")
                else:
                    logger.warning("#%d: no model prompt captured (agent %s)", item["id"], res.agent)
                results.append(entry)
                if pause:
                    await asyncio.sleep(pause)
        finally:
            runner.close()
    finally:
        sandbox.cleanup()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bakeoff-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    path.write_text(json.dumps({"models": models, "results": results},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def blind_labels(models: List[str], seed: int) -> Dict[str, str]:
    """Model → letter, shuffled per message so a letter says nothing about the model."""
    rnd = random.Random(seed)
    order = list(models)
    rnd.shuffle(order)
    return {m: "ABCDEFGH"[i] for i, m in enumerate(order)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Same real prompts, several models, side by side.")
    ap.add_argument("--ids", required=True, help="memory ids of owner messages, comma-separated")
    ap.add_argument("--models", required=True, help="provider:model, comma-separated")
    ap.add_argument("--pause", type=float, default=20.0)
    ap.add_argument("--out", default="data/evals/results")
    args = ap.parse_args(argv)

    hub = Path.cwd()
    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv
        load_dotenv(hub / ".env")
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    path = asyncio.run(run(ids, models, args.pause, hub / args.out))
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
