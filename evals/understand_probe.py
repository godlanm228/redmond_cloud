"""Run the understanding step alone on real owner messages, before it decides anything.

Lesson of Sep 29, 2026: the router's model was swapped after a check on three
phrases, and the first replay of real conversations found it answering
«Суисайд» with silence. A classifier is measured on the owner's own messages
first; this probe does exactly that and nothing else - no answers are
generated, nothing is written.

Each message is read with the conversation that really preceded it (from the
memory table). Output: one block per message and a JSON file for comparison
between versions.

    ./venv/bin/python -m evals.understand_probe --scenarios data/evals/scenarios.json --recent 20
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _history(conn, before_id: int, day_start: float) -> List[Dict[str, str]]:
    rows = conn.execute("SELECT user, bot FROM memory WHERE id < ? AND timestamp >= ? "
                        "ORDER BY id DESC LIMIT 4", (before_id, day_start)).fetchall()
    out: List[Dict[str, str]] = []
    for user, bot in reversed(rows):
        if user.lstrip().startswith("("):
            # As the live bot sees it: the router state carries the agent's name.
            who = "Newser" if "дайджест" in user else "Iris"
            out.append({"who": who, "text": bot})
        else:
            out += [{"who": "Влад", "text": user}, {"who": "бот", "text": bot}]
    return out


def _messages(conn, scenarios: Path, recent: int) -> List[Tuple[str, int, str, float, Dict]]:
    """(label, memory id or 0, text, timestamp, expect)."""
    out = []
    if scenarios.exists():
        for sc in json.loads(scenarios.read_text(encoding="utf-8")):
            for i, t in enumerate(sc["turns"]):
                if "memory_id" in t:
                    row = conn.execute("SELECT user, timestamp FROM memory WHERE id=?",
                                       (t["memory_id"],)).fetchone()
                    if row and not row[0].lstrip().startswith("("):
                        out.append((f"{sc['name']}#{i}", t["memory_id"], row[0], row[1],
                                    t.get("expect", {})))
                elif not t["text"].lstrip().startswith("("):
                    ts = datetime.fromisoformat(t["at"]).timestamp()
                    out.append((f"{sc['name']}#{i}", 0, t["text"], ts, t.get("expect", {})))
    if recent:
        seen = {mid for _l, mid, *_ in out if mid}
        rows = conn.execute("SELECT id, user, timestamp FROM memory WHERE user NOT LIKE '(%' "
                            "ORDER BY id DESC LIMIT ?", (recent,)).fetchall()
        for mid, text, ts in reversed(rows):
            if mid not in seen:
                out.append((f"memory#{mid}", mid, text, ts, {}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/evals/scenarios.json")
    ap.add_argument("--recent", type=int, default=0)
    ap.add_argument("--out", default="data/evals/results")
    args = ap.parse_args(argv)

    from contextlib import suppress
    with suppress(ImportError):
        from dotenv import load_dotenv
        load_dotenv(".env")
    import logging
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    from config.config_loader import load_app_config
    from logic import understanding
    from utils import llm_gate
    from utils.time import OWNER_TZ

    llm_gate.configure_from(load_app_config())
    conn = sqlite3.connect("file:data/memory.sqlite?mode=ro", uri=True)
    results = []
    for label, mid, text, ts, expect in _messages(conn, Path(args.scenarios), args.recent):
        at = datetime.fromtimestamp(ts, timezone.utc).astimezone(OWNER_TZ)
        day_start = at.replace(hour=0, minute=0, second=0).timestamp()
        history = _history(conn, mid, day_start) if mid else []
        now = at.strftime("%Y-%m-%d %H:%M, %A")
        started = time.time()
        u = understanding.understand(text, history, now=now)
        took = round(time.time() - started, 1)
        rec = {"label": label, "text": text, "at": at.isoformat(timespec="minutes"),
               "seconds": took, "expect": expect, "understanding": u.to_dict() if u else None}
        results.append(rec)
        print(f"\n== {label} [{rec['at']}] {took}s  «{' / '.join(text.split(chr(10)))[:160]}»")
        if u is None:
            print("   NO UNDERSTANDING")
            continue
        want = expect.get("agent")
        mark = "" if not want or u.addressee in ([want] if isinstance(want, str) else want) \
            else f"   <-- expected {want}"
        print(f"   → {u.addressee}{' +research' if u.research else ''} | urgency {u.urgency}"
              f"{' «' + u.urgency_quote + '»' if u.urgency_quote else ''} | {u.model}{mark}")
        print(f"   about: {u.about}")
        if u.refers_to:
            print(f"   refers to: {u.refers_to}")
        for f in u.facts:
            print(f"   fact [{f.when}/{f.topic}]: {f.fact}   ← «{f.quote}»")
        for c in u.commands:
            print(f"   command {c.type} until={c.until!r} hours={c.hours} scope={c.scope} ← «{c.quote}»")
        for d in u.dropped:
            print(f"   DROPPED: {d}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"understand-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
