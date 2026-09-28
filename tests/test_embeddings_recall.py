"""Embeddings cache (utils/embeddings), hybrid memory recall (logic/recall)
and the load_tools round trip inside the generator.

No network: the embedding API is faked. Memory rows are real exchanges from
the production database (Aug-Sep 2026), shortened.
"""


from logic import recall
from utils import db, embeddings


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)
        self.reason = ""

    def json(self):
        return self._payload


def fake_post_factory(vectors_for, calls, status_by_model=None):
    def post(url, headers=None, json=None, timeout=None):
        model = url.split("/models/")[1].split(":")[0]
        calls.append((model, json))
        status = (status_by_model or {}).get(model, 200)
        if status != 200:
            return FakeResponse(status, {"error": {"message": "not found"}})
        texts = [r["content"]["parts"][0]["text"] for r in json["requests"]]
        return FakeResponse(200, {"embeddings": [{"values": vectors_for(t)} for t in texts]})
    return post


def setup_function(_):
    embeddings._model_in_use = None


# ---------- embeddings ----------

def test_request_shape_per_model(monkeypatch):
    b2 = embeddings._body("gemini-embedding-2", ["привет"], embeddings.QUERY)["requests"][0]
    assert b2["content"]["parts"][0]["text"].startswith("task: search result | query: ")
    assert "task_type" not in b2 and b2["output_dimensionality"] == embeddings.DIM
    b1 = embeddings._body("gemini-embedding-001", ["привет"], embeddings.DOCUMENT)["requests"][0]
    assert b1["task_type"] == "RETRIEVAL_DOCUMENT"


def test_falls_back_to_the_older_model_when_the_new_one_is_missing(monkeypatch):
    calls = []
    monkeypatch.setenv("REDMOND_GEMINI_API_KEY", "k")
    monkeypatch.setattr(embeddings.requests, "post",
                        fake_post_factory(lambda t: [1.0, 0.0], calls,
                                          {"gemini-embedding-2": 404}))
    assert embeddings.embed_query("привет") == [1.0, 0.0]
    assert [c[0] for c in calls] == ["gemini-embedding-2", "gemini-embedding-001"]
    assert embeddings.model_id() == "gemini-embedding-001"


def test_no_key_means_no_vectors_and_no_request(monkeypatch):
    monkeypatch.delenv("REDMOND_GEMINI_API_KEY", raising=False)
    assert embeddings.embed_query("привет") is None


def test_sync_computes_only_missing_or_changed(monkeypatch):
    calls = []
    monkeypatch.setenv("REDMOND_GEMINI_API_KEY", "k")
    monkeypatch.setattr(embeddings.requests, "post",
                        fake_post_factory(lambda t: [float(len(t)), 1.0], calls))
    assert embeddings.sync("tool", [("a", "первый"), ("b", "второй")]) == 2
    assert embeddings.sync("tool", [("a", "первый"), ("b", "второй")]) == 0
    assert embeddings.sync("tool", [("a", "первый"), ("b", "второй, но другой")]) == 1
    assert len(calls) == 2


def test_failed_api_leaves_the_cache_untouched(monkeypatch):
    monkeypatch.setenv("REDMOND_GEMINI_API_KEY", "k")
    monkeypatch.setattr(embeddings.requests, "post",
                        fake_post_factory(lambda t: [1.0], [], {"gemini-embedding-2": 500}))
    assert embeddings.sync("tool", [("a", "текст")]) == 0
    assert embeddings.load("tool") == {}


def test_nearest_orders_by_similarity(monkeypatch):
    monkeypatch.setenv("REDMOND_GEMINI_API_KEY", "k")
    vec = {"боль в животе": [1.0, 0.0], "дота вечером": [0.0, 1.0], "поджелудочная": [0.9, 0.1]}
    monkeypatch.setattr(embeddings.requests, "post",
                        fake_post_factory(lambda t: vec[t.split("text: ")[-1]], []))
    embeddings.sync("memory", [("1", "боль в животе"), ("2", "дота вечером"), ("3", "поджелудочная")])
    got = embeddings.nearest("memory", [1.0, 0.0], k=2)
    assert [ref for ref, _ in got] == ["1", "3"]


# ---------- recall ----------

REAL = [
    ("Проснулся в 12 Поел овсянку Снова болит живот Надо думаю к врачу ехатт",
     "Записала: проснулся в 12, овсянка, снова болит живот, думаешь ехать к врачу."),
    ("Курю кальян с Настей Чилю Ничего не планирую кроме чила Потом в доту пойду",
     "Записала: кальян с Настей, отдых и Дота."),
    ("(scheduled: утренний дайджест)", "**Утренний дайджест — вт, 8 сентября** …"),
]


def _seed_memory():
    conn = db.connect()
    conn.execute("CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "user TEXT NOT NULL, bot TEXT NOT NULL, important INTEGER DEFAULT 0, "
                 "timestamp REAL NOT NULL)")
    for i, (u, b) in enumerate(REAL):
        conn.execute("INSERT INTO memory(user, bot, timestamp) VALUES(?,?,?)", (u, b, 1.0 + i))
    conn.commit()


def _fake_vectors(monkeypatch):
    """Health talk points one way, leisure the other."""
    def vec(text):
        t = text.lower()
        if "живот" in t or "врач" in t or "боль" in t:
            return [1.0, 0.05]
        if "дот" in t or "кальян" in t:
            return [0.05, 1.0]
        return [0.5, 0.5]
    monkeypatch.setenv("REDMOND_GEMINI_API_KEY", "k")
    monkeypatch.setattr(embeddings.requests, "post", fake_post_factory(vec, []))


class FakeMem:
    """Full-text search that matches on any shared word - exactly the K10 problem."""
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, top_k=5):
        words = set(query.lower().split())
        out = []
        for r in self.rows:
            if words & set(r["user"].lower().split()):
                out.append({**r, "score": 0.9})
        return out[:top_k]


def _mem_rows():
    return [{"id": r["id"], "user": r["user"], "bot": r["bot"]}
            for r in db.query("SELECT id, user, bot FROM memory ORDER BY id")]


def test_bot_output_is_not_indexed(monkeypatch):
    _seed_memory()
    _fake_vectors(monkeypatch)
    recall.backfill(sleep=lambda s: None)
    assert set(embeddings.load("memory")) == {"1", "2"}


def test_meaning_finds_what_words_miss(monkeypatch):
    """'у меня боль, к доктору?' shares no word with 'болит живот, к врачу'."""
    _seed_memory()
    _fake_vectors(monkeypatch)
    recall.backfill(sleep=lambda s: None)
    q = embeddings.embed_query("у меня боль, к доктору?")
    docs = recall.recall(FakeMem(_mem_rows()), "у меня боль, к доктору?", q)
    assert docs and "болит живот" in docs[0]


def test_word_overlap_alone_no_longer_pulls_in_unrelated_memories(monkeypatch):
    """K10: sharing the word 'потом' with a leisure exchange is not relevance."""
    _seed_memory()
    _fake_vectors(monkeypatch)
    recall.backfill(sleep=lambda s: None)
    q = embeddings.embed_query("болит живот, что делать потом")
    docs = recall.recall(FakeMem(_mem_rows()), "болит живот, что делать потом", q)
    assert all("кальян" not in d for d in docs)


def test_without_vectors_the_old_full_text_path_is_kept():
    _seed_memory()
    docs = recall.recall(FakeMem(_mem_rows()), "кальян потом", None)
    assert docs and "кальян" in docs[0].lower()


def test_backfill_is_paced_and_survives_a_quota_hit(monkeypatch):
    """Sep 28, 2026: the first backfill sent 200 texts at once and got 429."""
    _seed_memory()
    monkeypatch.setenv("REDMOND_GEMINI_API_KEY", "k")
    answers = iter([429, 200, 200, 200])
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        status = next(answers)
        calls.append(len(json["requests"]))
        if status != 200:
            return FakeResponse(status, {"error": {"message": "quota"}})
        return FakeResponse(200, {"embeddings": [{"values": [1.0, 0.0]} for _ in json["requests"]]})

    monkeypatch.setattr(embeddings.requests, "post", post)
    sleeps = []
    done = recall.backfill(batch=1, pause=10, sleep=sleeps.append)
    assert done == 2 and set(embeddings.load("memory")) == {"1", "2"}
    assert max(calls) == 1, "batches larger than asked for"
    assert sleeps[0] == 20, "no longer wait after a quota hit"
