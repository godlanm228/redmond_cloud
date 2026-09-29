"""Монитор моделей: снятое и вышедшее у провайдеров видно без ручного разбора.

Каталоги ниже — ответы /models Groq и Gemini для нашего ключа на 28.09.2026
(без tts/whisper/guard/image). В этот день `qwen/qwen3.6-27b` уже две недели
как был снят, а мы узнали об этом, только разбирая логи руками.
"""

from types import SimpleNamespace

from utils import model_catalog as mc
from utils import model_healthcheck as mh

GROQ_28_09 = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
GEMINI_28_09 = [
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest", "gemini-flash-lite-latest",
    "gemini-pro-latest", "gemini-2.5-flash-lite", "gemini-3-flash-preview",
    "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-omni-1.1-flash", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash",
]


def test_family_reads_versions():
    assert mc.family("qwen/qwen3.8-27b") == ("qwen/qwen*-27b", (3, 8), False)
    assert mc.family("gemini-3.1-flash-lite") == ("gemini-*-flash-lite", (3, 1), False)
    assert mc.family("gemini-3-flash-preview")[2] is True
    assert mc.family("openai/gpt-oss-120b") is None


def test_successor_is_found_in_the_same_family():
    assert mc.successor("qwen/qwen3.6-27b", GROQ_28_09) == "qwen/qwen3.8-27b"
    assert mc.successor("gemini-3.6-flash", GEMINI_28_09) == "gemini-3.8-flash"
    assert mc.successor("gemini-3.1-flash-lite", GEMINI_28_09) == "gemini-3.5-flash-lite"


def test_preview_is_not_offered_as_successor():
    assert mc.successor("gemini-2.5-pro", GEMINI_28_09) is None


def test_review_on_the_real_28_09_state():
    configured = {
        "groq_model": ("groq", "openai/gpt-oss-120b"),
        "groq_fallback_model": ("groq", "qwen/qwen3.6-27b"),
        "gemini_model": ("gemini", "gemini-3.6-flash"),
    }
    found = {f.role: f for f in mc.review(configured, {"groq": GROQ_28_09, "gemini": GEMINI_28_09})}
    assert found["groq_fallback_model"].gone and found["groq_fallback_model"].newer == "qwen/qwen3.8-27b"
    assert not found["gemini_model"].gone and found["gemini_model"].newer == "gemini-3.8-flash"
    assert "groq_model" not in found


def test_unreachable_catalog_is_not_read_as_removed():
    configured = {"groq_fallback_model": ("groq", "qwen/qwen3.6-27b")}
    assert mc.review(configured, {"groq": None}) == []


def test_removed_fallback_is_replaced_but_primary_is_not(monkeypatch):
    monkeypatch.setattr(mc, "list_groq", lambda key: GROQ_28_09)
    monkeypatch.setattr(mc, "list_gemini", lambda key: ["gemini-3.8-flash"])
    config = SimpleNamespace(groq_api_key="k", gemini_api_key="k",
                             groq_model="openai/gpt-oss-120b",
                             groq_fallback_model="qwen/qwen3.6-27b",
                             gemini_model="gemini-3.6-flash")
    findings = mh.review_catalog(config)
    assert config.groq_fallback_model == "qwen/qwen3.8-27b", "резерв остался мёртвым"
    assert config.gemini_model == "gemini-3.6-flash", "основную модель сменили без владельца"
    text = mh.describe_findings(findings)
    assert "qwen/qwen3.6-27b" in text and "qwen/qwen3.8-27b" in text
    assert "gemini-3.8-flash" in text


def test_gemini_models_are_checked_without_spending_their_quota(monkeypatch):
    """Sep 29, 2026: a start pinged every Gemini model of every pool with a
    generation request - one of its 20 a day, several restarts a day."""
    from types import SimpleNamespace
    from utils import gemini, groq, llm_gate, model_catalog, model_healthcheck
    llm_gate.configure(pools={"chat_groq": ["openai/gpt-oss-120b"],
                              "chat_gemini": ["gemini-3.6-flash", "gemini-2.5-flash-lite", "gemini-9-gone"]})
    generated = []
    monkeypatch.setattr(gemini, "generate", lambda *a, **kw: generated.append(kw.get("model")))
    monkeypatch.setattr(groq, "chat", lambda m, msgs, **kw: ({"choices": []}, ""))
    monkeypatch.setattr(model_catalog, "list_gemini",
                        lambda key: ["gemini-3.6-flash", "gemini-2.5-flash-lite"])
    llm_gate.report("gemini-2.5-flash-lite", 404)  # listed, yet «no longer available»
    config = SimpleNamespace(groq_api_key="k", gemini_api_key="k", groq_model="",
                             groq_fallback_model="", gemini_model="")
    status = {m: s for _p, m, s, _d in model_healthcheck.check_models(config)}
    assert generated == [], "a Gemini model was pinged with a generation request"
    assert status["gemini-3.6-flash"] == model_healthcheck.OK
    assert status["gemini-2.5-flash-lite"] == model_healthcheck.GONE
    assert status["gemini-9-gone"] == model_healthcheck.GONE
    assert status["openai/gpt-oss-120b"] == model_healthcheck.OK
