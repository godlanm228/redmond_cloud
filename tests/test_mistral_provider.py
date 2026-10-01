"""Mistral through the same OpenAI-compatible client as Groq.

Checked before connecting (Oct 1, 2026, official terms): the free plan trains
on prompts unless switched off in the console; «labs»/preview models always
train. Data stays in the EU by default, 30 days for abuse monitoring.
"""

from types import SimpleNamespace

from utils import groq, llm_gate


def _capture(monkeypatch):
    sent = {}

    def post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, json=json)
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {
            "choices": [{"message": {"content": "ok"}}]}, text="")

    monkeypatch.setattr(groq.requests, "post", post, raising=False)
    return sent


def test_a_mistral_model_goes_to_mistral_with_its_own_key(monkeypatch):
    monkeypatch.setenv("REDMOND_MISTRAL_API_KEY", "m-key")
    sent = _capture(monkeypatch)
    completion, err = groq.chat("mistral/mistral-large-latest", [{"role": "user", "content": "hi"}],
                                api_key="groq-key", extra={"reasoning_effort": "low"})
    assert err == "" and groq.text_of(completion) == "ok"
    assert sent["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer m-key", "не ключ Groq"
    assert sent["json"]["model"] == "mistral-large-latest"
    assert "reasoning_effort" not in sent["json"]


def test_a_forced_tool_becomes_any(monkeypatch):
    monkeypatch.setenv("REDMOND_MISTRAL_API_KEY", "m-key")
    sent = _capture(monkeypatch)
    tool = {"type": "function", "function": {"name": "delegate_research", "parameters": {}}}
    groq.chat("mistral/mistral-large-latest", [{"role": "user", "content": "x"}], tools=[tool],
              tool_choice={"type": "function", "function": {"name": "delegate_research"}})
    assert sent["json"]["tool_choice"] == "any"


def test_labs_models_and_a_missing_key_are_refused(monkeypatch):
    monkeypatch.setenv("REDMOND_MISTRAL_API_KEY", "m-key")
    _, err = groq.chat("mistral/labs-devstral", [{"role": "user", "content": "x"}])
    assert "always train" in err
    monkeypatch.delenv("REDMOND_MISTRAL_API_KEY")
    _, err = groq.chat("mistral/mistral-large-latest", [{"role": "user", "content": "x"}])
    assert err == "no Mistral API key"


def test_groq_is_untouched(monkeypatch):
    sent = _capture(monkeypatch)
    groq.chat("openai/gpt-oss-120b", [{"role": "user", "content": "x"}], api_key="g")
    assert "api.groq.com" in sent["url"] and sent["json"]["model"] == "openai/gpt-oss-120b"
    assert llm_gate.provider_of("mistral/mistral-large-latest") == "mistral"
    assert llm_gate.provider_of("openai/gpt-oss-120b") == "groq"
