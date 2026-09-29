"""Text in, text out, through whichever model of a pool can take it now.

For calls without tools: the router's one-word decision, digest translation,
composing an answer from gathered results. The model is picked by
utils.llm_gate (limits learned from the providers); the provider is picked by
the model id. Callers name a job ("router", "compose", "background"), not a
model — model ids live in config and in the gate's pools, where the weekly
model check sees them.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

from utils import llm_gate

logger = logging.getLogger(__name__)

# gpt-oss always reasons, and the reasoning counts against max_tokens: a
# 20-token budget for a one-word answer comes back empty.
_REASONING_ROOM = 400


def complete(model: str, prompt: str, *, system: str = "", max_tokens: int = 400,
             temperature: float = 0.3, api_key: str = "") -> str:
    """One call to one model. '' on any failure (the gate has been told)."""
    if llm_gate.provider_of(model) == "gemini":
        from utils import gemini
        return gemini.generate_text(prompt, system=system, model=model, temperature=temperature,
                                    max_tokens=max_tokens, api_key=api_key)
    from utils import groq
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    extra = None
    if model.startswith("openai/gpt-oss"):
        extra = {"reasoning_effort": "low"}
        max_tokens += _REASONING_ROOM
    completion, _err = groq.chat(model, messages, temperature=temperature,
                                 max_tokens=max_tokens, api_key=api_key, extra=extra)
    return groq.text_of(completion)


def text(job: Union[str, Sequence[str]], prompt: str, *, system: str = "",
         max_tokens: int = 400, temperature: float = 0.3, priority: str = llm_gate.OWNER,
         max_wait: float = 0.0, exclude: Sequence[str] = (), sleep=None) -> Tuple[str, str]:
    """(answer, model that gave it) or ('', '') if no model of the job's pool
    could answer. Models the gate knows to be out are not called at all."""
    models = llm_gate.pool(job) if isinstance(job, str) else list(job)
    models = [m for m in models if m not in set(exclude)]
    tokens = llm_gate.estimate_tokens(system, prompt)
    tried: List[str] = []
    while True:
        kwargs = {"sleep": sleep} if sleep else {}
        model = llm_gate.acquire([m for m in models if m not in tried], tokens, priority,
                                 max_wait, **kwargs)
        if model is None:
            if tried:
                logger.warning("Ни одна модель из %s не ответила (%s)", job, ", ".join(tried))
            return "", ""
        tried.append(model)
        out = complete(model, prompt, system=system, max_tokens=max_tokens,
                       temperature=temperature)
        if out:
            if len(tried) > 1:
                logger.info("%s: ответила %s (до неё: %s)", job, model, ", ".join(tried[:-1]))
            return out, model
