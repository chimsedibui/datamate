"""OpenAI-compatible chat client (hosted API during development, local vLLM in the Docker image).

Configuration (environment or `.env` at the project root):
    LLM_BASE_URL   e.g. https://api.example.com/v1  or  http://localhost:8000/v1
    LLM_API_KEY    API key ("EMPTY" for local vLLM)
    LLM_MODEL      model name
    LLM_FALLBACK_BASE_URL / LLM_FALLBACK_API_KEY / LLM_FALLBACK_MODEL
                   optional second endpoint, used for a call when the primary stays rate-limited or erroring
    LLM_ENABLE_THINKING  optional "0"/"1": sets chat_template_kwargs.enable_thinking (Qwen3 on vLLM)
    BUILD_LLM_BASE_URL / BUILD_LLM_API_KEY / BUILD_LLM_MODEL
                   optional separate endpoint for semantic-layer builds (LLM.for_build()); QA always uses LLM_*
    LLM_BUDGET_USD optional spend cap for this process (with LLM_PRICE_IN / LLM_PRICE_OUT in USD per 1M tokens);
                   once reached, new (uncached) calls raise BudgetExceeded
    LLM_CACHE_DIR  optional, default .cache/llm — responses are cached on disk by request hash,
                   which makes re-runs deterministic and free.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


def _create(client: OpenAI, req: dict):
    """chat.completions.create, streamed and reassembled into a non-streaming-shaped response.

    Long non-streamed generations (thinking builds run minutes) get cut by idle timeouts on some hosted
    gateways, which leaves the request hanging until the client timeout and retrying forever; streaming keeps
    bytes flowing. `req` itself (the cache key) is unchanged."""
    if os.environ.get("LLM_STREAM", "1") == "0":
        return client.chat.completions.create(**req)
    content, reasoning, usage = [], [], None
    for ch in client.chat.completions.create(**req, stream=True, stream_options={"include_usage": True}):
        if getattr(ch, "usage", None):
            usage = ch.usage
        for c in ch.choices or []:
            d = c.delta
            if d is None:
                continue
            if d.content:
                content.append(d.content)
            r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
            if r is None and getattr(d, "model_extra", None):
                r = d.model_extra.get("reasoning_content") or d.model_extra.get("reasoning")
            if r:
                reasoning.append(r)
    msg = SimpleNamespace(content="".join(content), reasoning_content="".join(reasoning) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _reasoning(msg) -> str | None:
    return getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)


class BudgetExceeded(RuntimeError):
    pass


_SPENT = [0.0]
_SPENT_LOCK = threading.Lock()
_CTX_ERR = re.compile(r"maximum context length is (\d+) tokens and your request has (\d+) input tokens")


def spent_usd() -> float:
    return _SPENT[0]


@dataclass
class ChatResult:
    content: str
    reasoning: str | None = None
    usage: dict = field(default_factory=dict)
    cached: bool = False


class LLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 cache_dir: str | None = None):
        self.model = model or os.environ["LLM_MODEL"]
        self.client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL"),
                             api_key=api_key or os.environ.get("LLM_API_KEY", "EMPTY"),
                             timeout=600, max_retries=3)
        self.max_out = int(os.environ.get("BUILD_LLM_MAX_OUTPUT" if model else "LLM_MAX_OUTPUT") or 0)
        self.think_env = None if model else os.environ.get("LLM_ENABLE_THINKING")
        self.fallback = None
        if os.environ.get("LLM_FALLBACK_MODEL") and not model:
            self.fallback = (os.environ["LLM_FALLBACK_MODEL"],
                             OpenAI(base_url=os.environ.get("LLM_FALLBACK_BASE_URL"),
                                    api_key=os.environ.get("LLM_FALLBACK_API_KEY", "EMPTY"), timeout=600, max_retries=2))
        self.cache = Path(cache_dir or os.environ.get("LLM_CACHE_DIR", ROOT / ".cache" / "llm"))
        self.cache.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_build(cls) -> "LLM":
        """Client for semantic-layer builds: BUILD_LLM_* when set, else the default LLM_* endpoint."""
        if os.environ.get("BUILD_LLM_MODEL"):
            return cls(os.environ["BUILD_LLM_MODEL"], os.environ.get("BUILD_LLM_BASE_URL"),
                       os.environ.get("BUILD_LLM_API_KEY"))
        return cls()

    def _reasoning_model(self) -> bool:
        return bool(re.match(r"(gpt-5|o\d)", self.model.split("/")[-1]))

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 8000,
             seed: int = 42, thinking: bool | None = None, **extra) -> ChatResult:
        """`thinking` overrides LLM_ENABLE_THINKING for this call; it only has an effect when that variable
        is set (i.e. for Qwen3-style models that support the switch)."""
        if self.max_out:
            max_tokens = min(max_tokens, self.max_out)
        if self._reasoning_model():  # OpenAI reasoning models: no temperature/seed, max_completion_tokens
            req = dict(model=self.model, messages=messages, max_completion_tokens=max_tokens,
                       reasoning_effort="medium" if thinking else "low", **extra)
            return self._run(req)
        think = self.think_env
        if think in ("0", "1") and thinking is not None:
            think = "1" if thinking else "0"
        if think in ("0", "1") and "extra_body" not in extra:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": think == "1"}}
        if think == "0":  # Qwen3 soft switch; some hosted providers ignore chat_template_kwargs
            messages = [{**m, "content": m["content"] + " /no_think"} if m["role"] == "user" else m
                        for m in messages]
        req = dict(model=self.model, messages=messages, temperature=temperature,
                   max_tokens=max_tokens, seed=seed, **extra)
        return self._run(req)

    def _run(self, req: dict) -> ChatResult:
        budget = float(os.environ.get("LLM_BUDGET_USD") or 0)
        key = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        fp = self.cache / f"{key}.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            return ChatResult(d["content"], d.get("reasoning"), d.get("usage", {}), cached=True)
        if budget and _SPENT[0] >= budget:
            raise BudgetExceeded(f"LLM budget ${budget} reached (spent ${_SPENT[0]:.2f})")
        t = time.time()
        r, served_by = None, self.model
        for attempt in range(4 if self.fallback else 8):
            try:
                r = _create(self.client, req)
                if r.choices and (r.choices[0].message.content or _reasoning(r.choices[0].message)):
                    break  # reasoning only = thinking used the whole output budget: the caller decides what to do
                r = None  # free endpoints sometimes return an empty completion
            except (RateLimitError, APIConnectionError) as e:
                if "insufficient_quota" in str(e) or "credit" in str(e).lower():
                    raise
                print(f"[llm] attempt {attempt}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
            except APIStatusError as e:
                print(f"[llm] attempt {attempt}: HTTP {e.status_code}: {str(e)[:200]}", file=sys.stderr, flush=True)
                # vLLM rejects prompt + max_tokens > context; our token estimate can be short (Vietnamese, code),
                # so retry once with the output budget that actually fits.
                m = _CTX_ERR.search(str(e))
                if e.status_code == 400 and m and "max_tokens" in req:
                    fit = int(m.group(1)) - int(m.group(2)) - 16
                    if 1024 <= fit < req["max_tokens"]:
                        req = {**req, "max_tokens": fit}
                        continue
                if e.status_code < 500:
                    raise
            time.sleep(min(60, 5 * 2 ** attempt))
        if r is None and self.fallback:
            served_by, client = self.fallback
            for attempt in range(6):
                try:
                    r = _create(client, {**req, "model": served_by})
                    if r.choices and r.choices[0].message.content:
                        break
                    r = None
                except (RateLimitError, APIConnectionError):
                    pass
                except APIStatusError as e:
                    if e.status_code < 500:
                        raise
                time.sleep(min(60, 5 * 2 ** attempt))
        if r is None:
            raise RuntimeError(f"LLM call failed after retries (model {self.model})")
        msg = r.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        content = msg.content or ""
        m = re.match(r"\s*<think>(.*?)</think>\s*", content, re.S)  # no reasoning parser: thinking inline
        if m:
            reasoning, content = reasoning or m.group(1).strip(), content[m.end():]
        usage = r.usage.model_dump() if r.usage else {}
        cost = (usage.get("prompt_tokens", 0) * float(os.environ.get("LLM_PRICE_IN") or 0)
                + usage.get("completion_tokens", 0) * float(os.environ.get("LLM_PRICE_OUT") or 0)) / 1e6
        with _SPENT_LOCK:
            _SPENT[0] += cost
        usage["latency_s"] = round(time.time() - t, 2)
        usage["served_by"] = served_by
        res = ChatResult(content, reasoning, usage)
        # write-then-rename: another thread asking the same request never reads a half-written file
        tmp = fp.with_name(f"{fp.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"content": res.content, "reasoning": reasoning, "usage": usage,
                                   "request": req}, ensure_ascii=False))
        os.replace(tmp, fp)
        return res
