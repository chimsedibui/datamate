"""OpenAI-compatible chat client (hosted API during development, local vLLM in the Docker image).

Configuration (environment or `.env` at the project root):
    LLM_BASE_URL   e.g. https://api.example.com/v1  or  http://localhost:8000/v1
    LLM_API_KEY    API key ("EMPTY" for local vLLM)
    LLM_MODEL      model name
    LLM_CACHE_DIR  optional, default .cache/llm — responses are cached on disk by request hash,
                   which makes re-runs deterministic and free.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


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
        self.cache = Path(cache_dir or os.environ.get("LLM_CACHE_DIR", ROOT / ".cache" / "llm"))
        self.cache.mkdir(parents=True, exist_ok=True)

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 8000,
             seed: int = 42, **extra) -> ChatResult:
        req = dict(model=self.model, messages=messages, temperature=temperature,
                   max_tokens=max_tokens, seed=seed, **extra)
        key = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        fp = self.cache / f"{key}.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            return ChatResult(d["content"], d.get("reasoning"), d.get("usage", {}), cached=True)
        t = time.time()
        r = self.client.chat.completions.create(**req)
        msg = r.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        usage = r.usage.model_dump() if r.usage else {}
        usage["latency_s"] = round(time.time() - t, 2)
        res = ChatResult(msg.content or "", reasoning, usage)
        fp.write_text(json.dumps({"content": res.content, "reasoning": reasoning, "usage": usage,
                                  "request": req}, ensure_ascii=False))
        return res
