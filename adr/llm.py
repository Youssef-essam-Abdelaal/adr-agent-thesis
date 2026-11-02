"""
One small client for every model call in the system.

Images in, JSON out, with retries, an optional disk cache and per-component
token accounting. Each component tags its calls ("agent", "judge", "diag",
"ground" for the element finder) so cost overhead can be reported per component.

Where a model runs is set per role in the config (<role>_base_url):
  ""                          OpenAI (the default; uses the key)
  "http://host:port/v1"       any OpenAI-compatible server, e.g. vLLM serving UI-TARS
  "ollama:http://host:port"   Ollama's own API (context length and temperature can be set per call)
ADR_NO_OPENAI=1 makes any call that would reach OpenAI fail, so an open-model run cannot spend money by mistake.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from urllib import request as urlrequest

from PIL import Image

ImageLike = Union[str, Path, bytes, Image.Image]

# when set, called with a record of every ask() (prompt, images, reply, tokens, cost), so that the offline replay
# (adr/v3/replay.py) can keep the full history of a run
CALL_HOOK = None

# process-local usage counters, keyed by tag
USAGE: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))


def reset_usage() -> None:
    USAGE.clear()


def usage_snapshot() -> Dict[str, Dict[str, float]]:
    return {k: dict(v) for k, v in USAGE.items()}


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    f = Path.home() / ".openai_key"
    if f.exists():
        return f.read_text().strip()
    raise RuntimeError("no OpenAI key: set OPENAI_API_KEY or write it to ~/.openai_key")


def to_png_bytes(img: ImageLike) -> bytes:
    if isinstance(img, bytes):
        return img
    if isinstance(img, (str, Path)):
        return Path(img).read_bytes()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"_parse_error": True, "_raw": (text or "")[:500]}


class LLM:
    def __init__(self, model: str, effort: str = "low", max_tokens: int = 6000,
                 detail: str = "high", cache_dir: Optional[str] = None, timeout: float = 120,
                 base_url: str = "", ctx: int = 8192, temperature: float = 0.0, max_side: int = 0):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.detail = detail
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.base_url = base_url or ""
        self.ollama = self.base_url.startswith("ollama:")
        self.openrouter = "openrouter.ai" in self.base_url     # a paid gateway with its own key
        self.ctx, self.temperature, self.max_side = ctx, temperature, max_side
        if self.base_url:
            self.timeout = max(timeout, 300)        # self-hosted open models are slower
        self._client = None

    @classmethod
    def for_role(cls, cfg, role: str, **kw) -> "LLM":
        """The model of one role (agent, judge, diag, ground) as set in the config."""
        kw.setdefault("detail", cfg.image_detail)
        return cls(getattr(cfg, f"{role}_model"), getattr(cfg, f"{role}_effort"),
                   base_url=getattr(cfg, f"{role}_base_url"), ctx=cfg.local_ctx,
                   temperature=cfg.local_temperature, max_side=cfg.local_max_side, **kw)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            if self.base_url:
                key = "EMPTY"
                if self.openrouter:
                    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
                    if not key:
                        raise RuntimeError("no OpenRouter key: set OPENROUTER_API_KEY")
                self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout, max_retries=0)
            else:
                if os.environ.get("ADR_NO_OPENAI") == "1":
                    raise RuntimeError(f"ADR_NO_OPENAI=1 but {self.model!r} has no base_url, so it would call OpenAI")
                self._client = OpenAI(api_key=api_key(), timeout=self.timeout, max_retries=0)
        return self._client

    def ask(self, system: str, text: str, images: Iterable[ImageLike] = (), tag: str = "misc",
            schema: Optional[dict] = None) -> dict:
        """schema: a JSON schema the reply must follow; only Ollama uses it (its plain JSON mode can mangle keys)."""
        if os.environ.get("ADR_FAKE_LLM") == "1":     # dry run: no API, no key (adr/fake_llm.py)
            from .fake_llm import reply
            u = USAGE[tag]
            u["calls"] += 1
            u["prompt_tokens"] += 1000
            u["completion_tokens"] += 100
            out = reply(system, text, tag)
            if CALL_HOOK is not None:
                CALL_HOOK({"tag": tag, "model": self.model, "system": system, "text": text,
                           "pngs": [to_png_bytes(i) for i in images], "reply": json.dumps(out), "fake": True})
            return out
        pngs = [to_png_bytes(i) for i in images]
        key = None
        if self.cache_dir:
            h = hashlib.sha256()
            for part in (self.model, self.effort, self.base_url, system, text):
                h.update(part.encode())
            for p in pngs:
                h.update(hashlib.sha256(p).digest())
            key = self.cache_dir / f"{h.hexdigest()}.json"
            if key.exists():
                return json.loads(key.read_text())

        if self.ollama:
            out = self._ask_ollama(system, text, pngs, tag, schema)
            if key and not out.get("_parse_error"):
                key.parent.mkdir(parents=True, exist_ok=True)
                key.write_text(json.dumps(out))
            return out
        content = [{"type": "text", "text": text}]
        for p in pngs:
            url = "data:image/png;base64," + base64.b64encode(p).decode()
            content.append({"type": "image_url", "image_url": {"url": url, "detail": self.detail}})
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_completion_tokens=self.max_tokens,
        )
        if self.openrouter:                     # the same request as to OpenAI, through OpenRouter's gateway
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = self.max_tokens
            extra = {"usage": {"include": True}}           # the real cost of each call
            prov = self.model.split("/", 1)[0] if "/" in self.model else ""
            if prov:                                       # the model's own maker (openai, anthropic), no fallback
                extra["provider"] = {"order": [prov], "allow_fallbacks": False}
            if self.effort:
                extra["reasoning"] = {"effort": self.effort}
            else:                                          # no reasoning: fixed temperature, as for gpt-4o below
                kwargs["temperature"] = self.temperature
            if self.model.startswith("anthropic/"):       # prompt caching: the same replies, the long system
                extra["cache_control"] = {"type": "ephemeral"}   # prompt costs 0.1x when repeated
            kwargs["extra_body"] = extra
        elif self.base_url:                     # open model: no reasoning effort, fixed temperature
            kwargs.pop("max_completion_tokens")
            kwargs.update(max_tokens=min(self.max_tokens, 2048), temperature=self.temperature)
        elif self.effort:
            kwargs["reasoning_effort"] = self.effort
        else:                                   # an API model without reasoning (such as gpt-4o): fixed temperature,
            kwargs["temperature"] = self.temperature   # as for the open models (default 0)

        for attempt in range(3 if self.openrouter else 1):
            t0 = time.time()
            r = self._create(kwargs)
            u = USAGE[tag]
            u["calls"] += 1
            u["seconds"] += time.time() - t0
            if r.usage:
                u["prompt_tokens"] += r.usage.prompt_tokens or 0
                u["completion_tokens"] += r.usage.completion_tokens or 0
                if getattr(r.usage, "cost", None) is not None:
                    u["cost_usd"] += float(r.usage.cost)
            raw = r.choices[0].message.content or ""
            out = parse_json(raw)
            if CALL_HOOK is not None:
                CALL_HOOK({"tag": tag, "model": self.model, "system": system, "text": text,
                           "pngs": pngs if attempt == 0 else [], "format_retry": attempt,
                           "reply": raw, "finish_reason": r.choices[0].finish_reason,
                           "usage": r.usage.model_dump() if hasattr(r.usage, "model_dump") else None,
                           "provider": getattr(r, "provider", None), "served_model": getattr(r, "model", None),
                           "id": getattr(r, "id", None), "seconds": round(time.time() - t0, 2),
                           "sent": {k: v for k, v in kwargs.items() if k != "messages"}})
            if not out.get("_parse_error"):
                break
            # a reply that is not the JSON asked for (Claude sometimes writes its own tool-call markup) is shown
            # back with a request for the JSON only; OpenAI's JSON mode never needed this
            kwargs = dict(kwargs, messages=kwargs["messages"] + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your reply was not in the required format. Reply again with the JSON "
                                            "only, as the instructions describe."}])
        if key and not out.get("_parse_error"):
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_text(json.dumps(out))
        return out

    def chat(self, messages: List[dict], tag: str = "misc", max_tokens: int = 1000) -> str:
        """A multi-turn call that returns plain text (for agents with their own output format, such as UI-TARS).
        Images go in as OpenAI-style image_url parts with data URLs."""
        if os.environ.get("ADR_FAKE_LLM") == "1":
            from .fake_llm import reply_text
            u = USAGE[tag]
            u["calls"] += 1
            u["prompt_tokens"] += 1000
            u["completion_tokens"] += 100
            return reply_text(messages, tag)
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_tokens)
        if self.base_url:
            kwargs["temperature"] = self.temperature
        t0 = time.time()
        r = self._create(kwargs)
        u = USAGE[tag]
        u["calls"] += 1
        u["seconds"] += time.time() - t0
        if r.usage:
            u["prompt_tokens"] += r.usage.prompt_tokens or 0
            u["completion_tokens"] += r.usage.completion_tokens or 0
        return r.choices[0].message.content or ""

    def _ask_ollama(self, system: str, text: str, pngs: List[bytes], tag: str, schema: Optional[dict] = None) -> dict:
        """Ollama's own chat API: JSON output, and the context length and temperature set per call."""
        body = {"model": self.model, "stream": False, "format": schema or "json", "keep_alive": "30m",
                "options": {"num_ctx": self.ctx, "temperature": self.temperature},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": text,
                              "images": [base64.b64encode(self._shrink(p)).decode() for p in pngs]}]}
        url = self.base_url[len("ollama:"):].rstrip("/") + "/api/chat"
        data = json.dumps(body).encode()
        t0, delay = time.time(), 2.0
        for attempt in range(6):
            try:
                req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                    r = json.loads(resp.read().decode())
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        u = USAGE[tag]
        u["calls"] += 1
        u["seconds"] += time.time() - t0
        u["prompt_tokens"] += r.get("prompt_eval_count") or 0
        u["completion_tokens"] += r.get("eval_count") or 0
        return parse_json((r.get("message") or {}).get("content", ""))

    def _shrink(self, png: bytes) -> bytes:
        """Big screenshots (OSWorld is 1920x1080) cost thousands of image tokens; shrink them for local models."""
        if not self.max_side:
            return png
        im = Image.open(io.BytesIO(png))
        if max(im.size) <= self.max_side:
            return png
        im.thumbnail((self.max_side, self.max_side))
        return to_png_bytes(im.convert("RGB"))

    def _create(self, kwargs):
        import openai
        delay = 2.0
        for attempt in range(6):
            try:
                return self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError as e:
                msg = str(e)
                if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort")
                    continue
                raise
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                    openai.InternalServerError):
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
