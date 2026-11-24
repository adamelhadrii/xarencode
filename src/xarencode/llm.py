"""Client for llama-server's OpenAI-compatible API (stdlib http.client, SSE streaming)."""
from __future__ import annotations

import http.client
import json
import re
import socket
import threading
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urlparse


class LlmError(Exception):
    pass


class ServerUnavailable(LlmError):
    pass


class ContextOverflow(LlmError):
    pass


class ServerError(LlmError):
    pass


class Interrupted(LlmError):
    pass


@dataclass
class TextDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolCallDelta:
    index: int
    id: str | None
    name: str | None
    arguments: str


@dataclass
class Done:
    finish_reason: str | None
    usage: dict | None = None
    timings: dict | None = None


@dataclass
class PromptProgress:
    total: int       # prompt tokens in this request
    cache: int       # tokens already in the KV cache (skipped)
    processed: int   # tokens evaluated so far, counting the cached ones
    time_ms: float = 0.0


def _error_from_body(status: int, body: bytes) -> LlmError:
    try:
        err = json.loads(body).get("error", {})
        if not isinstance(err, dict):
            err = {"message": str(err)}
    except (ValueError, AttributeError):
        err = {"message": body.decode("utf-8", "replace")[:300]}
    msg = err.get("message", "") or f"HTTP {status}"
    if status == 400 and (err.get("type") == "exceed_context_size_error"
                          or (re.search("context", msg, re.I) and re.search("exceed", msg, re.I))):
        return ContextOverflow(msg)
    return ServerError(f"HTTP {status}: {msg}")


def _msg(body: bytes) -> str:
    try:
        err = json.loads(body).get("error", {})
        return err.get("message", "") if isinstance(err, dict) else str(err)
    except (ValueError, AttributeError):
        return body.decode("utf-8", "replace")[:300]


class LlamaClient:
    def __init__(self, base_url: str):
        u = urlparse(base_url)
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 80
        self._lock = threading.Lock()
        self._conn: http.client.HTTPConnection | None = None
        self._cancelled = False

    # -- helpers
    def _connect(self, timeout: float) -> http.client.HTTPConnection:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.connect()
        except OSError as e:
            raise ServerUnavailable(f"cannot reach llama-server at {self.host}:{self.port} ({e})") from e
        return conn

    def health(self) -> str:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=2)
            try:
                conn.request("GET", "/health")
                status = conn.getresponse().status
            finally:
                conn.close()
        except OSError:
            return "down"
        return "ok" if status == 200 else "loading" if status == 503 else "down"

    def _body(self, messages, tools, max_tokens, sampling, stream, progress=False) -> bytes:
        s = dict(sampling)
        thinking = bool(s.pop("thinking", False))
        body = {
            "messages": messages, "stream": stream, "max_tokens": max_tokens, "min_p": 0,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if tools:
            body.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        body.update({k: s[k] for k in ("temperature", "top_p", "top_k", "min_p") if k in s})
        if stream:
            body["stream_options"] = {"include_usage": True}
            if progress:
                body["return_progress"] = True
        return json.dumps(body).encode()

    def _post(self, body: bytes):
        self._cancelled = False
        conn = self._connect(timeout=5)
        conn.sock.settimeout(600)  # prompt processing on CPU can take a long time before the first byte
        with self._lock:
            self._conn = conn
        try:
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
        except OSError as e:
            conn.close()
            if self._cancelled:
                raise Interrupted() from e
            raise ServerUnavailable(f"llama-server connection failed ({e})") from e
        if resp.status != 200:
            data = resp.read()
            conn.close()
            raise _error_from_body(resp.status, data)
        return conn, resp

    def cancel(self) -> None:
        """Thread-safe: shut the active socket so the reader raises Interrupted."""
        self._cancelled = True
        with self._lock:
            conn = self._conn
        if conn is not None and conn.sock is not None:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    # -- API
    def props(self) -> dict:
        """GET /props (model info, `is_sleeping`); raises ServerUnavailable / ServerError."""
        return self._request("GET", "/props")

    def slot(self, action: str, filename: str | None = None, slot_id: int = 0) -> dict:
        """POST /slots/<id>?action=save|restore|erase; save and restore take a file name inside --slot-save-path."""
        body = {"filename": filename} if filename else {}
        return self._request("POST", f"/slots/{slot_id}?action={action}", body)

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
        conn = self._connect(timeout=5)
        conn.sock.settimeout(timeout)
        try:
            payload = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, payload, {"Content-Type": "application/json"} if payload else {})
            resp = conn.getresponse()
            data = resp.read()
        except (OSError, http.client.HTTPException) as e:
            raise ServerUnavailable(f"llama-server connection failed ({e})") from e
        finally:
            conn.close()
        if resp.status != 200:
            raise ServerError(f"HTTP {resp.status}: {_msg(data)}")
        try:
            out = json.loads(data or b"{}")
        except ValueError as e:
            raise ServerError(f"bad response ({e})") from e
        return out if isinstance(out, dict) else {}

    def stream_chat(self, messages, tools, *, max_tokens, sampling, progress: bool = False) -> Iterator:
        conn, resp = self._post(self._body(messages, tools, max_tokens, sampling, True, progress))
        try:
            while True:
                try:
                    line = resp.readline()
                except (OSError, http.client.HTTPException) as e:
                    if self._cancelled:
                        raise Interrupted() from e
                    raise ServerUnavailable(f"stream broke ({e})") from e
                if not line:
                    if self._cancelled:
                        raise Interrupted()
                    yield Done(None)
                    return
                line = line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if "error" in obj:
                    err = obj["error"]
                    raise ServerError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
                yield from self._events(obj)
        finally:
            conn.close()
            with self._lock:
                self._conn = None

    @staticmethod
    def _events(obj: dict) -> Iterator:
        pp = obj.get("prompt_progress")
        if isinstance(pp, dict) and pp.get("total"):
            yield PromptProgress(int(pp["total"]), int(pp.get("cache", 0)), int(pp.get("processed", 0)), float(pp.get("time_ms", 0)))
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("reasoning_content"):
                yield ReasoningDelta(d["reasoning_content"])
            if d.get("content"):
                yield TextDelta(d["content"])
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") or {}
                yield ToolCallDelta(tc.get("index", 0), tc.get("id"), fn.get("name"), fn.get("arguments") or "")
            if ch.get("finish_reason"):
                yield Done(ch["finish_reason"], obj.get("usage"), obj.get("timings"))
        if not obj.get("choices") and (obj.get("usage") or obj.get("timings")):
            yield Done(None, obj.get("usage"), obj.get("timings"))

    def complete(self, messages, *, max_tokens, sampling, tools=None) -> str:
        conn, resp = self._post(self._body(messages, tools, max_tokens, sampling, False))
        try:
            try:
                data = json.loads(resp.read())
            except (OSError, ValueError) as e:
                if self._cancelled:
                    raise Interrupted() from e
                raise ServerError(f"bad response ({e})") from e
        finally:
            conn.close()
            with self._lock:
                self._conn = None
        if "error" in data:
            raise ServerError(str(data["error"]))
        return (data["choices"][0]["message"].get("content") or "").strip()
