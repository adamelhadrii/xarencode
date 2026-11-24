#!/usr/bin/env python3
"""A tiny stand-in for llama-server: scripted SSE replies, health, and fake model files.

Script format: a JSON list of responses; each response is a list of steps consumed by one
POST /v1/chat/completions.  Steps: {"text"}, {"reasoning"}, {"tool_call": {"name", "arguments"}},
{"text_tool_call": {"name", "arguments"}}, {"finish_reason"}, {"error": {"status","type","message"}},
{"delay": seconds}.  Unknown flags are accepted and ignored so it can be used as server.binary.
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FILE_SIZE = 200_000


def file_bytes(size=FILE_SIZE):
    return bytes((i * 31 + 7) & 0xFF for i in range(size))


def demo_steps(messages):
    last = messages[-1] if messages else {}
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if last.get("role") == "tool":
        if "Created" in last["content"] or "Edited" in last["content"]:
            return [{"text": "Done! I wrote the file for you."}]
        return [{"text": "I looked around. Here is what I found:\n\n```\n" + last["content"][:300] + "\n```\nAnything else?"}]
    if "write" in user.lower() or "create" in user.lower():
        args = json.dumps({"path": "hello.py", "content": "print('hello from xaren')\n"})
        return [{"text": "Sure, writing it now. "}, {"tool_call": {"name": "write_file", "arguments": args}}]
    if "hi" == user.strip().lower():
        return [{"text": "Hello!"}]
    return [{"text": "Let me look at the folder. "}, {"tool_call": {"name": "list_dir", "arguments": json.dumps({"path": "."})}}]


class State:
    def __init__(self, script=None, demo=False, health_503=0, file_size=FILE_SIZE):
        self.script = list(script or [])
        self.demo = demo
        self.health_503 = health_503
        self.file_size = file_size
        self.requests = []
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: State

    def log_message(self, *a):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    # -- plumbing
    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, data: bytes):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _sse(self, obj):
        self._chunk(b"data: " + (obj if isinstance(obj, str) else json.dumps(obj)).encode() + b"\n\n")

    def _file_route(self):
        name = self.path.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        if (self.path.startswith("/files/") or "/resolve/main/" in self.path) and name.endswith(".gguf"):
            return name
        return None

    # -- GET/HEAD
    def do_GET(self):
        self._get(head=False)

    def do_HEAD(self):
        self._get(head=True)

    def _get(self, head):
        st = self.state
        if self.path == "/health":
            with st.lock:
                if st.health_503 > 0:
                    st.health_503 -= 1
                    return self._json(503, {"error": {"code": 503, "message": "Loading model"}})
            return self._json(200, {"status": "ok"})
        if self.path == "/props":
            return self._json(200, {"model": "mock", "n_ctx": 4096})
        if self._file_route():
            data = file_bytes(st.file_size)
            start = 0
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start = int(rng[6:].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = data[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Linked-Size", str(len(data)))
            self.send_header("X-Linked-ETag", '"%s"' % hashlib.sha256(data).hexdigest())
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            self.end_headers()
            if not head:
                self.wfile.write(body)
            return
        self._json(404, {"error": {"message": "not found"}})

    # -- POST
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        st = self.state
        with st.lock:
            st.requests.append(body)
            if st.script:
                steps = st.script.pop(0)
            elif st.demo:
                steps = demo_steps(body.get("messages", []))
            else:
                steps = [{"text": "ok"}]
        err = next((s["error"] for s in steps if "error" in s), None)
        if err:
            return self._json(err.get("status", 500), {"error": {"code": err.get("status", 500), "type": err.get("type", "server_error"),
                                                                 "message": err.get("message", "boom")}})
        prompt_tokens = sum(len(json.dumps(m)) for m in body.get("messages", [])) // 3
        if not body.get("stream"):
            text = "".join(s.get("text", "") for s in steps)
            return self._json(200, {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(text) // 4}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._stream(steps, prompt_tokens)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _delta(self, delta, finish=None):
        return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def _stream(self, steps, prompt_tokens):
        finish, produced, idx = "stop", 0, 0
        for s in steps:
            if "delay" in s:
                time.sleep(s["delay"])
            elif "text" in s or "text_tool_call" in s:
                text = s.get("text")
                if text is None:
                    t = s["text_tool_call"]
                    text = "<tool_call>\n" + json.dumps({"name": t["name"], "arguments": t["arguments"]}) + "\n</tool_call>"
                for i in range(0, len(text), 4):
                    self._sse(self._delta({"content": text[i:i + 4]}))
                    produced += 1
            elif "reasoning" in s:
                self._sse(self._delta({"reasoning_content": s["reasoning"]}))
            elif "tool_call" in s:
                tc = s["tool_call"]
                args = tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"])
                self._sse(self._delta({"tool_calls": [{"index": idx, "id": f"call_{idx + 1}", "type": "function",
                                                       "function": {"name": tc["name"], "arguments": ""}}]}))
                for i in range(0, len(args), 8):
                    self._sse(self._delta({"tool_calls": [{"index": idx, "function": {"arguments": args[i:i + 8]}}]}))
                    produced += 1
                idx += 1
                finish = "tool_calls"
            elif "finish_reason" in s:
                finish = s["finish_reason"]
        self._sse(self._delta({}, finish))
        self._sse({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": produced},
                   "timings": {"predicted_per_second": 12.5, "prompt_per_second": 60.0}})
        self._sse("[DONE]")
        self._chunk(b"")


class MockServer:
    """In-process server for tests."""

    def __init__(self, script=None, **kw):
        self.state = State(script, **kw)
        handler = type("H", (Handler,), {"state": self.state})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    @property
    def requests(self):
        return self.state.requests

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--script")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--health-503", type=int, default=0)
    ap.add_argument("--file-size", type=int, default=FILE_SIZE)
    args, _ = ap.parse_known_args(argv)
    script = json.load(open(args.script)) if args.script else None
    state = State(script, args.demo, args.health_503, args.file_size)
    handler = type("H", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.daemon_threads = True
    print(f"mock llama-server on {args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
