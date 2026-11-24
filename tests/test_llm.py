import threading
import unittest

from tests.mock_llama_server import MockServer
from xarencode.llm import (ContextOverflow, Done, Interrupted, LlamaClient, ReasoningDelta, ServerError, ServerUnavailable,
                           TextDelta, ToolCallDelta)

MSGS = [{"role": "user", "content": "hi"}]
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


class LlmTests(unittest.TestCase):
    def serve(self, script=None, **kw):
        s = MockServer(script, **kw).start()
        self.addCleanup(s.stop)
        return s, LlamaClient(s.url)

    def test_stream_text_and_usage(self):
        s, c = self.serve([[{"reasoning": "hmm"}, {"text": "Hello world"}]])
        ev = list(c.stream_chat(MSGS, [], max_tokens=50, sampling=SAMPLING))
        self.assertEqual("".join(e.text for e in ev if isinstance(e, TextDelta)), "Hello world")
        self.assertTrue(any(isinstance(e, ReasoningDelta) for e in ev))
        dones = [e for e in ev if isinstance(e, Done)]
        self.assertEqual(dones[0].finish_reason, "stop")
        self.assertEqual(dones[-1].timings["predicted_per_second"], 12.5)
        self.assertGreater(dones[-1].usage["prompt_tokens"], 0)

    def test_request_body(self):
        s, c = self.serve()
        list(c.stream_chat(MSGS, [{"type": "function"}], max_tokens=33, sampling={**SAMPLING, "thinking": False}))
        b = s.requests[0]
        self.assertTrue(b["stream"])
        self.assertEqual((b["max_tokens"], b["tool_choice"], b["parallel_tool_calls"], b["cache_prompt"], b["min_p"]), (33, "auto", False, True, 0))
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(b["top_k"], 20)

    def test_tool_call_fragments(self):
        s, c = self.serve([[{"tool_call": {"name": "list_dir", "arguments": '{"path": "."}'}}]])
        ev = list(c.stream_chat(MSGS, [], max_tokens=50, sampling=SAMPLING))
        tcs = [e for e in ev if isinstance(e, ToolCallDelta)]
        self.assertEqual(tcs[0].name, "list_dir")
        self.assertEqual("".join(t.arguments for t in tcs), '{"path": "."}')
        self.assertEqual([e for e in ev if isinstance(e, Done)][0].finish_reason, "tool_calls")

    def test_context_overflow(self):
        s, c = self.serve([[{"error": {"status": 400, "type": "exceed_context_size_error", "message": "too big"}}]])
        with self.assertRaises(ContextOverflow):
            list(c.stream_chat(MSGS, [], max_tokens=5, sampling=SAMPLING))

    def test_overflow_by_message(self):
        s, c = self.serve([[{"error": {"status": 400, "type": "invalid_request_error", "message": "prompt exceeds the context window"}}]])
        with self.assertRaises(ContextOverflow):
            list(c.stream_chat(MSGS, [], max_tokens=5, sampling=SAMPLING))

    def test_server_error(self):
        s, c = self.serve([[{"error": {"status": 500, "message": "kaboom"}}]])
        with self.assertRaises(ServerError) as cm:
            list(c.stream_chat(MSGS, [], max_tokens=5, sampling=SAMPLING))
        self.assertIn("kaboom", str(cm.exception))

    def test_unavailable(self):
        c = LlamaClient("http://127.0.0.1:1")
        with self.assertRaises(ServerUnavailable):
            list(c.stream_chat(MSGS, [], max_tokens=5, sampling=SAMPLING))
        self.assertEqual(c.health(), "down")

    def test_health_states(self):
        s, c = self.serve(health_503=1)
        self.assertEqual(c.health(), "loading")
        self.assertEqual(c.health(), "ok")

    def test_complete(self):
        s, c = self.serve([[{"text": "a summary"}]])
        self.assertEqual(c.complete(MSGS, max_tokens=20, sampling=SAMPLING), "a summary")
        self.assertFalse(s.requests[0]["stream"])

    def test_cancel_mid_stream(self):
        s, c = self.serve([[{"text": "abcdefgh"}, {"delay": 5}, {"text": "never"}]])
        seen = []
        threading.Timer(0.4, c.cancel).start()
        with self.assertRaises(Interrupted):
            for e in c.stream_chat(MSGS, [], max_tokens=50, sampling=SAMPLING):
                seen.append(e)
        self.assertTrue(seen)
        self.assertNotIn("never", "".join(e.text for e in seen if isinstance(e, TextDelta)))


if __name__ == "__main__":
    unittest.main()
