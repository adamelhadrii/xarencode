import datetime as dt
import json
import os
import unittest
from pathlib import Path

from tests.helpers import IsolatedTestCase
from xarencode import persona, sessions
from xarencode.config import default_config


class SessionTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.cfg = default_config()

    def test_roundtrip(self):
        s = sessions.Session.new(self.cfg, self.work)
        self.assertFalse(s.path.exists())  # lazy: nothing written until the first message
        s.append({"role": "user", "content": "hello there"})
        s.append({"role": "tool", "tool_call_id": "c1", "content": "x"}, {"summary": "ok", "diff": None, "tool": "read_file"})
        lines = s.path.read_text().splitlines()
        self.assertEqual(json.loads(lines[0])["t"], "meta")
        self.assertEqual(json.loads(lines[0])["title"], "hello there")
        loaded = sessions.Session.load(s.path)
        self.assertEqual(loaded.messages()[0]["content"], "hello there")
        self.assertEqual(loaded.entries[1][1]["summary"], "ok")

    def test_prune_keeps_newest(self):
        for i in range(5):
            s = sessions.Session.new(self.cfg, self.work)
            s.meta["created"] = f"2026-01-0{i + 1}T00:00:00"
            s.append({"role": "user", "content": f"m{i}"})
        self.assertEqual(sessions.prune(2), 3)
        left = sessions.list_sessions()
        self.assertEqual([x.title for x in left], ["m4", "m3"])
        self.assertEqual(sessions.find_session("2").title, "m3")
        self.assertEqual(sessions.find_session(None).title, "m4")
        self.assertIsNone(sessions.find_session("zzz"))

    def test_rewrite(self):
        s = sessions.Session.new(self.cfg, self.work)
        s.append({"role": "user", "content": "a"})
        s.rewrite([{"role": "user", "content": "summary"}])
        self.assertEqual(len(sessions.Session.load(s.path).entries), 1)

    def test_redaction(self):
        text = (f"path {Path.home()}/proj key sk-abcdefghijklmnopqrstuv "
                "token=abc123supersecret AKIAABCDEFGHIJKLMNOP Bearer abcdefghijklmnop1234 ghp_abcdefghijklmnopqrstuvwxyz")
        out = sessions.redact(text)
        self.assertIn("~/proj", out)
        self.assertNotIn(str(Path.home()), out)
        for secret in ("sk-abcdef", "abc123supersecret", "AKIAABCDEFGHIJKLMNOP", "abcdefghijklmnop1234", "ghp_abc"):
            self.assertNotIn(secret, out)
        self.assertEqual(sessions.redact("plain text"), "plain text")

    def test_redaction_env_json_and_url(self):
        text = ("GITHUB_TOKEN=abcdef123456\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG\nDB_PASSWORD=hunter2\n"
                '{"password": "p4ss word", "api_key": "zzz999"}\n'
                "url postgres://user:pw0rd@db.example:5432/x and https://github.com/a/b")
        out = sessions.redact(text)
        for secret in ("abcdef123456", "wJalrXUtnFEMI", "hunter2", "p4ss", "zzz999", "pw0rd"):
            self.assertNotIn(secret, out)
        self.assertIn("GITHUB_TOKEN=[redacted]", out)
        self.assertIn("postgres://[redacted]@db.example:5432/x", out)
        self.assertIn("https://github.com/a/b", out)
        self.assertEqual(sessions.redact(out), out)

    def test_redaction_leaves_prose_and_code_alone(self):
        for text in ["author: Adam El", "The monkey: a primate", "max_tokens=1024", "tokens = _tokenize(cmd)",
                     "key=lambda x: x", "keyboard=us", "tokenizer=bpe", "keys: 3", "export TOKEN=$TOKEN"]:
            self.assertEqual(sessions.redact(text), text)

    def test_redaction_authorization_and_name_forms(self):
        cases = {
            "Authorization: Basic dXNlcjpwYXNz": "dXNlcjpwYXNz",
            '-H "Authorization: Bearer abc.def.ghi"': "abc.def.ghi",
            '{"authorization": "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="}': "QWxhZGRpbjpvcGVuIHNlc2FtZQ",
            "API-KEY: xyz789": "xyz789", "access_token=t0k3n": "t0k3n", "accessToken: 'camel1'": "camel1",
            "passwd=pw123": "pw123", "credentials: cr3d": "cr3d", "SECRET=top1": "top1", "DB_PASSWORD=hunter2": "hunter2",
        }
        for text, secret in cases.items():
            out = sessions.redact(text)
            self.assertNotIn(secret, out, text)
            self.assertIn("[redacted]", out, text)
            self.assertEqual(sessions.redact(out), out)

    def test_export_markdown(self):
        s = sessions.Session.new(self.cfg, self.work)
        s.append({"role": "user", "content": f"look at {Path.home()}/a with password=hunter2"})
        s.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                  "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})}}]})
        s.append({"role": "tool", "tool_call_id": "c", "content": "print(1)"}, {"summary": "Read 1 lines"})
        s.append({"role": "assistant", "content": "It prints 1."})
        out_dir = self.tmp / "share"
        p = sessions.export(s, "md", out_dir, self.cfg)
        text = p.read_text()
        self.assertEqual(p.parent, out_dir)
        self.assertIn("**You:**", text)
        self.assertIn("read_file(a.py)", text)
        self.assertIn("It prints 1.", text)
        self.assertNotIn("hunter2", text)
        self.assertNotIn(str(Path.home()), text)

    def test_export_empty_and_bad_format(self):
        s = sessions.Session.new(self.cfg, self.work)
        with self.assertRaises(sessions.ExportError):
            sessions.export(s, "md", self.tmp, self.cfg)
        s.append({"role": "user", "content": "x"})
        with self.assertRaises(sessions.ExportError):
            sessions.export(s, "pdf", self.tmp, self.cfg)


class PersonaTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.cfg = default_config()

    def test_system_prompt_small_and_vibes(self):
        for vibe, line in persona.VIBES.items():
            self.cfg["persona"]["vibe"] = vibe
            p = persona.system_prompt(self.cfg, "/work/x")
            self.assertIn(line, p)
            self.assertLess(len(p) / 3, 250)
            self.assertLess(len(line) / 3, 40)
        self.cfg["persona"]["user_name"] = "Sam"
        self.assertIn("The user is Sam", persona.system_prompt(self.cfg, "/w"))

    def test_greeting_by_time(self):
        self.cfg["persona"]["user_name"] = "Sam"
        at = lambda h: persona.greeting(self.cfg, dt.datetime(2026, 1, 1, h))  # noqa: E731
        self.assertIn("Good morning, Sam", at(8))
        self.assertIn("Good afternoon", at(14))
        self.assertIn("Good evening", at(20))
        self.assertIn("midnight", at(2))
        self.cfg["persona"]["emoji"] = False
        self.assertNotIn("👋", at(8))

    def test_celebrate_and_verbs(self):
        for ev in ("tests_passed", "server_up", "edit_applied"):
            self.assertTrue(persona.celebrate(ev, self.cfg))
        self.assertEqual(persona.celebrate("nothing", self.cfg), "")
        self.cfg["persona"]["emoji"] = False
        self.assertTrue(persona.celebrate("server_up", self.cfg).isascii())
        self.assertIn(persona.spinner_verb(), persona.VERBS)

    def test_farewell(self):
        st = persona.SessionStats(turns=3, tool_calls=5, edits=2, tokens=900)
        text = persona.farewell(st, self.cfg)
        self.assertIn("3 turns", text)
        self.assertIn("2 edits", text)
        self.assertIn("Xaren", text)


if __name__ == "__main__":
    unittest.main()
