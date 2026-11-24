import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.helpers import IsolatedTestCase
from tests.mock_llama_server import MockServer

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "mock.toml"


class CliTests(IsolatedTestCase):
    def run_cli(self, *args, timeout=60, stdin=subprocess.DEVNULL):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-m", "xarencode", *args], capture_output=True, text=True, env=env,
                              cwd=self.work, timeout=timeout, stdin=stdin)

    def config_for(self, mock):
        text = FIXTURE.read_text().replace("port = 18091", f"port = {mock.port}")
        p = self.tmp / "mock.toml"
        p.write_text(text)
        return str(p)

    def test_yolo_flag_parses_in_both_spellings_without_clashing(self):
        from xarencode.cli import build_parser
        parser = build_parser()
        for argv in (["-yolo"], ["--yolo"], ["-p", "hi", "-yolo"], ["-yolo", "-p", "hi"], ["--yolo", "--workdir", "x"]):
            self.assertTrue(parser.parse_args(argv).yolo, argv)
        for argv in ([], ["-p", "yolo"], ["--workdir", "yolo"], ["doctor"], ["--resume"]):
            self.assertFalse(parser.parse_args(argv).yolo, argv)
        self.assertEqual(parser.parse_args(["-p", "yolo"]).prompt, "yolo")
        for bad in (["-y"], ["-o"], ["-l"], ["-yo"]):  # no clustering into single letters, no abbreviation
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(bad)

    def test_yolo_in_print_mode_approves_asks_but_default_does_not(self):
        script = lambda: MockServer([[{"tool_call": {"name": "run_shell", "arguments": json.dumps({"command": "touch made.txt"})}}],  # noqa: E731
                                     [{"text": "done"}]]).start()
        for flag, expect in (("-yolo", True), ("--yolo", True), (None, False)):
            (self.work / "made.txt").unlink(missing_ok=True)
            mock = script()
            self.addCleanup(mock.stop)
            extra = [flag] if flag else []
            r = self.run_cli("--config", self.config_for(mock), *extra, "-p", "make the file")
            self.assertEqual((self.work / "made.txt").exists(), expect, (flag, r.stderr))
            self.assertEqual("YOLO mode" in r.stderr, expect, flag)

    def test_yolo_in_the_config_file_is_ignored_with_a_warning(self):
        mock = MockServer([[{"tool_call": {"name": "run_shell", "arguments": json.dumps({"command": "touch made.txt"})}}],
                           [{"text": "done"}]]).start()
        self.addCleanup(mock.stop)
        cfg = Path(self.config_for(mock))
        cfg.write_text(cfg.read_text() + "\n[safety]\nyolo = true\n")
        r = self.run_cli("--config", str(cfg), "-p", "make the file")
        self.assertFalse((self.work / "made.txt").exists(), r.stderr)
        self.assertIn("[safety] yolo in the config is ignored", r.stderr)
        self.assertNotIn("YOLO mode", r.stderr)

    def test_default_config_template_has_no_yolo_setting(self):
        from xarencode.config import DEFAULTS, TEMPLATE
        self.assertNotIn("yolo", DEFAULTS["safety"])
        self.assertNotIn("yolo =", TEMPLATE)

    def test_version(self):
        r = self.run_cli("--version")
        self.assertEqual(r.stdout.strip(), "xarencode 0.1.0")

    def test_print_mode_with_tool_call(self):
        (self.work / "alpha.txt").write_text("a")
        (self.work / "beta.txt").write_text("b")
        mock = MockServer([[{"tool_call": {"name": "list_dir", "arguments": json.dumps({"path": "."})}}],
                           [{"text": "There are 2 files here."}]]).start()
        self.addCleanup(mock.stop)
        r = self.run_cli("--config", self.config_for(mock), "-p", "list the files here and count them")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("There are 2 files here.", r.stdout)
        self.assertIn("list_dir", r.stderr)
        tool_msg = [m for m in mock.requests[1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("alpha.txt", tool_msg["content"])
        self.assertTrue(list((self.tmp / "data" / "xarencode" / "sessions").glob("*.jsonl")))

    def test_print_mode_non_interactive_denies_writes(self):
        mock = MockServer([[{"tool_call": {"name": "write_file", "arguments": json.dumps({"path": "x.txt", "content": "1"})}}],
                           [{"text": "could not write"}]]).start()
        self.addCleanup(mock.stop)
        r = self.run_cli("--config", self.config_for(mock), "-p", "write x")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.work / "x.txt").exists())

    def test_print_mode_server_down(self):
        p = self.tmp / "down.toml"
        p.write_text("[server]\nmanage = false\nport = 1\n")
        r = self.run_cli("--config", str(p), "-p", "hi")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no llama-server", r.stderr)

    def test_doctor_without_llama_server_exits_1_with_hint(self):
        p = self.tmp / "d.toml"
        p.write_text('[server]\nbinary = "definitely-not-installed"\n')
        r = self.run_cli("--config", str(p), "doctor")
        self.assertEqual(r.returncode, 1)
        self.assertIn("llama-server not found", r.stdout)
        self.assertIn("pacman -S llama-cpp", r.stdout)

    def test_server_status_and_log(self):
        mock = MockServer().start()
        self.addCleanup(mock.stop)
        cfg = self.config_for(mock)
        self.assertEqual(self.run_cli("--config", cfg, "server", "status").stdout.strip(), "ok")
        self.assertEqual(self.run_cli("--config", cfg, "server", "log").returncode, 0)

    def test_refuses_root_and_missing_dir(self):
        p = self.tmp / "c.toml"
        p.write_text("")
        self.assertEqual(self.run_cli("--config", str(p), "--workdir", "/", "-p", "x").returncode, 2)
        self.assertEqual(self.run_cli("--config", str(p), "--workdir", str(self.tmp / "nope"), "-p", "x").returncode, 2)

    def test_first_run_creates_template_in_xdg(self):
        self.run_cli("doctor")
        self.assertTrue((self.tmp / "cfg" / "xarencode" / "config.toml").exists())
        self.assertFalse((self.home / ".config").exists())

    def test_bad_config(self):
        p = self.tmp / "bad.toml"
        p.write_text("[server\n")
        r = self.run_cli("--config", str(p), "doctor")
        self.assertEqual(r.returncode, 2)
        self.assertIn("bad config", r.stderr)


if __name__ == "__main__":
    unittest.main()
