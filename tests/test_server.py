import hashlib
import os
import socket
import sys
import time
import unittest
from pathlib import Path

from tests.helpers import IsolatedTestCase
from tests.mock_llama_server import MockServer, file_bytes
from xarencode import server
from xarencode.config import default_config, resolve_model
from xarencode.server import LlamaServer, PullError, ServerStartError, doctor, pull_model

MOCK = str(Path(__file__).parent / "mock_llama_server.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LifecycleTests(IsolatedTestCase):
    def cfg_for_mock(self):
        cfg = default_config()
        cfg["server"].update(binary=MOCK, port=free_port(), args=["--jinja", "-np", "1"])
        cfg["model"]["gguf"] = str(self.tmp / "fake.gguf")
        (self.tmp / "fake.gguf").write_bytes(b"x")
        return cfg

    def test_command_line(self):
        cfg = default_config()
        cmd = LlamaServer(cfg).command()
        self.assertEqual(cmd[0], "llama-server")
        self.assertEqual(cmd[cmd.index("-c") + 1], "4096")
        self.assertEqual(cmd[cmd.index("-t") + 1], "4")
        self.assertEqual(cmd[cmd.index("--port") + 1], "8091")
        self.assertIn("--jinja", cmd)

    def test_start_stop_with_mock_binary(self):
        cfg = self.cfg_for_mock()
        srv = LlamaServer(cfg)
        seen = []
        self.assertTrue(srv.ensure(seen.append, timeout=20))
        self.assertEqual(srv.status(), "ok")
        self.assertTrue(srv.pid_path.exists())
        self.assertFalse(srv.ensure())  # already up
        self.assertEqual(srv.stop(), "llama-server stopped")
        self.assertEqual(srv.status(), "down")
        self.assertFalse(srv.pid_path.exists())

    def test_missing_binary_and_model(self):
        cfg = default_config()
        cfg["server"]["binary"] = "definitely-not-installed"
        with self.assertRaises(ServerStartError) as cm:
            LlamaServer(cfg).start()
        self.assertIn("pacman -S llama-cpp", str(cm.exception))
        cfg = self.cfg_for_mock()
        (self.tmp / "fake.gguf").unlink()
        with self.assertRaises(ServerStartError) as cm:
            LlamaServer(cfg).start()
        self.assertIn("xarencode pull", str(cm.exception))

    def test_manage_false_raises_when_down(self):
        cfg = self.cfg_for_mock()
        cfg["server"]["manage"] = False
        with self.assertRaises(ServerStartError):
            LlamaServer(cfg).ensure()

    def test_exiting_server_reports_log(self):
        cfg = self.cfg_for_mock()
        cfg["server"]["binary"] = "/bin/false"
        with self.assertRaises(ServerStartError) as cm:
            LlamaServer(cfg).ensure(timeout=10)
        self.assertIn("exited", str(cm.exception))

    def test_ensure_timeout_stops_the_server_we_started(self):
        cfg = self.cfg_for_mock()
        fake = self.tmp / "never-ready"
        fake.write_text("#!/bin/sh\nexec sleep 60\n")
        fake.chmod(0o755)
        cfg["server"]["binary"] = str(fake)
        srv = LlamaServer(cfg)
        with self.assertRaises(ServerStartError):
            srv.ensure(timeout=1)
        self.assertIsNotNone(srv.proc.poll())  # the child was stopped, not orphaned
        self.assertFalse(srv.pid_path.exists())
        self.assertFalse(srv.started_by_us)

    def test_stop_leaves_unrelated_process_alone(self):
        import subprocess
        other = subprocess.Popen(["sleep", "30"])
        self.addCleanup(lambda: (other.kill(), other.wait()))
        srv = LlamaServer(self.cfg_for_mock())
        srv.pid_path.parent.mkdir(parents=True, exist_ok=True)
        srv.pid_path.write_text(str(other.pid))  # stale pid file whose pid now belongs to something else
        self.assertIn("stale pid file removed", srv.stop())
        self.assertIsNone(other.poll())
        self.assertFalse(srv.pid_path.exists())

    def _renamed_sleeper(self):
        import subprocess
        import threading
        child = subprocess.Popen(["bash", "-c", "exec -a zzz-title sleep 60"], start_new_session=True)
        threading.Thread(target=child.wait, daemon=True).start()
        self.addCleanup(lambda: child.poll() is None and child.kill())
        return child

    def test_stop_uses_start_time_even_if_the_title_changed(self):
        from xarencode.server import proc_start_time
        child = self._renamed_sleeper()
        time.sleep(0.2)
        srv = LlamaServer(self.cfg_for_mock())
        srv.pid_path.parent.mkdir(parents=True, exist_ok=True)
        srv.pid_path.write_text(f"{child.pid} {proc_start_time(child.pid)}")
        self.assertEqual(srv.stop(), "llama-server stopped")
        deadline = time.monotonic() + 3
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(child.poll())

    def test_stop_ignores_reused_pid_with_other_start_time(self):
        from xarencode.server import proc_start_time
        child = self._renamed_sleeper()
        time.sleep(0.2)
        srv = LlamaServer(self.cfg_for_mock())
        srv.pid_path.parent.mkdir(parents=True, exist_ok=True)
        srv.pid_path.write_text(f"{child.pid} {proc_start_time(child.pid) + 1000}")
        self.assertIn("stale pid file removed", srv.stop())
        self.assertIsNone(child.poll())

    def test_doctor_reports_missing(self):
        cfg = default_config()
        cfg["server"]["binary"] = "nope-llama"
        res = doctor(cfg)
        self.assertEqual(res[0][0], "fail")
        self.assertIn("pacman", res[0][1])


class PullTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.mock = MockServer(file_size=300_000).start()
        self.addCleanup(self.mock.stop)
        self.cfg = default_config()
        self.cfg["model"].update(gguf="o/r/fake.gguf", hf_base=self.mock.url)
        self.log = []
        self.devnull = open(os.devnull, "w")
        self._err, sys.stderr = sys.stderr, self.devnull

    def tearDown(self):
        sys.stderr = self._err
        self.devnull.close()
        super().tearDown()

    def test_pull_verifies_and_renames(self):
        dest = pull_model(self.cfg, self.log.append)
        self.assertEqual(dest, resolve_model(self.cfg))
        self.assertEqual(dest.read_bytes(), file_bytes(300_000))
        self.assertFalse(dest.with_name("fake.gguf.part").exists())

    def test_resume_with_range(self):
        dest = resolve_model(self.cfg)
        dest.parent.mkdir(parents=True)
        part = dest.with_name("fake.gguf.part")
        part.write_bytes(file_bytes(300_000)[:123_456])
        pull_model(self.cfg, self.log.append)
        self.assertEqual(dest.read_bytes(), file_bytes(300_000))
        self.assertTrue(any("resuming" in l for l in self.log))

    def test_corrupt_partial_fails_checksum(self):
        dest = resolve_model(self.cfg)
        dest.parent.mkdir(parents=True)
        part = dest.with_name("fake.gguf.part")
        part.write_bytes(b"\xff" * 1000)
        with self.assertRaises(PullError) as cm:
            pull_model(self.cfg, self.log.append)
        self.assertIn("sha256", str(cm.exception))
        self.assertFalse(part.exists())

    def test_already_present(self):
        pull_model(self.cfg, self.log.append)
        pull_model(self.cfg, self.log.append)
        self.assertIn("already downloaded", self.log)

    def test_storage_budget_asks_before_deleting(self):
        d = resolve_model(self.cfg).parent
        d.mkdir(parents=True)
        old = d / "old.gguf"
        with open(old, "wb") as f:
            f.truncate(30_000_000)  # sparse; pushes the total over the budget below
        self.cfg["model"]["storage_budget_gb"] = 0.11  # 118 MB; the 0.1 GiB margin alone is 107 MB
        with self.assertRaises(PullError):
            pull_model(self.cfg, self.log.append, confirm=lambda q: False)
        self.assertTrue(old.exists())
        pull_model(self.cfg, self.log.append, confirm=lambda q: True)
        self.assertFalse(old.exists())
        self.assertTrue(resolve_model(self.cfg).exists())

    def test_http_error(self):
        self.cfg["model"]["gguf"] = "o/r/missing.txt"
        with self.assertRaises(PullError):
            pull_model(self.cfg, self.log.append)


if __name__ == "__main__":
    unittest.main()
