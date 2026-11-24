import socket
import unittest

from tests.test_tools import ToolTestCase
from xarencode.safety import ASK, BLOCK
from xarencode.server import proc_start_time
from xarencode.tools import devserver


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DevServerTests(ToolTestCase):
    def tearDown(self):
        devserver.stop_all()
        super().tearDown()

    def setUp(self):
        super().setUp()
        self.port = free_port()
        (self.work / "index.html").write_text("hi")
        self.tool = self.reg.get("dev_server")

    def run_tool(self, **args):
        plan = self.tool.plan(args, self.tctx)
        return plan, self.tool.run(args, self.tctx, plan)

    def test_free_command_asks_then_starts_and_stops(self):
        cmd = f"python3 -m http.server {self.port} --bind 127.0.0.1"
        plan = self.tool.plan({"action": "start", "command": cmd}, self.tctx)
        self.assertEqual(plan.verdict, ASK)
        res = self.tool.run({"action": "start", "command": cmd}, self.tctx, plan)
        self.assertTrue(res.ok, res.content)
        self.assertEqual(res.detail["event"], "server_up")
        self.assertTrue(devserver.port_in_use(self.port))
        _, st = self.run_tool(action="status")
        self.assertIn("running", st.content)
        # starting again on the same port fails cleanly
        _, again = self.run_tool(action="start", command=cmd, name="other")
        self.assertIn("already in use", again.content)
        _, logs = self.run_tool(action="logs", name=res.detail["name"])
        self.assertTrue(logs.ok)
        _, stop = self.run_tool(action="stop", name=res.detail["name"])
        self.assertTrue(stop.ok, stop.content)
        self.assertFalse(devserver.port_in_use(self.port))

    def test_approval_key_skips_exec_process_name(self):
        plan = self.tool.plan({"action": "start", "command": "exec -a zzz-title npm run dev"}, self.tctx)
        self.assertEqual(plan.verdict, ASK)
        self.assertNotIn("zzz-title", plan.approval_key)
        self.assertEqual(plan.approval_key, "dev_server npm run dev")
        other = self.tool.plan({"action": "start", "command": "exec -a zzz-title sleep 60"}, self.tctx)
        self.assertIsNone(other.approval_key)  # not on the short list of commands that may be remembered

    def test_stop_leaves_unrelated_process_alone(self):
        import subprocess
        other = subprocess.Popen(["sleep", "30"], start_new_session=True)
        self.addCleanup(lambda: (other.kill(), other.wait()))
        devserver._save({"old": {"pid": other.pid, "pgid": other.pid, "cmd": "python3 -m http.server 9", "cwd": "/", "url": ""}})
        self.assertIn("stale entry", devserver.stop_server("old"))
        self.assertIsNone(other.poll())
        self.assertEqual(devserver.load_registry(), {})

    def _renamed_sleeper(self):
        """A child in its own session whose command line no longer resembles the registered command (like npm)."""
        import subprocess
        import threading
        child = subprocess.Popen(["bash", "-c", "exec -a zzz-title sleep 60"], start_new_session=True)
        threading.Thread(target=child.wait, daemon=True).start()  # reap it like init would for an orphan
        self.addCleanup(lambda: child.poll() is None and child.kill())
        return child

    def test_stop_from_a_fresh_session_uses_start_time(self):
        import time
        child = self._renamed_sleeper()
        time.sleep(0.2)
        entry = {"pid": child.pid, "pgid": child.pid, "start": proc_start_time(child.pid), "cmd": "npm run dev --port 5173",
                 "cwd": "/", "url": ""}
        self.assertNotIn("zzz-title", entry["cmd"])
        devserver._save({"web": entry})
        devserver._procs.pop(child.pid, None)  # a new process knows nothing about the Popen object
        reg = devserver.load_registry()
        self.assertTrue(devserver.is_ours(reg["web"]))
        self.assertEqual(devserver.stop_server("web"), "stopped 'web'")
        deadline = time.monotonic() + 3
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(child.poll())
        self.assertEqual(devserver.load_registry(), {})

    def test_reused_pid_with_other_start_time_is_not_signalled(self):
        child = self._renamed_sleeper()
        devserver._save({"web": {"pid": child.pid, "pgid": child.pid, "start": proc_start_time(child.pid) + 1000,
                                 "cmd": "sleep 60", "cwd": "/", "url": ""}})
        self.assertIn("stale entry", devserver.stop_server("web"))
        self.assertIsNone(child.poll())
        self.assertEqual(devserver.load_registry(), {})

    def test_start_records_start_time(self):
        _, res = self.run_tool(action="start", command=f"python3 -m http.server {self.port} --bind 127.0.0.1")
        self.assertTrue(res.ok, res.content)
        e = next(iter(devserver.load_registry().values()))
        self.assertEqual(e["start"], proc_start_time(e["pid"]))

    def test_preset_runs_without_asking(self):
        self.cfg["dev"]["site"] = {"dir": str(self.work), "cmd": f"python3 -m http.server {self.port} --bind 127.0.0.1",
                                   "url": f"http://localhost:{self.port}"}
        plan = self.tool.plan({"action": "start", "name": "site"}, self.tctx)
        self.assertEqual(plan.verdict.name, "ALLOW")
        res = self.tool.run({"action": "start", "name": "site"}, self.tctx, plan)
        self.assertTrue(res.ok, res.content)
        self.assertIn(str(self.port), res.content)

    def test_preset_with_an_elevator_is_never_allow(self):
        for word in ("sudo", "doas", "pkexec", "su -c", "run0"):
            self.cfg["dev"]["priv"] = {"dir": str(self.work), "cmd": f"{word} python3 -m http.server {self.port}"}
            plan = self.tool.plan({"action": "start", "name": "priv"}, self.tctx)
            self.assertEqual(plan.verdict, BLOCK, word)
            self.assertIn("elevated", plan.reason)
        self.cfg["safety"]["sudo"] = "never"
        self.assertEqual(self.tool.plan({"action": "start", "name": "priv"}, self.tctx).verdict, BLOCK)
        self.assertFalse(devserver.port_in_use(self.port))

    def test_blocked_command(self):
        plan = self.tool.plan({"action": "start", "command": "rm -rf x"}, self.tctx)
        self.assertEqual(plan.verdict, BLOCK)

    def test_stop_only_registered(self):
        _, res = self.run_tool(action="stop", name="nothing")
        self.assertFalse(res.ok)
        _, res = self.run_tool(action="stop")
        self.assertIn("needs a name", res.content)

    def test_failing_command_reports_log(self):
        _, res = self.run_tool(action="start", command="echo boom; exit 2", name="bad")
        self.assertFalse(res.ok)
        self.assertIn("boom", res.content)

    def test_start_needs_target(self):
        _, res = self.run_tool(action="start")
        self.assertIn("preset name or a command", res.content)


if __name__ == "__main__":
    unittest.main()
