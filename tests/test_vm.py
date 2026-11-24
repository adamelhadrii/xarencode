import os
import socket
import threading
import unittest

from tests.test_tools import ToolTestCase
from xarencode.safety import ASK
from xarencode.tools import vm


class VmTests(ToolTestCase):
    def machine(self, **over):
        disk = self.tmp / "d.qcow2"
        disk.write_bytes(b"x")
        m = {"disk": str(disk), "memory": "2G", "cpus": 2, "uefi": True, "display": "none", "ssh_port": 2222}
        m.update(over)
        self.cfg["vm"]["machines"]["t"] = m
        self.reg = __import__("xarencode.tools", fromlist=["build_registry"]).build_registry(self.cfg)

    def test_no_machines_drops_the_tool(self):
        from xarencode.tools import build_registry
        self.assertIsNone(build_registry(self.cfg | {"vm": {"machines": {}}}).get("vm"))

    def test_no_machines_says_so(self):
        tool = vm.tool(self.cfg | {"vm": {"machines": {}}})
        self.assertIn("None are configured", tool.description)
        plan = tool.plan({"action": "start", "name": "x"}, self.tctx)
        res = tool.run({"action": "start", "name": "x"}, self.tctx, plan)
        self.assertIn("no virtual machines are configured", res.content)

    def test_command_line(self):
        self.machine(iso=str(self.tmp / "i.iso"))
        (self.tmp / "i.iso").write_bytes(b"i")
        cmd = vm.QemuBackend().build_command("t", vm.spec_for(self.cfg, "t"))
        j = " ".join(cmd)
        for part in ("-machine q35", "-smp 2", "-m 2048M", "if=virtio,format=qcow2", "-cdrom", "-boot d", "OVMF_CODE.4m.fd",
                     "hostfwd=tcp:127.0.0.1:2222-:22", "-display none", "-qmp unix:", "-pidfile"):
            self.assertIn(part, j)
        self.assertEqual(cmd[0], "qemu-system-x86_64")

    def test_disk_must_be_regular_file(self):
        self.machine(disk="/dev/sda")
        with self.assertRaises(ValueError):
            vm.QemuBackend().build_command("t", vm.spec_for(self.cfg, "t"))
        self.machine(disk=str(self.tmp / "missing.img"))
        with self.assertRaises(ValueError):
            vm.QemuBackend().build_command("t", vm.spec_for(self.cfg, "t"))
        link = self.tmp / "lnk"
        os.symlink("/dev/null", link)
        self.machine(disk=str(link))
        with self.assertRaises(ValueError):
            vm.QemuBackend().build_command("t", vm.spec_for(self.cfg, "t"))

    def test_memory_limits(self):
        self.machine(memory="8G")
        with self.assertRaises(RuntimeError) as cm:
            vm.QemuBackend("3G").start("t", vm.spec_for(self.cfg, "t"))
        self.assertTrue("max_memory" in str(cm.exception) or "qemu" in str(cm.exception))
        self.assertEqual(vm.parse_mem("2G"), 2048)
        self.assertEqual(vm.parse_mem("512M"), 512)

    def test_tool_plan_asks_with_key(self):
        self.machine()
        tool = self.reg.get("vm")
        self.assertEqual(tool.parameters["properties"]["name"]["enum"], ["t"])
        plan = tool.plan({"action": "start", "name": "t"}, self.tctx)
        self.assertEqual((plan.verdict, plan.approval_key), (ASK, "vm:t"))
        self.assertEqual(tool.plan({"action": "status"}, self.tctx).verdict.name, "ALLOW")

    def test_qmp_client_against_fake_socket(self):
        path = str(self.tmp / "q.sock")
        got = []
        srv = socket.socket(socket.AF_UNIX)
        srv.bind(path)
        srv.listen(1)

        def serve():
            c, _ = srv.accept()
            f = c.makefile("rwb")
            f.write(b'{"QMP": {}}\n')
            f.flush()
            for _ in range(2):
                got.append(__import__("json").loads(f.readline())["execute"])
                f.write(b'{"return": {}}\n')
                f.flush()
            c.close()

        t = threading.Thread(target=serve)
        t.start()
        vm.qmp_command(path, "system_powerdown")
        t.join(2)
        srv.close()
        self.assertEqual(got, ["qmp_capabilities", "system_powerdown"])

    def test_stop_without_running(self):
        with self.assertRaises(RuntimeError):
            vm.QemuBackend().stop("ghost", False)


if __name__ == "__main__":
    unittest.main()
