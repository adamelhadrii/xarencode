"""vm tool: start/stop/status of configured virtual machines behind a backend interface (qemu)."""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import state_dir
from ..safety import ALLOW, ASK
from . import Plan, Tool, ToolContext, ToolResult, error, schema

OVMF_CODE = "/usr/share/edk2/x64/OVMF_CODE.4m.fd"
OVMF_VARS = "/usr/share/edk2/x64/OVMF_VARS.4m.fd"


@dataclass
class VMSpec:
    disk: str
    iso: str = ""
    memory: str = "2G"
    cpus: int = 2
    uefi: bool = True
    display: str = "gtk"
    ssh_port: int = 0


class VMBackend(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...
    def start(self, name: str, spec: VMSpec) -> str: ...
    def stop(self, name: str, force: bool) -> str: ...
    def status(self) -> list[dict]: ...


def parse_mem(s: str) -> int:
    """'2G' / '512M' -> MiB."""
    m = re.fullmatch(r"\s*(\d+)\s*([MGmg]?)[iI]?[bB]?\s*", str(s))
    if not m:
        raise ValueError(f"bad memory size {s!r}")
    n = int(m.group(1))
    return n * 1024 if m.group(2).lower() == "g" else n


def mem_available_mib() -> int:
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) // 1024
    except OSError:
        pass
    return 0


def vm_dir() -> Path:
    d = state_dir() / "vm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _regular_file(p: str) -> Path:
    real = Path(os.path.realpath(os.path.expanduser(p)))
    if real == Path("/dev") or Path("/dev") in real.parents:
        raise ValueError(f"{p}: device paths are not allowed")
    try:
        ok = stat.S_ISREG(real.stat().st_mode)
    except OSError:
        ok = False
    if not ok:
        raise ValueError(f"{p}: not an existing regular file")
    return real


def qmp_command(sock_path: str, command: str, timeout: float = 5.0) -> None:
    """Minimal QMP client: handshake, then one command."""
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        f = s.makefile("rwb")
        f.readline()  # greeting
        for cmd in ("qmp_capabilities", command):
            f.write(json.dumps({"execute": cmd}).encode() + b"\n")
            f.flush()
            while True:
                reply = json.loads(f.readline() or b"{}")
                if "return" in reply or "error" in reply:
                    break
            if "error" in reply:
                raise OSError(reply["error"].get("desc", "QMP error"))


class QemuBackend:
    name = "qemu"

    def __init__(self, max_memory: str = "3G"):
        self.max_memory = max_memory

    def available(self) -> tuple[bool, str]:
        if shutil.which("qemu-system-x86_64") is None:
            return False, "qemu-system-x86_64 not found (pacman -S qemu-system-x86)"
        return True, ""

    def build_command(self, name: str, spec: VMSpec) -> list[str]:
        d = vm_dir()
        disk = _regular_file(spec.disk)
        fmt = "qcow2" if disk.suffix == ".qcow2" else "raw"
        cmd = ["qemu-system-x86_64", "-name", name, "-machine", "q35"]
        if os.access("/dev/kvm", os.R_OK | os.W_OK):
            cmd += ["-enable-kvm", "-cpu", "host"]
        cmd += ["-smp", str(spec.cpus), "-m", f"{parse_mem(spec.memory)}M",
                "-drive", f"file={disk},if=virtio,format={fmt}"]
        if spec.iso:
            cmd += ["-cdrom", str(_regular_file(spec.iso)), "-boot", "d"]
        if spec.uefi:
            vars_copy = d / f"{name}_VARS.fd"
            cmd += ["-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
                    "-drive", f"if=pflash,format=raw,file={vars_copy}"]
        nic = "user,model=virtio-net-pci" + (f",hostfwd=tcp:127.0.0.1:{spec.ssh_port}-:22" if spec.ssh_port else "")
        cmd += ["-nic", nic, "-display", spec.display,
                "-qmp", f"unix:{d / (name + '.qmp')},server=on,wait=off", "-pidfile", str(d / f"{name}.pid")]
        return cmd

    def start(self, name: str, spec: VMSpec) -> str:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        need = parse_mem(spec.memory)
        if need > parse_mem(self.max_memory):
            raise RuntimeError(f"{spec.memory} exceeds vm.max_memory ({self.max_memory})")
        avail = mem_available_mib()
        if avail < need + 512:
            raise RuntimeError(f"not enough free RAM: {avail} MiB available, need {need + 512} MiB")
        if name in {s["name"] for s in self.status()}:
            raise RuntimeError(f"'{name}' is already running")
        cmd = self.build_command(name, spec)
        if spec.uefi:
            vars_copy = vm_dir() / f"{name}_VARS.fd"
            if not vars_copy.exists():
                shutil.copyfile(OVMF_VARS, vars_copy)
        log = open(vm_dir() / f"{name}.log", "wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        try:
            proc.wait(1.5)
        except subprocess.TimeoutExpired:
            return f"started '{name}' (pid {proc.pid}, {spec.memory}, {spec.cpus} cpus)"
        tail = (vm_dir() / f"{name}.log").read_text("utf-8", "replace")[-400:]
        raise RuntimeError(f"qemu exited immediately: {tail}")

    def stop(self, name: str, force: bool) -> str:
        sock = vm_dir() / f"{name}.qmp"
        if not sock.exists():
            raise RuntimeError(f"'{name}' is not running")
        try:
            qmp_command(str(sock), "quit" if force else "system_powerdown")
        except OSError as e:
            raise RuntimeError(f"could not reach '{name}' over QMP: {e}") from e
        return f"{'killed' if force else 'sent a power-down request to'} '{name}'"

    def status(self) -> list[dict]:
        out = []
        for pidfile in vm_dir().glob("*.pid"):
            try:
                pid = int(pidfile.read_text().strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                continue
            out.append({"name": pidfile.stem, "pid": pid})
        return out


BACKENDS = {"qemu": QemuBackend}


def get_backend(cfg: dict) -> VMBackend:
    kind = cfg["vm"].get("backend", "qemu")
    if kind not in BACKENDS:
        raise RuntimeError(f"unknown vm backend '{kind}' (available: {', '.join(BACKENDS)})")
    return BACKENDS[kind](cfg["vm"].get("max_memory", "3G"))


def spec_for(cfg: dict, name: str) -> VMSpec:
    m = cfg["vm"]["machines"][name]
    return VMSpec(m["disk"], m.get("iso", ""), m.get("memory", "2G"), int(m.get("cpus", 2)), bool(m.get("uefi", True)),
                  m.get("display", "gtk"), int(m.get("ssh_port", 0)))


NONE_MSG = "no virtual machines are configured; add a [vm.machines.NAME] section with disk = \"...\" to the XarenCode config"


def _plan(args: dict, tctx: ToolContext) -> Plan:
    machines = tctx.cfg["vm"].get("machines", {})
    action = args.get("action")
    if not machines:
        return Plan(ALLOW, "no machines", payload={"error": NONE_MSG})
    if action == "status":
        return Plan(ALLOW, "status")
    name = args.get("name")
    if name not in machines:
        return Plan(ALLOW, "unknown vm", payload={"error": f"unknown VM '{name}'; configured: {', '.join(machines)}"})
    what = f"{action} VM '{name}'" + (" (forced)" if args.get("force") else "")
    return Plan(ASK, what, what, "text", f"vm:{name}")


def _run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    if plan.payload and "error" in plan.payload:
        return error(plan.payload["error"])
    try:
        backend = get_backend(tctx.cfg)
        action = args["action"]
        if action == "status":
            st = backend.status()
            return ToolResult(True, "\n".join(f"{s['name']}: running (pid {s['pid']})" for s in st) or "no VMs running", f"{len(st)} running")
        name = args["name"]
        msg = backend.start(name, spec_for(tctx.cfg, name)) if action == "start" else backend.stop(name, bool(args.get("force")))
        return ToolResult(True, msg, msg)
    except (RuntimeError, ValueError) as e:
        return error(str(e))


def tool(cfg: dict) -> Tool:
    machines = list(cfg["vm"].get("machines", {}))
    name_spec: dict = {"type": "string"}
    if machines:
        name_spec["enum"] = machines
        desc = f"Start or stop a virtual machine. Configured: {', '.join(machines)}."
    else:
        desc = "Start or stop a virtual machine. None are configured yet."
    return Tool("vm", desc, schema({
        "action": {"type": "string", "enum": ["start", "stop", "status"]}, "name": name_spec, "force": {"type": "boolean"}}, ["action"]),
        _plan, _run, False, lambda a: f"{a.get('action', '')} {a.get('name', '')}".strip())
