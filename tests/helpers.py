"""Shared test helpers: isolated XDG dirs and a scripted frontend."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class IsolatedTestCase(unittest.TestCase):
    """Points XDG_* and HOME at a temp dir so tests never touch the real home."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._saved = {k: os.environ.get(k) for k in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")}
        self.home = self.tmp / "home"
        self.home.mkdir()
        os.environ["HOME"] = str(self.home)
        os.environ["XDG_CONFIG_HOME"] = str(self.tmp / "cfg")
        os.environ["XDG_DATA_HOME"] = str(self.tmp / "data")
        os.environ["XDG_STATE_HOME"] = str(self.tmp / "state")
        self.work = self.tmp / "work"
        self.work.mkdir()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()


# --- scratch git repositories for the git safety tests (offline; no gpg needed; never reads the user's git config)
GIT_TEST_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(cwd: Path, *args: str, inp: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, input=inp, timeout=30,
                          stdin=None if inp is not None else subprocess.DEVNULL, env=dict(os.environ, **GIT_TEST_ENV)).stdout


def make_repo(path: Path) -> Path:
    """A repository with two commits (a.txt: one, two) and only harmless local config."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    git(path, "config", "user.name", "t")
    git(path, "config", "user.email", "t@t")
    (path / "a.txt").write_text("one\n")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "one")
    (path / "a.txt").write_text("two\n")
    git(path, "commit", "-qam", "two")
    return path


def program(path: Path, marker: Path) -> Path:
    """An executable that records that it ran (and fails, so git never uses its output)."""
    path.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
    path.chmod(0o755)
    return path


def forge_signed_commit(repo: Path) -> None:
    """Rewrites HEAD as a commit carrying a (fake) gpgsig header, as a signed commit has. No gpg is involved."""
    tree = git(repo, "rev-parse", "HEAD^{tree}").decode().strip()
    parent = git(repo, "rev-parse", "HEAD").decode().strip()
    body = (f"tree {tree}\nparent {parent}\nauthor t <t@t> 1 +0000\ncommitter t <t@t> 1 +0000\n"
            "gpgsig -----BEGIN PGP SIGNATURE-----\n \n AAAA\n -----END PGP SIGNATURE-----\n\nsigned\n")
    sha = git(repo, "hash-object", "-t", "commit", "-w", "--stdin", inp=body.encode()).decode().strip()
    git(repo, "update-ref", "HEAD", sha)


def hostile_fsmonitor(repo, prog):
    git(repo, "config", "core.fsmonitor", str(prog))
    return ["git status"]


def hostile_hookspath(repo, prog):
    hooks = repo / "myhooks"
    hooks.mkdir()
    for h in ("pre-commit", "post-checkout", "post-merge", "reference-transaction"):
        program(hooks / h, prog.with_name("unused"))
    git(repo, "config", "core.hooksPath", str(hooks))
    return []


def hostile_pager(repo, prog):
    git(repo, "config", "core.pager", str(prog))
    git(repo, "config", "pager.log", str(prog))
    return []


def hostile_textconv(repo, prog):
    (repo / ".gitattributes").write_text("* diff=foo\n")
    git(repo, "config", "diff.foo.textconv", str(prog))
    return ["git log -p -1", "git show HEAD", "git diff HEAD~1"]


def hostile_external_diff(repo, prog):
    git(repo, "config", "diff.external", str(prog))
    return ["git diff HEAD~1"]


def hostile_filter(repo, prog):
    (repo / ".gitattributes").write_text("* filter=foo\n")
    git(repo, "config", "filter.foo.clean", str(prog))
    git(repo, "config", "filter.foo.process", str(prog))
    (repo / "a.txt").write_text("three, not yet committed\n")
    return ["git status", "git diff"]


def hostile_gpg(repo, prog):
    forge_signed_commit(repo)
    git(repo, "config", "gpg.program", str(prog))
    git(repo, "config", "log.showSignature", "true")
    return ["git log -1", "git show", "git log --oneline -1", "git log --show-signature -1"]


def hostile_include(repo, prog):
    inc = repo.parent / (repo.name + ".inc")
    inc.write_text(f"[core]\n\tfsmonitor = {prog}\n")
    git(repo, "config", "include.path", str(inc))
    return ["git status"]


def hostile_alias(repo, prog):
    git(repo, "config", "alias.log", f"!{prog}")
    git(repo, "config", "alias.st", f"!{prog}")
    return []


HOSTILE_REPOS = {"fsmonitor": hostile_fsmonitor, "hookspath": hostile_hookspath, "pager": hostile_pager,
                 "textconv": hostile_textconv, "external": hostile_external_diff, "filter": hostile_filter,
                 "gpg": hostile_gpg, "include": hostile_include, "alias": hostile_alias}
# a single hostile or unknown key is enough to make git ask; none of these is on the safe list
HOSTILE_KEYS = [("core.sshCommand", "x"), ("core.editor", "x"), ("credential.helper", "x"), ("url.http://a/.insteadOf", "b"),
                ("safe.directory", "*"), ("core.worktree", "/tmp"), ("merge.x.driver", "x"), ("includeIf.gitdir:/.path", "x"),
                ("diff.x.command", "x"), ("gpg.ssh.program", "x"), ("gpg.x509.program", "x"), ("log.showSignature", "true"),
                ("core.attributesFile", "x"), ("status.showUntrackedFiles", "no"), ("sendemail.smtpServer", "x")]


def build_hostile(base: Path, marker: Path) -> dict[str, tuple[Path, list[str]]]:
    """One hostile repository per kind under `base`: name -> (repo, commands that run the program when unprotected)."""
    prog = program(base / "prog.sh", marker)
    out = {}
    for name, builder in HOSTILE_REPOS.items():
        repo = make_repo(base / name)
        out[name] = (repo, builder(repo, prog))
    return out


def build_nested(base: Path, marker: Path) -> None:
    """Clean outer repositories whose committed nested repository (a gitlink) names a program in ITS OWN config.
    ng_clean has a harmless nested repository; ng_deep keeps the hostile one a level further down; ng_mod adds a tracked
    .gitmodules with `ignore = none`. The nested working files are modified (newer mtime) so that git must look at them."""
    prog = program(base / "nprog.sh", marker)
    keys = {"ng_tc": [("diff.foo.textconv", prog)], "ng_ext": [("diff.external", prog)], "ng_fsm": [("core.fsmonitor", prog)],
            "ng_cmd": [("diff.foo.command", prog), ("filter.foo.clean", prog), ("filter.foo.process", prog)],
            "ng_mod": [("filter.foo.clean", prog), ("diff.foo.textconv", prog)], "ng_ok": [], "ng_clean": [], "ng_deep": [("filter.foo.clean", prog), ("diff.foo.textconv", prog)]}
    for name, pairs in keys.items():
        outer = make_repo(base / name)
        inner = make_repo(outer / "sub")
        target = inner
        if name == "ng_deep":
            target = make_repo(inner / "subsub")
            git(inner, "add", "subsub")
            git(inner, "commit", "-qm", "deeper")
        for k, v in pairs:
            git(target, "config", k, str(v))
        (target / ".gitattributes").write_text("* diff=foo filter=foo\n")
        (target / "a.txt").write_text("dirty\n")
        os.utime(target / "a.txt", (time.time() + 5, time.time() + 5))
        if name == "ng_mod":
            (outer / ".gitmodules").write_text('[submodule "sub"]\n\tpath = sub\n\turl = ./sub\n\tignore = none\n')
            git(outer, "add", ".gitmodules")
        git(outer, "add", "sub")
        git(outer, "commit", "-qm", "gitlink")


def _build_scripted():
    from xarencode.agent import Approval, BaseFrontend

    class ScriptedFrontend(BaseFrontend):
        """Records every call; approvals come from a queue (default: yes)."""

        def __init__(self, approvals=None):
            self.events: list[tuple] = []
            self.approvals = list(approvals or [])
            self.text_buf = ""
            self.stats_seen = []

        def text(self, d):
            self.text_buf += d

        def reasoning(self, d):
            self.events.append(("reasoning", d))

        def approve(self, call, plan):
            self.events.append(("approve", call.name, plan.verdict.name))
            return self.approvals.pop(0) if self.approvals else Approval("yes")

        def tool_output(self, line):
            self.events.append(("output", line))

        def tool_done(self, call, result):
            self.events.append(("done", call.name, result.ok, result.content))

        def notice(self, level, text):
            self.events.append(("notice", level, text))

        def stats(self, s):
            self.stats_seen.append(s)

        def names(self, kind):
            return [e for e in self.events if e[0] == kind]

    return ScriptedFrontend


def __getattr__(name):  # lazy so early test modules do not need the agent
    if name == "ScriptedFrontend":
        cls = _build_scripted()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
