import os
import shutil
import subprocess
import time
import unittest
import unittest.mock

from tests.helpers import (GIT_TEST_ENV, HOSTILE_KEYS, IsolatedTestCase, build_hostile, forge_signed_commit, git, make_repo,
                           program)
from xarencode.config import default_config
from xarencode.safety import ALLOW, ASK, BLOCK, classify_command
from xarencode.tools import ToolContext, build_registry
from xarencode.tools.shell import git_hardening_env, run_command


class ToolTestCase(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.cfg = default_config()
        self.reg = build_registry(self.cfg)
        self.tctx = ToolContext(self.work, self.cfg)

    def call(self, name, **args):
        tool = self.reg.get(name)
        plan = tool.plan(args, self.tctx)
        if plan.verdict == BLOCK:
            return plan, None
        return plan, tool.run(args, self.tctx, plan)


class FsTests(ToolTestCase):
    def test_read_file_paging(self):
        (self.work / "a.txt").write_text("\n".join(f"l{i}" for i in range(1, 11)))
        plan, res = self.call("read_file", path="a.txt", offset=3, limit=2)
        self.assertEqual(plan.verdict, ALLOW)
        self.assertEqual(res.content, "a.txt lines 3-4 of 10\nl3\nl4")

    def test_read_file_errors(self):
        (self.work / "b.bin").write_bytes(b"\0\1\2")
        self.assertIn("binary", self.call("read_file", path="b.bin")[1].content)
        self.assertTrue(self.call("read_file", path="missing")[1].content.startswith("ERROR:"))
        (self.work / "big.txt").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
        self.assertIn("2 MB", self.call("read_file", path="big.txt")[1].content)

    def test_read_outside_asks_and_forbidden_blocks(self):
        self.assertEqual(self.call("read_file", path="../x")[0].verdict, ASK)
        self.assertEqual(self.call("read_file", path="/proc/cpuinfo")[0].verdict, BLOCK)
        self.assertEqual(self.call("read_file", path="~/.ssh/id_rsa")[0].verdict, BLOCK)

    def test_list_dir(self):
        (self.work / "src").mkdir()
        (self.work / "src" / "m.py").write_text("")
        (self.work / "node_modules").mkdir()
        (self.work / "z.txt").write_text("")
        _, res = self.call("list_dir")
        self.assertEqual(res.content.splitlines(), ["src/", "z.txt"])
        _, res = self.call("list_dir", depth=2)
        self.assertEqual(res.content.splitlines(), ["src/", "  m.py", "z.txt"])

    def test_search_code(self):
        (self.work / "a.py").write_text("hello\nneedle here\n")
        _, res = self.call("search_code", pattern="needle")
        self.assertEqual(res.content, "a.py:2: needle here")
        _, res = self.call("search_code", pattern="zzz")
        self.assertEqual(res.content, "no matches")
        _, res = self.call("search_code", pattern="NEEDLE", ignore_case=True, glob="*.py")
        self.assertIn("a.py:2", res.content)

    def test_write_file(self):
        tool = self.reg.get("write_file")
        plan = tool.plan({"path": "d/new.txt", "content": "a\nb\n"}, self.tctx)
        self.assertEqual((plan.verdict, plan.preview_kind), (ASK, "diff"))
        self.assertFalse((self.work / "d").exists())  # planning never writes
        res = self.reg.get("write_file").run({"path": "d/new.txt", "content": "a\nb\n"}, self.tctx, plan)
        self.assertTrue(res.ok)
        self.assertEqual((self.work / "d" / "new.txt").read_text(), "a\nb\n")
        self.assertEqual(res.detail["event"], "edit_applied")

    def test_write_file_limits(self):
        self.assertEqual(self.call("write_file", path="../evil", content="x")[0].verdict, BLOCK)
        self.assertEqual(self.call("write_file", path=".git/config", content="x")[0].verdict, BLOCK)
        self.assertEqual(self.call("write_file", path="big", content="x" * (512 * 1024 + 1))[0].verdict, BLOCK)

    def test_edit_file(self):
        p = self.work / "e.txt"
        p.write_text("one\ntwo\nthree\n")
        tool = self.reg.get("edit_file")
        args = {"path": "e.txt", "old_string": "two", "new_string": "2"}
        plan = tool.plan(args, self.tctx)
        self.assertEqual(plan.verdict, ASK)
        self.assertIn("-two", plan.preview)
        self.assertIn("+2", plan.preview)
        self.assertEqual(p.read_text(), "one\ntwo\nthree\n")
        res = tool.run(args, self.tctx, plan)
        self.assertTrue(res.ok)
        self.assertEqual(p.read_text(), "one\n2\nthree\n")

    def test_edit_errors(self):
        p = self.work / "e.txt"
        p.write_text("x = 1\nx = 1\ny = 2\n")
        plan, res = self.call("edit_file", path="e.txt", old_string="x = 1", new_string="x = 2")
        self.assertEqual(plan.verdict, ALLOW)  # failing edits need no approval
        self.assertIn("2 places", res.content)
        self.assertTrue(res.content.startswith("ERROR:"))
        _, res = self.call("edit_file", path="e.txt", old_string="y = 3", new_string="y")
        self.assertIn("not found", res.content)
        self.assertIn("closest", res.content)
        _, res = self.call("edit_file", path="e.txt", old_string="x = 1", new_string="x = 3", replace_all=True)
        self.assertTrue(res.ok)
        self.assertEqual(p.read_text(), "x = 3\nx = 3\ny = 2\n")

    def test_edit_strips_line_number_prefixes(self):
        p = self.work / "n.py"
        p.write_text("a = 1\nb = 2\n")
        _, res = self.call("edit_file", path="n.py", old_string="1: a = 1  \n2| b = 2", new_string="1: a = 9\n2| b = 8")
        self.assertTrue(res.ok, res.content)
        self.assertEqual(p.read_text(), "a = 9\nb = 8\n")

    def test_edit_keeps_crlf(self):
        p = self.work / "w.txt"
        p.write_bytes(b"a\r\nb\r\n")
        _, res = self.call("edit_file", path="w.txt", old_string="a", new_string="z")
        self.assertTrue(res.ok)
        self.assertEqual(p.read_bytes(), b"z\r\nb\r\n")


class ShellTests(ToolTestCase):
    def test_run_and_exit_codes(self):
        plan, res = self.call("run_shell", command="echo hi && ls")
        self.assertEqual(plan.verdict, ALLOW)
        self.assertTrue(res.content.startswith("exit=0"))
        self.assertIn("hi", res.content)
        self.tctx.cfg["safety"]["allow"] = []
        plan = self.reg.get("run_shell").plan({"command": "exit 3"}, self.tctx)
        self.assertEqual(plan.verdict, ASK)
        res = self.reg.get("run_shell").run({"command": "exit 3"}, self.tctx, plan)
        self.assertFalse(res.ok)
        self.assertIn("exit=3", res.content)

    def test_live_output_and_env(self):
        lines = []
        self.tctx.output = lines.append
        cmd = "echo $PAGER; echo $NO_COLOR; pwd"
        plan = self.reg.get("run_shell").plan({"command": cmd}, self.tctx)
        self.reg.get("run_shell").run({"command": cmd}, self.tctx, plan)
        self.assertEqual(lines[:2], ["cat", "1"])
        self.assertTrue(lines[2].endswith("work"))

    def test_timeout_kills(self):
        plan = self.reg.get("run_shell").plan({"command": "sleep 30"}, self.tctx)
        res = self.reg.get("run_shell").run({"command": "sleep 30", "timeout": 1}, self.tctx, plan)
        self.assertFalse(res.ok)
        self.assertIn("timed out", res.content)

    def test_server_commands_redirected(self):
        plan, res = self.call("run_shell", command="npm run dev")
        self.assertIn("dev_server", res.content)

    def test_blocked_never_runs(self):
        plan, res = self.call("run_shell", command="dd if=/dev/zero of=/dev/sda")
        self.assertEqual(plan.verdict, BLOCK)
        self.assertIsNone(res)

    def test_tests_passed_event(self):
        (self.work / "t.py").write_text("print('ok')")
        plan = self.reg.get("run_shell").plan({"command": "python3 -m unittest -h"}, self.tctx)
        res = self.reg.get("run_shell").run({"command": "python3 -m unittest -h"}, self.tctx, plan)
        self.assertEqual(res.detail.get("event"), "tests_passed")

    def test_cancel_event(self):
        import threading
        self.tctx.cancelled = ev = threading.Event()
        threading.Timer(0.3, ev.set).start()
        plan = self.reg.get("run_shell").plan({"command": "sleep 20"}, self.tctx)
        res = self.reg.get("run_shell").run({"command": "sleep 20"}, self.tctx, plan)
        self.assertIn("interrupted", res.content)


class RegistryTests(ToolTestCase):
    def test_schemas_and_size(self):
        import json
        names = [s["function"]["name"] for s in self.reg.schemas()]
        self.assertEqual(names, ["read_file", "list_dir", "search_code", "write_file", "edit_file", "run_shell", "dev_server"])  # no vm tool without machines
        self.assertLess(len(json.dumps(self.reg.schemas())) / 3, 900)


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class GitConfigTests(ToolTestCase):
    """A repository's own config can name programs that read-only git commands run (fsmonitor, textconv, gpg.program,
    clean filters ...). Such a repository must ask, and even a command forced past that must not run the program."""

    def setUp(self):
        super().setUp()
        patcher = unittest.mock.patch.dict(os.environ, GIT_TEST_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.marker = self.tmp / "PWNED"
        self.verdict = lambda cmd, wd=None: classify_command(cmd, wd or self.work, self.cfg).verdict

    def hardened(self, cmd, wd=None):
        self.marker.unlink(missing_ok=True)
        run_command(cmd, wd or self.work, timeout=20, harden_git=True)
        return self.marker.exists()

    def unprotected(self, cmd, wd=None):
        self.marker.unlink(missing_ok=True)
        run_command(cmd, wd or self.work, timeout=20)
        return self.marker.exists()

    def test_a_clean_repository_is_allowed_everywhere_it_is_reached(self):
        repo = make_repo(self.work / "clean")
        git(repo, "config", "remote.origin.url", "https://example.invalid/x.git")
        git(repo, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
        for key, val in [("branch.main.remote", "origin"), ("branch.main.merge", "refs/heads/main"), ("color.ui", "auto"),
                         ("core.autocrlf", "input"), ("pull.rebase", "true"), ("merge.ff", "only"), ("init.defaultBranch", "main")]:
            git(repo, "config", key, val)
        for cmd in ["git -C clean log -p -1", "git -C clean diff HEAD~1", "git -C clean show HEAD", "git -C ./clean status",
                    "git --git-dir=clean/.git --work-tree=clean log -1", "git -C clean -C . status"]:
            self.assertEqual(self.verdict(cmd), ALLOW, cmd)
        for cmd in ["git log -p -1", "git diff HEAD~1", "git status", "git -C . log", "git show HEAD^"]:
            self.assertEqual(self.verdict(cmd, repo), ALLOW, cmd)
        self.assertEqual(self.verdict("git log -1", self.tmp), ALLOW)  # no repository at all: git just fails

    def test_every_hostile_repository_asks_however_it_is_reached(self):
        hostile = build_hostile(self.work, self.marker)
        for name, (repo, triggers) in hostile.items():
            for cmd in [f"git -C {name} log -p -1", f"git -C {name} show HEAD", f"git -C {name} status", f"git -C {name}/. diff HEAD~1",
                        f"git --git-dir={name}/.git --work-tree={name} log -p -1", f"git -C {name} -C . log",
                        f"git --git-dir {name}/.git log -1", f"git -C {name}/.git log -1"]:
                self.assertEqual(self.verdict(cmd), ASK, (name, cmd))
            for cmd in ["git log -p -1", "git show HEAD", "git status", "git diff HEAD~1", "git log --oneline -1", "git blame a.txt"]:
                self.assertEqual(self.verdict(cmd, repo), ASK, (name, cmd))
            self.assertEqual(self.verdict("git log -1", repo / "sub"), ASK, name)  # a subfolder uses the same repository
            d = classify_command(f"git -C {name} log", self.work, self.cfg)
            self.assertIsNone(d.approval_key, "an 'always' answer must not carry over to another repository")
        self.assertFalse(self.marker.exists())

    def test_classification_alone_never_ran_anything(self):
        build_hostile(self.work, self.marker)
        (self.work / "x").mkdir()
        for name in ("fsmonitor", "textconv", "gpg", "include"):
            self.verdict(f"git -C {name} log -p")
        self.assertFalse(self.marker.exists())

    def test_hardening_stops_the_programs_even_when_classification_is_bypassed(self):
        hostile = build_hostile(self.work, self.marker)
        for name, (repo, triggers) in hostile.items():
            for cmd in triggers:
                if triggers[0] != "git status" or name != "filter":  # git status only runs a clean filter sometimes
                    self.assertTrue(self.unprotected(cmd, repo), f"control: {cmd} should run the program ({name})")
                self.assertFalse(self.hardened(cmd, repo), f"{name}: {cmd} ran a program from the repo config")
                self.assertFalse(self.hardened(cmd.replace("git ", f"git -C {name} ", 1)),
                                 f"{name}: git -C {name} {cmd} ran a program")
                self.assertFalse(self.hardened(cmd.replace("git ", f"git --git-dir={name}/.git --work-tree={name} ", 1)),
                                 f"{name}: --git-dir form of {cmd} ran a program")
        self.assertTrue(self.unprotected("git diff", hostile["filter"][0]), "control: the clean filter runs on git diff")

    def test_forcing_the_allow_execution_path_through_run_shell_runs_nothing(self):
        """The same, but through the real tool: whatever the plan says, an ALLOW run is hardened."""
        hostile = build_hostile(self.work, self.marker)
        tool = self.reg.get("run_shell")
        for name, (repo, triggers) in hostile.items():
            for cmd in triggers:
                cmd = cmd.replace("git ", f"git -C {name} ", 1)
                plan = tool.plan({"command": cmd}, self.tctx)
                self.assertEqual(plan.verdict, ASK, cmd)
                plan.verdict = ALLOW  # as if the classifier had been fooled
                self.marker.unlink(missing_ok=True)
                tool.run({"command": cmd}, self.tctx, plan)
                self.assertFalse(self.marker.exists(), cmd)

    def test_nested_repository_with_textconv_through_dash_c(self):
        """The verifier's case: the workdir is clean, the nested `evil` repo has a textconv driver in its attributes."""
        make_repo(self.work)
        evil = make_repo(self.work / "evil")
        (evil / ".gitattributes").write_text("* diff=foo\n")
        git(evil, "config", "diff.foo.textconv", f"touch '{self.marker}'; cat")
        git(evil, "config", "diff.external", f"sh -c 'touch {self.marker}x'")
        for cmd in ["git -C evil log -p -1", "git -C evil show HEAD", "git -C evil diff HEAD~1",
                    "git --git-dir=evil/.git --work-tree=evil log -p -1"]:
            self.assertEqual(self.verdict(cmd), ASK, cmd)
            self.assertFalse(self.hardened(cmd), cmd)
            self.assertFalse(self.marker.with_name("PWNEDx").exists(), cmd)
        self.assertEqual(self.verdict("git status"), ALLOW)  # the workdir's own repository is fine

    def test_signed_commit_with_a_repo_gpg_program(self):
        repo = make_repo(self.work / "r")
        forge_signed_commit(repo)
        git(repo, "config", "gpg.program", str(program(self.tmp / "fakegpg", self.marker)))
        git(repo, "config", "log.showSignature", "true")
        for cmd in ["git log -1", "git log --oneline -1", "git show", "git log --show-signature -1"]:
            self.assertEqual(self.verdict(cmd, repo), ASK, cmd)
            self.assertFalse(self.hardened(cmd, repo), cmd)
        self.assertTrue(self.unprotected("git log -1", repo), "control: the unprotected command runs gpg.program")

    def test_each_unsafe_key_is_enough_to_ask(self):
        for i, (key, value) in enumerate(HOSTILE_KEYS):
            repo = make_repo(self.work / f"k{i}")
            git(repo, "config", key, value)
            self.assertEqual(self.verdict("git log -1", repo), ASK, key)
            self.assertEqual(self.verdict(f"git -C k{i} status"), ASK, key)

    def test_unparseable_config_or_unreadable_git_dir_asks(self):
        repo = make_repo(self.work / "broken")
        with open(repo / ".git" / "config", "a") as f:
            f.write("\n[core\nfsmonitor = x\n")
        self.assertEqual(self.verdict("git status", repo), ASK)
        gitfile = self.work / "link"
        gitfile.mkdir()
        (gitfile / ".git").write_text("gitdir: ../missing\n")
        self.assertEqual(self.verdict("git status", gitfile), ASK)

    def test_gitfile_worktrees_and_parent_repositories_are_followed(self):
        main = make_repo(self.work / "main")
        git(main, "worktree", "add", "-q", str(self.work / "wt"), "-b", "side")
        self.assertEqual(self.verdict("git status", self.work / "wt"), ALLOW)
        git(main, "config", "core.fsmonitor", str(program(self.tmp / "p", self.marker)))  # common config now hostile
        self.assertEqual(self.verdict("git status", self.work / "wt"), ASK)
        self.assertEqual(self.verdict("git -C wt status"), ASK)
        (main / "sub" / "deeper").mkdir(parents=True)
        self.assertEqual(self.verdict("git status", main / "sub" / "deeper"), ASK)  # found by walking up

    def test_paths_outside_the_workdir_or_unknowable_ask(self):
        other = make_repo(self.tmp / "other")  # a perfectly clean repository, but not under the working directory
        for cmd in [f"git -C {other} log", "git -C .. log", f"git --git-dir={other}/.git log", "git -C ~ log", "git -C $HOME log",
                    "git -C $X log", f"git --work-tree={other} status", "git -C /tmp log", "git -C"]:
            self.assertEqual(self.verdict(cmd), ASK, cmd)
        for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_EXTERNAL_DIFF", "GIT_CONFIG_PARAMETERS"):
            with unittest.mock.patch.dict(os.environ, {var: "x"}):
                self.assertEqual(self.verdict("git status"), ASK, var)
        self.assertEqual(self.verdict("GIT_DIR=x git status"), ASK)

    def test_a_committed_nested_repository_cannot_run_its_fsmonitor_through_the_outer_status(self):
        """`git status` in a clean outer repo asks the nested repo (a gitlink) for its state, and that reads ITS config."""
        outer = make_repo(self.work / "outer")
        inner = make_repo(outer / "inner")
        git(inner, "config", "core.fsmonitor", str(program(self.tmp / "p", self.marker)))
        (inner / "g").write_text("dirty\n")
        git(outer, "add", "inner")
        git(outer, "commit", "-qm", "gitlink")
        self.assertTrue(self.unprotected("git status", outer), "control: the unprotected status runs the inner fsmonitor")
        self.assertFalse(self.hardened("git status", outer))
        self.assertFalse(self.hardened("git diff", outer))
        self.assertFalse(self.hardened("git log -p -1", outer))
        self.assertEqual(self.verdict("git status", outer), ASK)  # and the classifier no longer calls such a repository clean

    NESTED_KEYS = ["filter.foo.clean", "filter.foo.process", "diff.foo.textconv", "diff.external", "diff.foo.command", "core.fsmonitor"]
    NESTED_CMDS = ["git status", "git diff", "git log -p -1", "git show HEAD", "git -C {r} status", "git -C {r} diff",
                   "git --git-dir={r}/.git --work-tree={r} diff", "git --git-dir={r}/.git --work-tree={r} status",
                   "git diff --submodule=diff", "git log -p -1 --submodule=diff", "git status --ignore-submodules=none",
                   "git diff --ignore-submodules=none", "git status --porcelain"]

    def nested_fixture(self, name, key, how="gitlink"):
        """A clean outer repository holding a nested repository whose own config names the program (the verifier's case)."""
        outer = make_repo(self.work / name)
        inner = make_repo(outer / "sub")
        prog = program(self.tmp / f"prog_{name}", self.marker)
        git(inner, "config", key, str(prog))
        (inner / ".gitattributes").write_text("* diff=foo filter=foo\n")
        (inner / "a.txt").write_text("four, not yet committed\n")
        os.utime(inner / "a.txt", (time.time() + 5, time.time() + 5))
        if how == "gitlink":
            git(outer, "add", "sub")
            git(outer, "commit", "-qm", "gitlink")
        return outer, inner

    def test_a_nested_repository_with_a_hostile_config_asks_and_cannot_run_anything(self):
        """A committed nested repository (gitlink) is read with ITS OWN config. Every key that names a program must make the
        outer git command ask, and even a command forced to ALLOW must not fire it (checked through the real run_shell)."""
        tool = self.reg.get("run_shell")
        for i, key in enumerate(self.NESTED_KEYS):
            name = f"n{i}"
            outer, inner = self.nested_fixture(name, key)
            fired_unprotected = False
            for tmpl in self.NESTED_CMDS:
                cmd = tmpl.format(r=name)
                wd = self.work if "{r}" in tmpl else outer
                self.assertEqual(self.verdict(cmd, wd), ASK, (key, cmd))
                fired_unprotected |= self.unprotected(cmd, wd)
                self.assertFalse(self.hardened(cmd, wd), (key, cmd, "hardened run fired the nested repository's program"))
                plan = tool.plan({"command": cmd}, ToolContext(wd, self.cfg))
                self.assertEqual(plan.verdict, ASK, cmd)
                plan.verdict = ALLOW  # as if the classifier had been fooled
                self.marker.unlink(missing_ok=True)
                tool.run({"command": cmd}, ToolContext(wd, self.cfg), plan)
                self.assertFalse(self.marker.exists(), (key, cmd, "run_shell fired the program"))
            self.assertTrue(fired_unprotected, f"control: {key} must fire in at least one unprotected command")

    def test_nested_repositories_are_found_below_the_first_level_and_in_the_gitmodules_case(self):
        outer, inner = self.nested_fixture("deep", "core.fsmonitor")
        git(inner, "config", "--unset", "core.fsmonitor")  # sub is clean, but ITS nested repository is not
        deeper = make_repo(inner / "subsub")
        git(deeper, "config", "filter.foo.clean", str(program(self.tmp / "p2", self.marker)))
        git(inner, "add", "subsub")
        git(inner, "commit", "-qm", "deeper gitlink")
        self.assertEqual(self.verdict("git status", outer), ASK)
        # a tracked .gitmodules that sets ignore = none does not get around the static override either
        (outer / ".gitmodules").write_text('[submodule "sub"]\n\tpath = sub\n\turl = ./sub\n\tignore = none\n')
        git(outer, "add", ".gitmodules")
        self.assertFalse(self.hardened("git status", outer))
        self.assertFalse(self.hardened("git diff --submodule=diff", outer))

    def test_a_nested_repository_is_found_from_a_subfolder_and_through_dash_c(self):
        outer, inner = self.nested_fixture("sf", "core.fsmonitor")
        (outer / "src" / "deep").mkdir(parents=True)
        for wd, cmd in [(outer / "src", "git status"), (outer / "src" / "deep", "git diff"), (self.work, "git -C sf/src status"),
                        (outer, "git -C src log -p -1"), (outer, "cd src && git status")]:
            self.assertEqual(self.verdict(cmd, wd), ASK, (wd, cmd))
        self.assertFalse(self.hardened("git status", outer / "src"))

    def test_a_clean_nested_repository_and_a_real_submodule_stay_usable(self):
        outer = make_repo(self.work / "ok")
        inner = make_repo(outer / "sub")
        git(outer, "add", "sub")
        git(outer, "commit", "-qm", "gitlink")
        for cmd in ["git status", "git diff", "git log -p -1", "git show HEAD", "git diff HEAD~1"]:
            self.assertEqual(self.verdict(cmd, outer), ALLOW, cmd)
        (outer / "plain.txt").write_text("x\n")
        self.assertIn("plain.txt", subprocess.run(["bash", "-c", "git status --short"], cwd=outer, capture_output=True, text=True,
                                                   env=dict(os.environ, **git_hardening_env(outer, dict(os.environ), "git status"))).stdout)
        # an untracked nested repository (not in the index) is only listed as a folder by git status
        make_repo(outer / "loose")
        self.assertEqual(self.verdict("git status", outer), ALLOW)
        # a submodule added the normal way brings submodule.* keys into the outer config: ask
        src = make_repo(self.tmp / "src")
        git(outer, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(src), "sm")
        self.assertEqual(self.verdict("git status", outer), ASK)

    def test_submodule_arguments_ask(self):
        repo = make_repo(self.work / "ok")
        for arg in ["--ignore-submodules", "--ignore-submodules=none", "--ignore-sub", "--submodule", "--submodule=diff", "--submodule=log",
                    "--recurse-submodules", "--no-recurse-submodules", "--recurse"]:
            self.assertEqual(self.verdict(f"git status {arg}", repo), ASK, arg)
            self.assertEqual(self.verdict(f"git diff {arg}", repo), ASK, arg)
            self.assertEqual(self.verdict(f"git log -p {arg}", repo), ASK, arg)
        for cmd in ["git status", "git diff --stat", "git log --oneline", "git show --stat"]:
            self.assertEqual(self.verdict(cmd, repo), ALLOW, cmd)

    def test_hardening_names_the_submodule_keys(self):
        env = git_hardening_env(self.work, dict(os.environ), "git status")
        pairs = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
        self.assertEqual(pairs["diff.ignoresubmodules"], "all")
        self.assertEqual(pairs["status.submodulesummary"], "false")
        self.assertEqual(pairs["submodule.recurse"], "false")
        self.assertEqual(pairs["fetch.recursesubmodules"], "false")

    def test_a_credential_file_asks_through_the_shell_and_read_file_with_no_key(self):
        (self.home / ".aws").mkdir()
        (self.home / ".aws" / "credentials").write_text("k\n")
        plan, _ = self.call("read_file", path="~/.aws/credentials")
        self.assertEqual((plan.verdict, plan.approval_key), (ASK, None))
        self.assertEqual(self.reg.get("run_shell").plan({"command": "cat ~/.aws/credentials"}, self.tctx).verdict, ASK)
        self.assertEqual(self.reg.get("run_shell").plan({"command": "cat /etc/os-release"}, self.tctx).verdict, ALLOW)

    def test_alias_and_include_asks_but_aliases_cannot_shadow_builtins(self):
        repo = make_repo(self.work / "al")
        git(repo, "config", "alias.log", f"!touch '{self.marker}'")
        self.assertEqual(self.verdict("git log", repo), ASK)
        self.assertFalse(self.hardened("git log -1", repo))  # git ignores aliases named like builtins


if __name__ == "__main__":
    unittest.main()
