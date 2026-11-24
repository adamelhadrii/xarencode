"""Differential test for the shell classifier.

It generates command strings, classifies them, runs ONLY the ones classified ALLOW in a throwaway sandbox (a scratch
HOME and working directory, a 2 s timeout, no absolute paths except /dev/null) and checks that nothing in the sandbox
changed and that no process is left running (found by an environment marker, killed afterwards). It executes commands, so it is skipped unless XARENCODE_SLOW=1. Knobs: XARENCODE_DIFF_N (default 300 strings)
and XARENCODE_DIFF_SEED (default 1).

The sandbox also holds files named like options (-delete, --pre=sh, -o ...) for the glob families, one clean git
repository, and one hostile repository per kind (fsmonitor, textconv, gpg.program + signed commit, include.path, ...): the
programs they name touch a marker in the sandbox, so any ALLOW that runs one is reported as a mutation. Nested
repositories (a clean outer repository whose committed gitlink has a hostile filter, textconv, diff.external or
fsmonitor config, one level and two levels deep, with and without a .gitmodules `ignore = none`) are in the sandbox too.
Every ALLOW string is run twice when it mentions git: once with the plain environment (the classifier alone must be right)
and once with the hardening environment that run_shell applies. "program ran" counts markers touched by either run.
Credential files sit in the scratch HOME so that reads of them are classified (they ask; a mutation is not expected).

Standalone: PYTHONPATH=src python3 -B -m tests.test_differential N SEED
"""
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import GIT_TEST_ENV, build_hostile, build_nested, make_repo
from xarencode.safety import ALLOW, classify_command
from xarencode.tools.shell import git_hardening_env

BASE = ["ls", "cat a", "echo x", "printf %s y", "head -1 a", "wc -l a", "grep x a", "sort a", "uniq a", "cut -c1 a",
        "tail -n1 a", "find . -name a", "git status", "git log", "git diff", "git -C . status", "git config x y",
        "git branch -m x", "git stash", "git checkout .", "git init", "git gc", "git diff --output=f", "git log --outp=f",
        "tee f", "sed -i s/a/b/ a", "sed -n 'w f' a", "sed e a", "cp a f", "mv a f", "ln -s a f", "install a f",
        "truncate -s0 a", "script -q f", "tar cf f a", "zip f a", "unzip a", "python3 -c 'open(\"f\",\"w\")'",
        "perl -e 'open F,\">f\"'", "awk 'BEGIN{system(\"touch f\")}'", "curl -o f x", "wget x", "ssh x", "touch f",
        "find . -exec touch f \\;", "find . -fprint f", "find . -fls f", "find . -delete", "find . -okdir touch f \\;",
        "xargs touch", "date", "date -s 1", "pwd", "stat a", "diff a a", "true", "less a", "man ls", "ls -R .", "ls -la",
        "grep --exclude-from=a x a", "printf -v v x", "read v", "declare v=1", "export v=1", "cd ..", "source a", ". ./a",
        "builtin echo x", "command touch f", "exec touch f", "eval touch f", "f(){ touch f; }; f", "alias ls=touch; ls f",
        "echo $((1+2))", "echo ${v:-$(touch f)}", "cat <(touch f)", "echo ~root", "sleep 5", "yes", "nohup sleep 5",
        "sleep 9 &", "printf '%s' \">\"", "echo '>' f", "echo \">\"", "cat", "cat ~/.aws/credentials", "cat $HOME/.netrc", "grep -r x ~/.config", "cat ~/.config/gh/hosts.yml", "head ~/.npmrc",
        "git lfs prune", "git submodule foreach touch f", "git prune", "git gc --prune=now", "cp /dev/null a", "trash a",
        "rsync --delete -a . g", "sed -i s/b/c/ a", "uvx x", "go run .", "nc -l 9", "yt-dlp x"]
FLAGS = ["sort --o=f a", "sort --outp f a", "sort --out=f a", "sort -o f a", "sort -of a", "sort --output f a",
         "sort --compress-prog=touch a", "sort --comp=touch a", "file -C -m mg", "file --compile -m mg", "file --comp -m mg",
         "file -m mg a", "file --magic=mg a", "file a", "tail -f a", "tail --foll a", "tail --f a", "tail -F a", "tail -n1 a",
         "date -s now", "date --se=now", "date --s=now", "ss -D f", "ss -K", "ss --ki", "ss --diag=f", "tree -o f",
         "hostname -F a", "uniq a f", "uniq -f1 a f", "git diff --no-index --out=f a a", "git diff --no-index --output=f a a",
         "git diff --no-index --ext a a", "git branch --list", "git branch newb", "git branch -l -D x", "git tag -l",
         "git tag -l -d x", "git remote -v", "grep -d recurse x ~", "grep -drecurse x ~", "grep --dir=recurse x .",
         "grep --directories recurse x .", "grep -r x ~", "grep --rec x ~", "rg x ~", "rg --pre=touch x a", "diff -r a ~",
         "du --files0-from=a", "wc --files0-from=a", "cut -d: -f1 a", "stat -c %s a", "env ls", "env -S 'touch f'",
         "env --sp='touch f' ls", "env -i ls", "env -- ls", "time -o f ls", "time -p ls", "exec -a zzz ls",
         "stdbuf -oL ls", "stdbuf -o L ls", "nice -n 5 ls", "timeout 1 ls", "nohup ls", "command ls", "command -v ls",
         "xargs -a a touch"]
WORDS = ["-o", "f", "-delete", "-fprint", "--output=f", "-C", "-m", "mg", "touch", "a"]
PLAIN_READ = ["ls", "cat", "sort", "uniq", "find .", "file", "tail", "head", "grep x", "git log", "git diff", "date", "wc",
              "echo", "stat", "du", "git status", "diff", "rg x", "cut -c1", "tree"]
NESTED_REPOS = ["ng_clean", "ng_tc", "ng_ext", "ng_fsm", "ng_cmd", "ng_deep", "ng_mod", "ng_ok", "ng_clean/sub", "ng_tc/sub"]
GIT_REPOS = NESTED_REPOS + ["clean", "fsmonitor", "textconv", "gpg", "include", "external", "filter", "alias", "pager", "hookspath", ".", "clean/.",
             "./clean", "clean/..", "nope", "textconv/.git", "clean/.git"]
GIT_SUBS = ["log", "log -p", "log -p -1", "log --oneline -1", "log --show-signature -1", "log --textconv -1", "show", "show HEAD",
            "show HEAD~1", "show HEAD^", "diff", "diff HEAD~1", "diff HEAD^", "diff HEAD~", "diff --stat", "diff --ext-diff",
            "status", "blame a.txt", "log HEAD~1..HEAD", "ls-files", "rev-parse HEAD~1", "describe", "shortlog", "branch", "tag",
            "remote -v", "log -p main~1", "diff --no-ext-diff", "log HEAD~1 -- a.txt", "stash list", "config core.pager x",
            "diff --submodule=diff", "log -p -1 --submodule=diff", "status --ignore-submodules=none", "diff --ignore-submodules=none",
            "status --porcelain", "show --submodule=log", "grep --recurse-submodules x", "ls-files --recurse-submodules",
            "status --submodule", "diff --cached", "log -p --stat", "status -uall"]
GIT_OPTS = ["", "--no-pager ", "--no-optional-locks "]
GLOB_WORDS = ["*", "./*", "-*", "--*", "?", "[-a]*", "./-*", "-d*", "--o*", "*/*", ".*", "a *", "-- *", "*.txt"]
GLOB_CMDS = ["ls", "cat", "find", "find .", "rg x", "file", "tail", "tail -n1", "sort", "uniq", "head", "wc", "wc -l", "grep x", "grep -l x",
             "du", "stat", "diff a", "tree", "git log", "echo", "printf %s", "cut -c1", "date", "xargs", "env", "sort -u"]
OPTION_FILES = ["-delete", "--pre=sh", "-o", "--output=x", "-exec", "-fprint", "-n", "--files0-from=a", "-f", "-C"]
SEPS = [";", " ; ", "&&", " || ", "|", " | ", "\n", " & ", ";\n", "|&", " #'\n", " #\"\n", " # \n", "\r\n", " \\\n"]
REDIRS = [">f", " > f", ">>f", "2>f", "&>f", ">|f", "1>f", " >/dev/null", " 2>/dev/null", " 2>&1", " >&2", "<a", "<<<x",
          " <<E\nx\nE\n", " <>f", ">&f", " >\"f\"", " >'f'", " >$HOME/f", " >~/f", " >${X:-f}", " >$X", ">\\\nf", " >f\\\n",
          " {fd}>f", " #>f", " <<'E'\n$'\\'' touch f\nE\n", " <<E", " >{f,g}"]
WRAPS = [lambda s: s, lambda s: f"({s})", lambda s: "{ " + s + "; }", lambda s: f"$({s})", lambda s: f"echo `{s}`",
         lambda s: f"echo \"$({s})\"", lambda s: f"bash -c '{s}'", lambda s: f"cat <({s})", lambda s: f"X=1 {s}",
         lambda s: f"env {s}", lambda s: f"command {s}", lambda s: f"nohup {s}", lambda s: f"timeout 2 {s}",
         lambda s: f"echo \"a>b\" {s}", lambda s: f"echo $'\\'' {s} \\'", lambda s: f"echo $'a\\'b' {s}",
         lambda s: f"echo \\\" {s} \\\"", lambda s: f"eval '{s}'", lambda s: f"echo ${{X:-'}}' {s} '",
         lambda s: f"echo \"${{X:-'}}\" {s}", lambda s: f"{s} #>f", lambda s: f"echo $\"{s}\"", lambda s: f"echo '\\' {s}",
         lambda s: f"echo \"\\\\\" {s} \"x\"", lambda s: f"cat <<'E'\n{s}\nE",
         lambda s: f"echo $'\\''; {s}; \\'", lambda s: f"echo $'\\'' ; {s} ; echo \\'", lambda s: f"x=$'\\'' {s} \\'",
         lambda s: f"echo \"$(echo $'\\'' ; {s} ; \\')\"", lambda s: f"echo \"it's\" {s} 'x'",
         lambda s: f"bash -c \"echo \\$HOME; {s}\"", lambda s: f"bash -c \"echo \\$'\\\\'' ; {s}; \\\\'\"",
         lambda s: f"eval \"echo \\`x\\`; {s}\"", lambda s: f"echo \"{s}\"", lambda s: f"sh -c \"{s}\"",
         lambda s: f"echo '{s}'; {s}", lambda s: f"echo `echo $'\\''` ; {s} ; echo `\\'`",
         lambda s: f"echo #'\n{s}\n'", lambda s: f"echo #\"\n{s}\n\"", lambda s: f"ls # `\n{s}\n`",
         lambda s: f"echo a #'\n{s}\n'", lambda s: f"echo \"$(echo ')')\" {s}", lambda s: f"echo \"\\'\" {s}",
         lambda s: f"echo \"\\\"\" {s}", lambda s: f"f(){{ {s}; }}; f", lambda s: f"exec {s}", lambda s: f"builtin {s}",
         lambda s: f"echo $(( 1 )) {s}", lambda s: f"echo ${{v:-$({s})}}", lambda s: f"xargs {s}", lambda s: f"time {s}",
         lambda s: f"echo x | {s}", lambda s: f"{s} \\#", lambda s: f"echo \\; {s}"]
QUOTES = ["", "\"", "'", "\\", "$'", "$\"", "`", "$(", ")", "#", "\\\n", "\r"]
ABSOLUTE = re.compile(r"(?<![\w.~}])/")

SOUP = ["ls", "echo", "cat", "a", "f", "touch", "tee", "sort", "x", "-r", "'", "\"", "`", "$(", ")", "(", "$'", "$\"", "\\", ">",
        "<", ";", "|", "&", " ", " ", " ", "\n", "$X", "~", "{", "}", "#", "--o=f", "2>", ">&", "&&", "'\\''", "\\'", "[", "*",
        "{-o,f}", "\\\n", "-delete", "find", ".", "#'", "#\""]


def word_mutation(rng: random.Random) -> str:
    a, b = rng.sample(WORDS, 2)
    return rng.choice([f"{{{a},{b}}}", f"'{a}'\"{b}\"", f"{a[:2]}\\\n{a[2:]}", f"\"{a[:2]}\\\n{a[2:]}\"", f"{a}\\;", f"$'{a}'",
                       f"$\"{a}\"", f"\"\\\"{a}\"", f"\"\\'{a}\"", f"'a'\"b\"'c'{a}", f"~+/{a}", f"{a}#x", f"\\{a}",
                       f"{a[:1]}{{{a[1:]},}}", f"#{a}", f"{a}{{,}}"])


def git_command(rng: random.Random) -> str:
    sub, repo, repo2, opts = rng.choice(GIT_SUBS), rng.choice(GIT_REPOS), rng.choice(GIT_REPOS), rng.choice(GIT_OPTS)
    s = rng.choice([f"git {opts}-C {repo} {sub}", f"git {opts}-C {repo} -C {repo2} {sub}", f"git --git-dir={repo}/.git {sub}",
                    f"git --git-dir={repo}/.git --work-tree={repo} {sub}", f"git --git-dir {repo} {sub}", f"git {opts}{sub}",
                    f"git --work-tree={repo} {sub}", f"git --bare -C {repo} {sub}", f"cd {repo} && git {sub}",
                    f"git -C {repo} {sub} *", f"git -C {repo} {sub} {{a,b}}", f"git -C '{repo}' {sub}", f"git -C \"{repo}\" {sub}",
                    f"(cd {repo}; git {sub})", f"GIT_DIR={repo}/.git git {sub}", f"git -C {repo} {sub} | head", f"git -C {repo}{sub}"])
    if rng.random() < .25:
        s += rng.choice(SEPS) + rng.choice(BASE)
    return s


def glob_command(rng: random.Random) -> str:
    s = rng.choice(GLOB_CMDS) + " " + rng.choice(GLOB_WORDS)
    if rng.random() < .3:
        s = rng.choice(PLAIN_READ) + rng.choice(SEPS) + s
    if rng.random() < .15:
        s += rng.choice(REDIRS)
    return s


def probes() -> list[str]:
    """Directed cases that always run first: every nested-repository fixture through every way of naming it."""
    subs = ["status", "diff", "log -p -1", "show HEAD", "status --porcelain", "diff --stat", "diff HEAD~1", "blame a.txt", "ls-files -s"]
    out = []
    for repo in NESTED_REPOS[:8]:
        for sub in subs:
            out += [f"git -C {repo} {sub}", f"git --git-dir={repo}/.git --work-tree={repo} {sub}", f"git --git-dir={repo}/.git {sub}",
                    f"cd {repo} && git {sub}", f"git -C {repo} --no-pager {sub}", f"git -C {repo}/sub {sub}"]
        out += [f"git -C {repo} diff --submodule=diff", f"git -C {repo} log -p -1 --submodule=diff", f"git -C {repo} status --ignore-submodules=none"]
    return out


def generate(rng: random.Random) -> str:
    r = rng.random()
    if r < .15:
        return git_command(rng)
    if r < .27:
        return glob_command(rng)
    if r < .4:  # token soup: mostly unparseable or ASK, now and then a surprising ALLOW
        return "".join(rng.choice(SOUP) for _ in range(rng.randint(2, 14)))
    if r < .62:  # option smuggling: an allow-listed command with mutated words
        s = rng.choice(PLAIN_READ) + "".join(" " + (word_mutation(rng) if rng.random() < .6 else rng.choice(WORDS))
                                             for _ in range(rng.randint(1, 3)))
        if rng.random() < .3:
            s += rng.choice(SEPS) + rng.choice(BASE)
        return s
    parts = []
    for _ in range(rng.randint(1, 3)):
        c = rng.choice(BASE if rng.random() < .55 else FLAGS)
        if rng.random() < .4:
            c += rng.choice(REDIRS)
        if rng.random() < .4:
            c = rng.choice(WRAPS)(c)
        parts.append(c)
    s = parts[0]
    for p in parts[1:]:
        s += rng.choice(SEPS) + p
    if rng.random() < .12:
        i = rng.randrange(len(s) + 1)
        s = s[:i] + rng.choice(QUOTES) + s[i:]
    return s


def safe_to_run(s: str) -> bool:
    """Rails: no absolute path except /dev/null, no parent directories, no sudo, no ~user, no NUL."""
    t = s.replace("/dev/null", "")
    return not (ABSOLUTE.search(t) or "sudo" in s or ".." in s or "\x00" in s or re.search(r"~[A-Za-z+-]", s))


def snapshot(root: Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        st = p.lstat()
        out[str(p)] = (st.st_size, st.st_mtime_ns, st.st_mode, os.readlink(p) if p.is_symlink() else None)
    return out


def fresh_sandbox(root: Path) -> tuple[Path, Path]:
    shutil.rmtree(root, ignore_errors=True)
    work, home = root / "a" / "b" / "w", root / "home"
    work.mkdir(parents=True)
    home.mkdir()
    (work / "a").write_text("b\na\n")
    (work / "mg").write_text("0\tstring\tXYZ\txyz\n")
    (home / ".bashrc").write_text("keep\n")
    for name in OPTION_FILES:
        (work / name).write_text("x\n")
    if shutil.which("git"):
        make_repo(work / "clean")
        build_hostile(work, root / "PWNED")  # the hostile programs touch this marker, which the snapshot would show
        build_nested(work, root / "PWNED")
    for rel in (".aws/credentials", ".netrc", ".npmrc", ".config/gh/hosts.yml"):
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text("secret\n")
    return work, home


def marked_pids(mark: str) -> list[int]:
    """Processes whose environment carries `mark`: anything the sandboxed command left running, however it detached."""
    found = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            try:
                with open(f"/proc/{d}/environ", "rb") as f:
                    if mark.encode() in f.read():
                        found.append(int(d))
            except OSError:
                pass
    return found


def run_sandboxed(s: str, work: Path, home: Path, mark: str, harden: bool = False) -> list[int]:
    """Runs one ALLOW string; returns the pids still alive afterwards (they are killed here)."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "LC_ALL": "C", mark: "1", "GIT_OPTIONAL_LOCKS": "0",
           **GIT_TEST_ENV}  # no index refresh: `git status` rewrites .git/index by design, which is not a mutation to hunt
    if harden:
        env = git_hardening_env(work, env, s)  # what run_shell does to every ALLOW command
    p = subprocess.Popen(["bash", "-c", s], cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    timed_out = False
    try:
        p.wait(timeout=2)
    except subprocess.TimeoutExpired:
        timed_out = True
    time.sleep(0.02)
    left = [] if timed_out else marked_pids(mark)  # the shell has exited: anything still marked outlived it
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()
    time.sleep(0.02)
    for pid in set(left) | set(marked_pids(mark)):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return left


def run_differential(n: int, seed: int, root: Path, execute: bool = True) -> tuple[dict, dict, dict]:
    rng = random.Random(seed)
    mark = f"XVMARK_{seed}_{os.getpid()}"
    seen: set[str] = set()
    stats = dict(generated=0, allow=0, executed=0, skipped=0, leaked=0, program_ran=0)
    findings: dict[str, list[str]] = {}
    leaks: dict[str, int] = {}
    work, home = fresh_sandbox(root)
    for s in probes() + [generate(rng) for _ in range(n)]:
        if s in seen:
            continue
        seen.add(s)
        stats["generated"] += 1
        if classify_command(s, work, {}).verdict != ALLOW:
            continue
        stats["allow"] += 1
        if not execute:
            continue
        if not safe_to_run(s):
            stats["skipped"] += 1
            continue
        for harden in ((False, True) if "git" in s else (False,)):
            before = snapshot(root)
            left = run_sandboxed(s, work, home, mark, harden)
            stats["executed"] += 1
            if left:
                stats["leaked"] += 1
                leaks[s] = len(left)
            after = snapshot(root)
            if before != after:
                changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
                findings[s] = sorted(os.path.relpath(k, root) for k in changed)[:4]
                stats["program_ran"] += "PWNED" in findings[s] or any(k.endswith("PWNED") for k in changed)
                work, home = fresh_sandbox(root)
                break
    return stats, findings, leaks


@unittest.skipUnless(os.environ.get("XARENCODE_SLOW") == "1", "slow: executes ALLOW-classified commands in a sandbox; set XARENCODE_SLOW=1")
class DifferentialTests(unittest.TestCase):
    def test_allowed_commands_do_not_touch_the_filesystem(self):
        n = int(os.environ.get("XARENCODE_DIFF_N", "300"))
        seed = int(os.environ.get("XARENCODE_DIFF_SEED", "1"))
        with tempfile.TemporaryDirectory() as tmp:
            stats, findings, leaks = run_differential(n, seed, Path(tmp) / "r")
        print(f"\ndifferential seed={seed}: {stats}, mutating ALLOW: {len(findings)}, leaked: {len(leaks)}")
        self.assertEqual(findings, {}, "ALLOW verdicts that changed the filesystem")
        self.assertEqual(leaks, {}, "ALLOW verdicts that left a process running")
        self.assertEqual(stats["program_ran"], 0)
        self.assertGreater(stats["executed"], 0)


class GeneratorTests(unittest.TestCase):
    def test_generated_strings_classify_without_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats, _, _ = run_differential(300, 7, Path(tmp) / "r", execute=False)
        self.assertGreater(stats["generated"], 200)

    def test_known_bad_families_are_never_allow(self):
        """The strings behind earlier holes (comments, braces, line continuation) must stay ASK or worse."""
        nl = "\n"
        known = [f"echo a #'{nl}touch pwned{nl}'", f"echo #'{nl}sleep 9 &{nl}'", "sort {-o,f} a", "find . {-fprint,f}",
                 f"find . -d\\{nl}elete", f"sort --\\{nl}output=f", "{rm,-rf,x}", "echo {1..3}"]
        with tempfile.TemporaryDirectory() as tmp:
            for c in known:
                self.assertNotEqual(classify_command(c, Path(tmp), {}).verdict, ALLOW, c)


if __name__ == "__main__":
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    with tempfile.TemporaryDirectory() as tmp:
        stats, findings, leaks = run_differential(n, seed, Path(tmp) / "r")
    print(f"seed={seed}", stats)
    print("MUTATING ALLOW:", len(findings))
    for s, ch in list(findings.items())[:40]:
        print("  ", repr(s), ch)
    print("LEAKED:", len(leaks))
    for s, k in list(leaks.items())[:20]:
        print("  ", repr(s), k)
