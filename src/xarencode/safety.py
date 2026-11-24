"""Command and path classification. A speed bump, not a sandbox: approval is the real gate."""
from __future__ import annotations

import fnmatch
import glob
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Verdict(Enum):
    ALLOW = 0
    ASK = 1
    ELEVATED = 2
    BLOCK = 3


@dataclass
class Decision:
    verdict: Verdict
    reason: str = ""
    approval_key: str | None = None
    tag: str = ""  # "downloads from the internet" / "deletes files": such commands are asked every time


ALLOW, ASK, ELEVATED, BLOCK = Verdict.ALLOW, Verdict.ASK, Verdict.ELEVATED, Verdict.BLOCK

BLOCKED_NAMES = {
    "dd", "mke2fs", "mkswap", "wipefs", "fdisk", "sfdisk", "cfdisk", "gdisk", "sgdisk", "parted",
    "shred", "blkdiscard", "badblocks", "hdparm", "cryptsetup", "efibootmgr", "grub-install", "flashrom",
}
ELEVATORS = {"sudo", "doas", "pkexec", "su", "run0"}
ELEVATOR_VALUE_FLAGS = {"-u", "-g", "-C", "-h", "-p", "-r", "-t", "-U", "-D", "-T", "--user", "--group"}
SHELLS = {"bash", "sh", "zsh", "dash"}
REPARSERS = SHELLS | {"eval", "xargs", "find", "su"}
PLAIN_ALLOW = {
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "which", "type", "file", "stat", "du", "df",
    "free", "uname", "whoami", "id", "hostname", "date", "uptime", "tree", "grep", "egrep", "fgrep", "rg",
    "sort", "uniq", "cut", "diff", "cmp", "basename", "dirname", "realpath", "readlink", "nproc", "lscpu",
    "ps", "pgrep", "ss", "true", "false", ":",
}
GIT_READ = {"status", "log", "diff", "show", "blame", "ls-files", "rev-parse", "describe", "shortlog"}
NO_MEMORY = {
    "rm", "mv", "find", "xargs", "eval", "bash", "sh", "zsh", "dash", "env", "chmod", "chown", "chgrp",
    "ln", "kill", "pkill", "killall", "truncate", "tee",
}
RAW_BLOCKS = [
    (re.compile(r"--no-preserve-root"), "rm --no-preserve-root"),
    (re.compile(r":\s*\(\s*\)\s*\{"), "fork bomb"),
    (re.compile(r"\b(?:curl|wget)\b[^|;&\n]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b"), "piping a download into a shell"),
    (re.compile(r"\bmkfs"), "mkfs"),
    (re.compile(r"\bwipefs"), "wipefs"),
    (re.compile(r"\bdd\b[^\n]*\bof="), "dd of="),
    (re.compile(r"\bsda\d*\b"), "references /dev/sda"),
    (re.compile(r"\b(?:nvme\d+n\d+|mmcblk\d+)(?:p\d+)?\b"), "references a block device"),
    (re.compile(r"\brm\b(?:[^;&|\n]*\s)?-[a-zA-Z]*(?:r[a-zA-Z]*f|f[a-zA-Z]*r)", re.I), "recursive forced rm"),
]
DEV_PATH = re.compile(r"(?<![\w.~-])/dev/[^\s;|&<>()\"'`$]*")
DEV_OK = re.compile(r"(?:null|zero|random|urandom|tty|stdin|stdout|stderr|fd/\d+|shm(?:/.*)?)")
SEPARATOR_CHARS = set(";|&()\n")
PROC_OK = {"cpuinfo", "meminfo", "version", "loadavg", "uptime"}
FORBIDDEN_READ = ("/dev", "/proc", "/sys")


def _rank(v: Verdict) -> int:
    return v.value


def _worst(decisions: list[Decision]) -> Decision:
    return max(decisions, key=lambda d: _rank(d.verdict))


def _real(p: str | Path) -> Path:
    return Path(os.path.realpath(p))


# Places that hold secrets, relative to the home folder (and absolute ones). Reading one asks, whether the command is a
# shell reader or read_file; it is never remembered. The folders .ssh and .gnupg are refused outright instead.
ASSISTANT_DIR = "." + "cl" + "aude"  # the other coding assistant's folder (settings, tokens); spelled in two parts on purpose
CREDENTIAL_HOME = (".aws", ".azure", ".config/gcloud", ".config/gh", ".config/git/credentials", ".git-credentials", ".netrc",
                   ".docker/config.json", ".kube", ".npmrc", ".pypirc", ".config/rclone", ".password-store",
                   ".local/share/keyrings", ".mozilla", ".bash_history", ".zsh_history", ASSISTANT_DIR)
CREDENTIAL_GLOBS = (".cargo/credentials*", ".config/chromium/*/Login Data*", ".config/google-chrome/*/Login Data*")
CREDENTIAL_ABS = ("/etc/shadow", "/etc/gshadow")
CREDENTIAL_ABS_GLOBS = ("/etc/sudoers*",)
CREDENTIAL_REASON = "credentials"


def credential_paths() -> list[Path]:
    home = _real(Path.home())
    out = [_real(home / x) for x in CREDENTIAL_HOME] + [Path(x) for x in CREDENTIAL_ABS]
    for pat in [glob.escape(str(home)) + "/" + g for g in CREDENTIAL_GLOBS] + list(CREDENTIAL_ABS_GLOBS):
        try:
            out += [_real(m) for m in glob.glob(pat)]
        except (OSError, ValueError):
            pass
    return out


def resolve_path(p: str, workdir: Path, *, write: bool, cfg: dict) -> tuple[Path, Verdict, str]:
    wd = _real(workdir)
    if "\x00" in p:
        return wd, BLOCK, "the path contains a NUL byte"
    full = _real(wd / os.path.expanduser(p))
    home = _real(Path.home())
    for bad in [Path(x) for x in FORBIDDEN_READ] + [home / ".ssh", home / ".gnupg"]:
        if full == bad or bad in full.parents:
            return full, BLOCK, f"{bad} is off limits"
    if not write:
        for bad in credential_paths():
            if full == bad or bad in full.parents:
                return full, ASK, f"{CREDENTIAL_REASON}: {bad} may hold secrets"
    inside = full == wd or wd in full.parents
    if write:
        if inside and ".git" in full.relative_to(wd).parts:
            return full, BLOCK, "writes inside .git are not allowed"
        if not inside:
            if cfg.get("safety", {}).get("write_outside_workdir"):
                return full, ASK, "outside the working directory"
            return full, BLOCK, "writes must stay inside the working directory"
        return full, ALLOW, ""
    return (full, ALLOW, "") if inside else (full, ASK, "outside the working directory")


def _tokenize(cmd: str) -> list[str]:
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=";<>|&()\n")
    lex.whitespace_split = True
    lex.whitespace = " \t\r"
    lex.commenters = ""
    return list(lex)


@dataclass
class _Scan:
    subs: list[str]       # inner text of each outermost $(...), <(...), >(...) and `...`
    angles: int           # '<' and '>' outside quotes
    suspicious: bool      # ANSI-C or locale quoting, unterminated or ambiguous quoting: the lexers may disagree
    masked: str           # the command with single-quoted text blanked out
    dq_escape: bool = False  # \$, \` or \<newline> inside double quotes: bash and shlex unquote it differently


def _backtick_end(cmd: str, i: int) -> int:
    j = i + 1
    while j < len(cmd) and cmd[j] != "`":
        j += 2 if cmd[j] == "\\" else 1
    return j


def _scan(cmd: str) -> _Scan:
    """A small bash-aware quote scanner. Fails closed: anything it cannot follow is marked suspicious.

    shlex and bash disagree about $'...' (bash lets \\' escape the quote), so any $' or $" outside single quotes is
    suspicious: text that looks quoted to one parser can be live code to the other.
    """
    n, i = len(cmd), 0
    subs: list[str] = []
    out: list[str] = []
    angles = 0
    suspicious = dq_escape = False
    stack: list[list] = [["cmd", 0, None]]  # [mode, open parens, start of the $( text or None]
    sub_levels = 0
    while i < n:
        c, nxt = cmd[i], cmd[i + 1:i + 2]
        mode = stack[-1][0]
        if c == "\\":
            if mode == "dq" and nxt in ("$", "`", "\n"):
                dq_escape = True
            out.append(cmd[i:i + 2])
            i += 2
            continue
        if mode == "cmd" and c == "'":
            j = cmd.find("'", i + 1)
            if j < 0:
                return _Scan(subs, angles, True, "".join(out), dq_escape)
            out.append("''")
            i = j + 1
            continue
        if mode == "dq" and c == '"':
            stack.pop()
        elif mode == "cmd" and c == '"':
            stack.append(["dq", 0, None])
        elif c == "$" and nxt in ("'", '"') and mode == "cmd":
            suspicious = True
        elif c in "$<>" and nxt == "(" and (c == "$" or mode == "cmd"):
            if c != "$":
                angles += 1
            stack.append(["cmd", 0, i + 2])
            sub_levels += 1
            out.append(cmd[i:i + 2])
            i += 2
            continue
        elif c == "`":
            j = _backtick_end(cmd, i)
            inner = cmd[i + 1:j]
            if j >= n or any(q in inner for q in "'\"\\"):
                suspicious = True  # unterminated, or quoting inside backticks that bash reads its own way
            angles += inner.count("<") + inner.count(">")
            if sub_levels == 0:
                subs.append(inner)
            i = j + 1
            out.append("``")
            continue
        elif mode == "cmd" and c == "(":
            stack[-1][1] += 1
        elif mode == "cmd" and c == ")":
            if stack[-1][2] is not None and stack[-1][1] == 0:
                start = stack.pop()[2]
                sub_levels -= 1
                if sub_levels == 0:
                    subs.append(cmd[start:i])
            elif stack[-1][1] > 0:
                stack[-1][1] -= 1
        elif mode == "cmd" and c in "<>":
            angles += 1
        out.append(c)
        i += 1
    if len(stack) > 1:
        suspicious = True
    return _Scan(subs, angles, suspicious, "".join(out), dq_escape)


PUNCT = set(";<>|&()\n")
_OP_RX = re.compile(r"&>>|&>|<<<|<<|<>|<&|>>|>&|>\||>|<|&&|\|\||\|&|\||&|;|\(|\)|\n|.", re.S)
_REDIRECTS = {"&>>", "&>", "<<<", "<<", "<>", "<&", ">>", ">&", ">|", ">", "<"}


def _split_segments(tokens: list[str]) -> tuple[list[list[str]], bool, bool, int]:
    """Returns (segments, unsafe_redirect, has_background, harmless_angle_count).

    A punctuation token such as ';>' or ')>' is split into its separator and redirect parts, so the separator still
    starts a new segment and the redirect is still judged. Anything that is not clearly harmless counts as unsafe.
    """
    segs: list[list[str]] = [[]]
    write = bg = False
    harmless = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not (t and set(t) <= PUNCT):
            segs[-1].append(t)
            i += 1
            continue
        ops = _OP_RX.findall(t)
        for k, op in enumerate(ops):
            if op not in _REDIRECTS:
                if op == "&":
                    bg = True
                segs.append([])
                continue
            if segs[-1] and segs[-1][-1].isdigit():
                segs[-1].pop()  # file descriptor number before the operator
            last = k == len(ops) - 1
            nxt = tokens[i + 1] if last and i + 1 < len(tokens) else ""
            target_ok = bool(nxt) and not set(nxt) <= PUNCT
            if op == "<":
                ok = target_ok
            elif op in (">&", "<&"):
                ok = target_ok and (nxt.isdigit() or nxt in ("-", "/dev/null"))
            elif op in ("&>", "&>>", ">", ">>", ">|"):
                ok = target_ok and nxt == "/dev/null"
            else:  # here-documents, here-strings and read/write opens
                ok = False
            if ok:
                harmless += op.count("<") + op.count(">")
            else:
                write = True
        i += 1
        if ops and ops[-1] in _REDIRECTS and i < len(tokens) and not set(tokens[i]) <= PUNCT:
            i += 1  # skip the redirect target
    return [s for s in segs if s], write, bg, harmless


def _strip_prefixes(tokens: list[str]) -> tuple[list[str], str | None, bool]:
    """Drop env assignments and wrappers. Returns (rest, elevator or None, saw_env_assignment)."""
    elevator = None
    env_set = False
    toks = list(tokens)
    while toks:
        t = toks[0]
        name = os.path.basename(t)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            env_set = True
            toks.pop(0)
        elif name in ELEVATORS:
            elevator = elevator or name
            toks.pop(0)
            if name == "su":
                break
            while toks and toks[0].startswith("-"):
                f = toks.pop(0)
                if f in ELEVATOR_VALUE_FLAGS and toks:
                    toks.pop(0)
        elif name in ("command", "nohup", "time", "exec", "stdbuf"):
            toks.pop(0)
            while toks and toks[0].startswith("-"):
                f = toks.pop(0)
                if name == "time" and f != "-p":
                    env_set = True  # GNU time -o FILE / -a / -f FMT write or format; keep it out of ALLOW
                if (name == "exec" and f == "-a") or (name == "stdbuf" and f in ("-i", "-o", "-e")):
                    if toks:
                        toks.pop(0)  # the process name / buffering mode that belongs to the flag
        elif name == "env":
            toks.pop(0)
            while toks and (toks[0].startswith("-") or re.match(r"^\w+=", toks[0])):
                env_set = env_set or toks[0] not in ("-i", "-", "--", "-0", "-v", "--ignore-environment", "--null", "--debug")
                toks.pop(0)
        elif name in ("nice", "ionice"):
            toks.pop(0)
            while toks and toks[0].startswith("-"):
                f = toks.pop(0)
                if f in ("-n", "-c") and toks:
                    toks.pop(0)
        elif name == "timeout":
            toks.pop(0)
            while toks and toks[0].startswith("-"):
                toks.pop(0)
            if toks:
                toks.pop(0)
        else:
            break
    return toks, elevator, env_set


def _positionals(args: list[str], value_flags: set[str] = frozenset()) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in value_flags:
            skip = True
        elif a == "--" or not a.startswith("-") or a == "-":
            out.append(a)
    return [a for a in out if a != "--"]


def _bad_target(target: str, wd: Path) -> bool:
    t = target.rstrip("/") or "/"
    if t in ("/", "~", "$HOME", "${HOME}", "..", ".", "*", "~/*", "/*") or t.startswith(("~", "$")):
        return True
    head, base = os.path.split(t)
    if base and set(base) <= set("*?[]!^.") and any(c in base for c in "*?["):
        # a bare glob ("*", ".*", "./*") matches the whole top level of its folder
        folder = _real(wd / os.path.expanduser(head or "."))
        if folder == wd or wd not in folder.parents:
            return True
    full = _real(wd / os.path.expanduser(t))
    return full == wd or wd not in full.parents


def _flags(args: list[str]) -> tuple[set[str], set[str]]:
    shorts, longs = set(), set()
    for a in args:
        if a == "--":
            break
        if a.startswith("--"):
            longs.add(a.split("=")[0])
        elif a.startswith("-") and len(a) > 1:
            shorts.update(a[1:])
    return shorts, longs


def matches_long(opt: str, *fulls: str) -> bool:
    """True if `opt` is a full long option, with or without =value, or any GNU-style abbreviation of one of `fulls`."""
    name = opt.split("=", 1)[0]
    return len(name) > 2 and name.startswith("--") and any(f.startswith(name) for f in fulls)


def _has_long(longs: set[str], *fulls: str) -> bool:
    return any(matches_long(o, *fulls) for o in longs)


def _git_allowed(args: list[str]) -> bool:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a in ("-C", "--git-dir", "--work-tree", "--namespace"):
            i += 2
        elif a in ("--no-pager", "--no-optional-locks", "-P", "--bare") or a.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
            i += 1
        else:
            return False  # -c, --config-env, --exec-path, -p/--paginate and anything unknown
    if i >= len(args):
        return False
    sub, rest = args[i], args[i + 1:]
    if any(a.startswith("--") and a != "--" and matches_long(a, "--output", "--ext-diff", "--open-files-in-pager", "--textconv", "--ignore-submodules",
                                              "--submodule", "--recurse-submodules", "--no-recurse-submodules")
           for a in rest):
        return False
    if sub in GIT_READ:
        return True
    if sub == "branch":
        safe = ("-a", "-r", "-v", "-vv", "--list", "--show-current", "--all", "--remotes", "-l")
        listing = "--list" in rest or "-l" in rest
        return all(a in safe or (listing and not a.startswith("-")) for a in rest)
    if sub == "tag":
        return not rest or (rest[0] in ("-l", "--list", "-n") and all(not a.startswith("-") or a.startswith("--sort=") for a in rest[1:]))
    if sub == "remote":
        return rest in ([], ["-v"], ["--verbose"]) or (rest[:1] == ["get-url"])
    return False


# --- the repository git will use: its own config must hold nothing that can run a program
# Local config keys that cannot make a read-only git command execute anything. Everything else (gpg.*, log.showSignature,
# *.textconv, *.command, filter.*, core.fsmonitor, core.hooksPath, core.pager, core.sshCommand, core.editor,
# diff.external, include*, credential.*, url.*.insteadOf, safe.*, alias.* ...) makes the command ASK.
GIT_SAFE_KEYS = re.compile(
    r"core\.(?:repositoryformatversion|filemode|bare|logallrefupdates|ignorecase|precomposeunicode|symlinks|autocrlf|eol|untrackedcache)"
    r"|extensions\.(?:objectformat|worktreeconfig|refstorage|compatobjectformat)"
    r"|remote\..+\.(?:url|fetch|pushurl)|branch\..+\.(?:remote|merge|rebase)|user\.(?:name|email)"
    r"|init\.defaultbranch|pull\.rebase|merge\.ff|color\..+", re.I)
GIT_RISKY_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_EXTERNAL_DIFF", "GIT_EXEC_PATH", "GIT_CONFIG",
                 "GIT_CONFIG_PARAMETERS", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_NAMESPACE")
_git_keys_cache: dict[tuple, frozenset | None] = {}


def _config_keys(path: Path) -> frozenset | None:
    """Every key set in one git config file (git itself parses it, nothing is executed); None when it cannot be read."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return frozenset()
    except OSError:
        return None
    ck = (str(path), st.st_mtime_ns, st.st_size)
    if ck in _git_keys_cache:
        return _git_keys_cache[ck]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(Path.home()), "LC_ALL": "C",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    try:
        r = subprocess.run(["git", "config", "--file", str(path), "--list", "--name-only", "-z"], capture_output=True,
                           stdin=subprocess.DEVNULL, timeout=5, env=env)
        keys = frozenset(k for k in r.stdout.decode("utf-8", "replace").split("\0") if k) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        keys = None
    if len(_git_keys_cache) > 512:
        _git_keys_cache.clear()
    _git_keys_cache[ck] = keys
    return keys


def _gitdir_of(p: Path) -> Path | None:
    """The git directory a `.git` entry or --git-dir value stands for: the directory itself or what a gitfile points to."""
    try:
        if p.is_dir():
            return _real(p)
        if p.is_file():
            with open(p, "rb") as f:
                first = f.read(4096).decode("utf-8", "replace").splitlines()[:1]
            if first and first[0].startswith("gitdir:"):
                target = _real(os.path.join(p.parent, first[0][7:].strip()))
                return target if target.is_dir() else None
    except OSError:
        pass
    return None


def _config_files(gitdir: Path) -> list[Path]:
    files = [gitdir / "config", gitdir / "config.worktree"]
    try:
        common = (gitdir / "commondir").read_text(errors="replace").strip()
    except OSError:
        common = ""
    if common:
        cd = _real(os.path.join(gitdir, common))
        files += [cd / "config", cd / "config.worktree"]
    return files


def _looks_bare(d: Path) -> bool:
    return (d / "HEAD").is_file() and (d / "objects").is_dir() and (d / "refs").is_dir()


def _inside(p: Path, wd: Path) -> bool:
    return p == wd or wd in p.parents


def _git_path(val: str, wd: Path, eff: Path) -> Path | None:
    """The folder a git path option names, as bash would pass it; None when it cannot be known (variables, globs)."""
    home = str(Path.home())
    val = val.replace("${HOME}", home).replace("$HOME", home).replace("${PWD}", str(wd)).replace("$PWD", str(wd))
    if "$" in val or any(c in val for c in "*?[`\n"):
        return None
    return _real(os.path.join(eff, os.path.expanduser(val)))


def _git_globals(args: list[str], wd: Path) -> tuple[Path | None, Path | None, Path | None, str]:
    """(effective directory, --git-dir or None, --work-tree or None, problem) for the global options before the subcommand."""
    eff, git_dir, work_tree, bare = wd, None, None, False
    raw: dict[str, str] = {}
    i = 0
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a in ("-C", "--git-dir", "--work-tree", "--namespace"):
            if i + 1 >= len(args):
                return None, None, None, "git option without a value"
            val, i = args[i + 1], i + 2
        elif a.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
            a, val = a.split("=", 1)
            i += 1
        else:
            bare = bare or a == "--bare"
            i += 1
            continue
        if a == "-C":
            eff = _git_path(val, wd, eff)
            if eff is None or not _inside(eff, wd):
                return None, None, None, "git -C points outside the working directory or cannot be checked"
        elif a in ("--git-dir", "--work-tree"):
            raw[a] = val
    for opt in ("--git-dir", "--work-tree"):
        if opt in raw:
            p = _git_path(raw[opt], wd, eff)
            if p is None or not _inside(p, wd):
                return None, None, None, f"git {opt} points outside the working directory or cannot be checked"
            if opt == "--git-dir":
                git_dir = p
            else:
                work_tree = p
    if bare and git_dir is None:
        git_dir = eff
    return eff, git_dir, work_tree, ""


def _gitdir_config_problem(g: Path) -> str | None:
    for f in _config_files(g):
        keys = _config_keys(f)
        if keys is None:
            return f"cannot parse {f}"
        bad = sorted(k for k in keys if not GIT_SAFE_KEYS.fullmatch(k))
        if bad:
            return f"the repository config sets {bad[0]}, which can run programs ({f})"
    return None


MAX_GITLINKS = 200


def nested_repos(eff: Path, git_dir: Path | None, work_tree: Path | None, depth: int = 0) -> list[tuple[Path, Path]] | None:
    """(work tree, git directory) of every committed nested repository (a gitlink, mode 160000, whose folder has its own
    .git) below the repository git will use, looked up through the index and followed down. None when the index cannot
    be listed in time. Such a repository is read with ITS OWN config when the outer status, diff or log asks about it."""
    cwd = work_tree or eff
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "XDG_CONFIG_HOME", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM")}
    env.update(LC_ALL="C", GIT_OPTIONAL_LOCKS="0")
    if git_dir is not None:
        env["GIT_DIR"] = str(git_dir)
    if work_tree is not None:
        env["GIT_WORK_TREE"] = str(work_tree)
    base = ["git", "-c", "core.fsmonitor=false"]
    try:
        if work_tree is not None:
            top = work_tree
        elif git_dir is not None:
            top = eff
        else:
            t = subprocess.run([*base, "rev-parse", "--show-toplevel"], cwd=cwd, env=env, capture_output=True, timeout=5,
                               stdin=subprocess.DEVNULL)
            if t.returncode != 0:
                return []  # no work tree (a bare repository) or no repository at all
            top = Path(t.stdout.decode("utf-8", "replace").strip())
        r = subprocess.run([*base, "ls-files", "-s", "-z", "--full-name"], cwd=top, env=env, capture_output=True, timeout=5,
                           stdin=subprocess.DEVNULL)  # from the top: the whole index, not just the folder git runs in
        if r.returncode != 0:
            return []  # no index: git fails by itself, there is nothing to recurse into
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    out: list[tuple[Path, Path]] = []
    for ent in r.stdout.split(b"\0"):
        head, _, path = ent.partition(b"\t")
        if not head.startswith(b"160000 "):
            continue
        folder = top / os.fsdecode(path)
        dot = folder / ".git"
        if not os.path.lexists(dot):
            continue
        g = _gitdir_of(dot)
        if g is None:
            return None
        out.append((_real(folder), g))
        if len(out) > MAX_GITLINKS or depth >= 3:
            return None
        deeper = nested_repos(folder, g, folder, depth + 1)
        if deeper is None:
            return None
        out += deeper
    return out if len(out) <= MAX_GITLINKS else None


def _git_repo_problem(args: list[str], wd: Path) -> str | None:
    """Why a git command must ask: the repository it will use (or a nested repository it will look into) has local
    config that is not on the safe list, or the repository cannot be determined. None when every key is harmless."""
    for var in GIT_RISKY_ENV:
        if os.environ.get(var):
            return f"{var} is set in the environment"
    eff, git_dir, work_tree, problem = _git_globals(args, wd)
    if problem:
        return problem
    cands: list[Path] = []
    if git_dir is not None:
        g = _gitdir_of(git_dir)
        if g is None:
            return "cannot read the git directory"
        cands.append(g)
    else:
        for d in [eff, *eff.parents]:
            dot = d / ".git"
            if os.path.lexists(dot):
                g = _gitdir_of(dot)
                if g is None:
                    return f"cannot read {dot}"
                cands.append(g)
            if _looks_bare(d):
                cands.append(d)
    for g in cands:
        why = _gitdir_config_problem(g)
        if why:
            return why
    if cands:
        nested = nested_repos(eff, git_dir, work_tree)
        if nested is None:
            return "cannot list the nested repositories (submodules) of this repository"
        for folder, g in nested:
            why = _gitdir_config_problem(g)
            if why:
                return f"nested repository {folder.name}: {why}"
    return None


def git_targets(command: str, wd: Path) -> list[tuple[Path, Path | None, Path | None]]:
    """(effective directory, --git-dir, --work-tree) of every git invocation in a command, for hardening the environment."""
    wd = _real(wd)
    out: list[tuple[Path, Path | None, Path | None]] = []
    try:
        segments, _, _, _ = _split_segments(_tokenize(command.replace("\\\n", "")))
    except ValueError:
        return out
    for seg in segments:
        toks, _, _ = _strip_prefixes(seg)
        if toks and os.path.basename(toks[0]) == "git":
            eff, git_dir, work_tree, problem = _git_globals(toks[1:], wd)
            if not problem and eff is not None:
                out.append((eff, git_dir, work_tree))
    return out


def _read_only(name: str, args: list[str], cfg: dict, line: str) -> bool:
    for prefix in cfg.get("safety", {}).get("allow", []):
        if line == prefix or line.startswith(prefix + " "):
            return True
    if args == ["--version"] and "/" not in name:
        return True
    shorts, longs = _flags(args)
    if name == "find":
        return not any(a.startswith(("-exec", "-ok", "-fprint", "-fls")) or a == "-delete" for a in args)
    if name == "git":
        return _git_allowed(args)
    if name == "pacman":
        return bool(args) and bool(re.fullmatch(r"-Q[a-zA-Z]*|-Si|-Ss|--query", args[0]))
    if name not in PLAIN_ALLOW:
        return False
    if name == "tail":
        return not (shorts & {"f", "F"} or _has_long(longs, "--follow"))
    if name == "sort":
        return "o" not in shorts and not _has_long(longs, "--output", "--compress-program")
    if name == "file":
        return not (shorts & {"C", "m"} or _has_long(longs, "--compile", "--magic-file"))
    if name == "uniq":
        return len(_positionals(args, {"-f", "-s", "-w"})) <= 1
    if name == "rg":
        return not any(a.startswith("--pre") or matches_long(a, "--pre", "--pre-glob") or a.split("=")[0] == "--hostname-bin"
                       for a in args)
    if name == "tree":
        return not (shorts & {"o", "R", "H"} or _has_long(longs, "--output", "--htmlbase"))
    if name == "date":
        return "s" not in shorts and not _has_long(longs, "--set")
    if name == "hostname":
        return not _positionals(args) and not (shorts & {"F", "b"} or _has_long(longs, "--file", "--boot"))
    if name == "ss":
        return not (shorts & {"K", "D"} or _has_long(longs, "--kill", "--diag"))
    return True


# "Don't ask again" is offered ONLY for these well-known developer commands, with the exact subcommand (and, for npm, make
# and cargo, the target) in the key. Everything else (git lfs, gh, docker, pacman, uvx, go run, sed -i ...) has no key: the
# menu offers yes / no / feedback only. Download and delete commands are cut off earlier (they carry a tag, see
# classify_command), and so is anything compound or with a redirect.
KEY_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9:_.-]*")
# script and make-target names that mean install, publish, delete or reset: never remembered
KEY_DENY = frozenset({"clean", "distclean", "mrproper", "install", "uninstall", "publish", "deploy", "release", "prune", "purge",
                      "reset", "nuke", "wipe", "destroy", "remove", "delete", "push", "upload", "update", "upgrade", "fetch",
                      "pull", "download", "uninstall", "dist-clean", "realclean", "cleanall"})
KEY_JS_RUNNERS = {"npm", "pnpm", "yarn"}
KEY_CARGO = frozenset({"build", "test", "check", "fmt", "clippy", "run"})
KEY_GO = frozenset({"build", "test", "vet", "fmt"})
KEY_TOOLS = frozenset({"tsc", "eslint", "prettier"})


def _key_target_ok(t: str) -> bool:
    return bool(KEY_TARGET.fullmatch(t)) and not (set(re.split(r"[:_.-]", t.lower())) & KEY_DENY) and t.lower() not in KEY_DENY


def _approval_key(name: str, args: list[str]) -> str | None:
    if any(a in ("-f", "--force", "--hard") or a.startswith("--force") for a in args):
        return None
    a0 = args[0] if args else ""
    if name in KEY_JS_RUNNERS:
        if a0 in ("test", "start"):
            return f"{name} {a0}"
        if a0 in ("run", "run-script") and len(args) > 1 and _key_target_ok(args[1]):
            return f"{name} run {args[1]}"
        return None
    if name == "vite":
        words = [a for a in args if not a.startswith("-")]
        return " ".join(["vite", *words]) if all(w in ("build", "dev", "preview", "serve") for w in words) and len(words) <= 1 else None
    if name == "pytest":
        return "pytest"
    if name.startswith("python"):
        if args[:2] in (["-m", "pytest"], ["-m", "unittest"], ["-m", "http.server"]):
            return f"{name} -m {args[1]}"
        if a0.endswith(".py") and not a0.startswith("-"):
            return f"{name} {a0}"
        return None
    if name == "node":
        return f"node {a0}" if a0.endswith((".js", ".mjs", ".cjs")) and not a0.startswith("-") else None
    if name == "cargo":
        return f"cargo {a0}" if a0 in KEY_CARGO else None
    if name == "go":
        return f"go {a0}" if a0 in KEY_GO else None
    if name == "make":
        targets = [a for a in args if not a.startswith("-")]
        if any(a.startswith("-") and not re.fullmatch(r"-(?:j\d*|k|s|w)", a) for a in args):
            return None
        if all(_key_target_ok(t) for t in targets):
            return " ".join(["make", *targets])
        return None
    if name in KEY_TOOLS:
        return name
    if name == "git":
        if a0 in ("add", "switch"):
            return f"git {a0}"
        if a0 == "commit":
            return None if "--amend" in args else "git commit"
        if a0 == "checkout":
            return "git checkout -b" if args[1:2] == ["-b"] else None
        if a0 == "stash":
            return "git stash" if len(args) == 1 or args[1] in ("push", "-u", "--include-untracked", "-m") else None
    return None


def _classify_segment(tokens: list[str], wd: Path, cfg: dict, depth: int, sudo_never: bool) -> tuple[Decision, bool]:
    toks, elevator, env_set = _strip_prefixes(tokens)
    if elevator and sudo_never:
        return Decision(BLOCK, "sudo is disabled (safety.sudo = never)"), True
    floor = Decision(ELEVATED, f"{elevator} needs your confirmation") if elevator else None
    if elevator == "su" or not toks:
        ds = [floor] if floor else [Decision(ASK, "variable assignment or empty command")]
        if elevator == "su" and "-c" in tokens:
            k = tokens.index("-c")
            if k + 1 < len(tokens):
                ds.append(classify_command(tokens[k + 1], wd, cfg, depth + 1))
        return _worst(ds), bool(elevator)
    raw_name = toks[0]
    name = os.path.basename(raw_name)
    args = toks[1:]
    ds: list[Decision] = []
    if env_set:  # GIT_EXTERNAL_DIFF, LD_PRELOAD, PAGER ... change what a "read-only" command executes
        ds.append(Decision(ASK, "sets environment variables for the command"))
    if "$" in raw_name:
        ds.append(Decision(ASK, "command name comes from a variable"))
    if name in BLOCKED_NAMES or name.startswith("mkfs"):
        ds.append(Decision(BLOCK, f"{name} is blocked"))
    elif name == "rm":
        shorts, longs = _flags(args)
        rec = bool(shorts & {"r", "R"}) or "--recursive" in longs
        force = "f" in shorts or "--force" in longs
        targets = _positionals(args)
        if rec and force:
            ds.append(Decision(BLOCK, "recursive forced rm is blocked"))
        elif rec and (not targets or any(_bad_target(t, wd) for t in targets)):
            ds.append(Decision(BLOCK, "recursive rm outside the working directory or on its root is blocked"))
        else:
            ds.append(Decision(ASK, "deletes files"))
    elif name in ("chmod", "chown", "chgrp"):
        shorts, longs = _flags(args)
        targets = _positionals(args)[1:]
        if ("R" in shorts or "--recursive" in longs) and any(_bad_target(t, wd) for t in targets):
            ds.append(Decision(BLOCK, f"recursive {name} outside the working directory is blocked"))
        else:
            ds.append(Decision(ASK, f"{name} changes permissions"))
    elif name in SHELLS and depth < 3:
        c_idx = next((i for i, a in enumerate(args) if re.fullmatch(r"-[a-z]*c[a-z]*", a)), None)
        if c_idx is not None and c_idx + 1 < len(args):
            ds.append(classify_command(args[c_idx + 1], wd, cfg, depth + 1))
        else:
            ds.append(Decision(ASK, "starts a shell"))
    elif name == "eval" and depth < 3:
        ds.append(_floor_ask(classify_command(" ".join(args), wd, cfg, depth + 1)))
    elif name == "find" and any(a in ("-exec", "-execdir", "-ok", "-okdir") for a in args) and depth < 3:
        for i, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                end = next((j for j in range(i + 1, len(args)) if args[j] in (";", "+")), len(args))
                ds.append(_floor_ask(classify_command(" ".join(shlex.quote(x) for x in args[i + 1:end]), wd, cfg, depth + 1)))
        ds.append(Decision(ASK, "find runs commands or deletes"))
    elif name == "xargs" and depth < 3:
        rest = [a for a in args if not a.startswith("-")]
        ds.append(_floor_ask(classify_command(" ".join(shlex.quote(x) for x in rest), wd, cfg, depth + 1)) if rest else Decision(ASK, "xargs"))
    elif depth >= 3 and name in SHELLS | {"eval", "xargs"}:
        ds.append(Decision(ASK, "deeply nested command"))
    elif _read_only(name, args, cfg, " ".join(toks)):
        why = _git_repo_problem(args, wd) if name == "git" else None
        # no approval key: "always" must not carry over to some other repository
        ds.append(Decision(ASK, why, None) if why else Decision(ALLOW, "read-only"))
    else:
        ds.append(Decision(ASK, "not on the read-only list", _approval_key(name, args)))
    if floor:
        ds.append(floor)
    return _worst(ds), bool(elevator)


def _brace_expand(s: str, limit: int = 64) -> list[str]:
    """Minimal bash brace expansion ({a,b}); returns the input unchanged when nothing expands."""
    out = [s]
    for _ in range(8):
        nxt, changed = [], False
        for item in out:
            m = re.search(r"\{([^{}]*,[^{}]*)\}", item)
            if m:
                changed = True
                nxt += [item[:m.start()] + alt + item[m.end():] for alt in m.group(1).split(",")]
            else:
                nxt.append(item)
        out = nxt
        if not changed or len(out) > limit:
            break
    return out[:limit]


def _comp_match(name: str, pat: str) -> bool:
    if name.startswith(".") and pat[:1] in ("*", "?"):
        return False  # a leading dot must be matched explicitly
    return fnmatch.fnmatchcase(name, pat)


GLOB_MATCH_LIMIT = 5000
GLOB_TIME_LIMIT = 1.0


def _glob_hits(piece: str, wd: Path, bad: list[Path]) -> Path | str | None:
    """A glob pattern that could expand to (or below) a protected location, also through symlinks.

    The pattern is first compared with the protected paths by name. Then it is expanded for real and every match is
    resolved with realpath, so `keys/*` or `key?/id_rsa` with `keys -> ~/.ssh` is caught. A glob that cannot be
    enumerated safely (too many matches, too slow, unreadable) comes back as a reason string, which means ASK.
    """
    pat = os.path.normpath(os.path.join(wd, os.path.expanduser(piece)))
    pc = [c for c in pat.split("/") if c]
    for b in bad:
        bc = [c for c in str(b).split("/") if c]
        if len(pc) >= len(bc) and all(_comp_match(bc[j], pc[j]) for j in range(len(bc))):
            return b
    deadline = time.monotonic() + GLOB_TIME_LIMIT
    try:
        for n, match in enumerate(glob.iglob(pat)):
            if n >= GLOB_MATCH_LIMIT or time.monotonic() > deadline:
                return "a glob with too many matches to check"
            real = _real(match)
            for b in bad:
                if real == b or b in real.parents or real in b.parents:
                    return b
    except (OSError, ValueError):
        return "a glob that cannot be expanded"
    return None


def _protected_paths() -> list[Path]:
    home = _real(Path.home())
    return [Path("/proc"), Path("/sys"), home / ".ssh", home / ".gnupg"] + credential_paths()  # /dev: the raw scan


def _touches_protected(tokens: list[str], wd: Path) -> str | None:
    """A token that names a protected location (same list read_file refuses), including globs that could reach it."""
    bad = _protected_paths()
    for t in tokens:
        for piece in {t, t.split("=", 1)[-1]}:
            if not piece or piece.startswith("-") and "=" not in piece:
                continue
            piece = piece.replace("${HOME}", "~").replace("$HOME", "~")
            for variant in _brace_expand(piece):
                if any(c in variant for c in "*?["):
                    hit = _glob_hits(variant, wd, bad)
                    if hit:
                        return str(hit)
                if variant.startswith("/proc/") or "/" in variant or variant.startswith("~") or variant.startswith("."):
                    full = _real(wd / os.path.expanduser(variant.split("*")[0].split("?")[0] or "."))
                    if full.parent == Path("/proc") and full.name in PROC_OK:
                        continue
                    for b in bad:
                        if full == b or b in full.parents:
                            return str(b)
    return None


RECURSIVE_READERS = {"grep", "egrep", "fgrep", "rg", "diff"}


def _recursive_root_covers_protected(segments: list[list[str]], wd: Path) -> str | None:
    """grep -r / rg / diff -r rooted at / or an ancestor of ~/.ssh would read protected files."""
    bad = _protected_paths()
    for seg in segments:
        toks, _, _ = _strip_prefixes(seg)
        if not toks:
            continue
        name, args = os.path.basename(toks[0]), toks[1:]
        if name not in RECURSIVE_READERS:
            continue
        shorts, longs = _flags(args)
        recursive = (name == "rg" or bool(shorts & {"r", "R"}) or _has_long(longs, "--recursive", "--dereference-recursive")
                     or any(a == "recurse" or a.endswith("=recurse") or a.startswith("-drecurse") for a in args))
        if not recursive:
            continue
        value_flags = {"-e", "-f", "-m", "-A", "-B", "-C", "-g", "-t", "-T", "--glob", "--include", "--exclude"}
        pos = _positionals(args, value_flags if name == "diff" else value_flags | {"-d", "--directories"})
        roots = pos or [str(wd)]
        if len(pos) <= 1 and name != "diff":
            roots = pos + [str(wd)]  # the first positional is the pattern; with no path the search starts in the cwd
        for r in roots:
            r = r.replace("${HOME}", "~").replace("$HOME", "~")
            full = _real(wd / os.path.expanduser(r))
            for b in bad:
                if full == b or full in b.parents or b in full.parents:
                    return str(b)
    return None


VAR_RX = re.compile(r"\$(?!HOME\b|\{HOME\}|PWD\b|\{PWD\}|\(|[?$#!@*0-9-])(?:[A-Za-z_]|\{)")


# Commands for which an unquoted glob is safe even when it expands to a file named like an option ("-delete", "--pre=sh"):
# a leading-dash filename cannot make any of them run a program or write a file. find, rg, file and tail are not here.
REV_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.")  # what may precede ~ in HEAD~1
GLOB_OK = {"ls", "cat", "wc", "head", "du", "stat", "grep", "diff"}
_REDIR_OK = re.compile(
    r">&[12](?![^\s;|&])"                         # 2>&1, >&2
    r"|(?:&>>?|>>?\|?)[ ]?/dev/null(?![^\s;|&])"  # >/dev/null, 2>/dev/null, &>/dev/null
    r"|<[ ]?(?=[A-Za-z0-9_./'\"-])")              # plain input: < file


def _simple_gate(cmd: str) -> str | None:
    """Strict, fail-closed gate for ALLOW: None when `cmd` is a plain chain of simple commands, else the reason.

    A hand-written scanner over the raw text (not shlex). It accepts only syntax on which bash and shlex agree: words,
    single-quoted spans, double-quoted spans without backslash, $, backtick or !, the separators | && || ; and a few
    harmless redirects. Comments, braces, backslashes, ANSI-C quoting, subshells, background jobs and so on are refused.
    """
    n, i = len(cmd), 0
    word_start = True
    first: str | None = None    # raw text of the first word of the current simple command
    in_first = False
    glob = False

    def end_segment(final: bool = False) -> str | None:
        nonlocal first, in_first, glob, word_start
        if first is None and not final:
            return "empty command"
        bad = glob and os.path.basename(first or "\0") not in GLOB_OK
        first, in_first, glob, word_start = None, False, False, True
        return "an unquoted glob in a command that is not a plain reader" if bad else None

    def add(text: str) -> None:
        nonlocal first, in_first
        if first is None:
            first, in_first = "", True
        if in_first:
            first += text

    while i < n:
        c = cmd[i]
        nxt = cmd[i + 1:i + 2]
        if c == "'":
            j = cmd.find("'", i + 1)
            if j < 0:
                return "unterminated quote"
            add(cmd[i:j + 1])
            word_start = False
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            while j < n and cmd[j] != '"':
                ch = cmd[j]
                if ch in "\\`!" or ord(ch) < 32 or ord(ch) == 127:
                    return "double quotes with a backslash, backtick, ! or control character"
                if ch == "$":
                    m = re.match(r"\$(?:HOME|PWD)(?![A-Za-z0-9_])", cmd[j:])
                    if not m:
                        return "variable expansion"
                    j += m.end()
                    continue
                j += 1
            if j >= n:
                return "unterminated quote"
            add(cmd[i:j + 1])
            word_start = False
            i = j + 1
            continue
        if c == " ":
            in_first = False
            word_start = True
            i += 1
            continue
        if c == ";":
            err = end_segment()
            if err:
                return err
            i += 1
            continue
        if c == "|":
            if nxt == "&":
                return "|& pipe"
            err = end_segment()
            if err:
                return err
            i += 2 if nxt == "|" else 1
            continue
        if c == "&":
            if nxt == "&":
                err = end_segment()
                if err:
                    return err
                i += 2
                continue
            m = _REDIR_OK.match(cmd, i)
            if not m:
                return "background job or redirect"
            if first is None:
                first = "\0"
            i, word_start = m.end(), True
            continue
        if c in "<>":
            m = _REDIR_OK.match(cmd, i)
            if not m:
                return "redirect that is not clearly harmless"
            if first is None:
                first = "\0"
            i, word_start = m.end(), True
            continue
        if c in "(){}`\\!#":
            return f"unsupported syntax {c!r}"
        if ord(c) < 32 or ord(c) == 127:
            return "control character"
        if c == "$":
            m = re.match(r"\$(?:HOME|PWD)(?![A-Za-z0-9_])|\$\?", cmd[i:])
            if not m:
                return "variable expansion or $-quoting"
            add(m.group())
            i += m.end()
            word_start = False
            continue
        if c == "~":
            if word_start:
                if nxt not in ("", "/", " ", ";", "|", "&"):
                    return "tilde expansion"
            elif not (cmd[i - 1] in REV_CHARS and (nxt.isdigit() or nxt in ("", " ", ";", "|", "&"))):
                return "tilde expansion"  # mid-word, bash does not expand it; only HEAD~1 style revisions are accepted
        elif c in "*?[":
            glob = True
        add(c)
        word_start = False
        i += 1
    return end_segment(True)


# --- download and delete commands: always asked in normal mode, never remembered (see risk_tags / classify_command)
YOLO_NOTICE = "YOLO mode: every approval is answered yes. Hard blocks (dd, mkfs, block devices, rm -rf) and sudo still apply."
DOWNLOAD_TAG = "downloads from the internet"
DELETE_TAG = "deletes files"
EDIT_TAG = "edits files in place"
OVERWRITE_TAG = "overwrites a file"
DOWNLOAD_ALWAYS = frozenset({"curl", "wget", "aria2c", "ssh", "sftp", "huggingface-cli", "hf", "npx", "bunx", "uvx", "yt-dlp",
                             "youtube-dl", "gallery-dl", "rclone", "nc", "ncat", "netcat", "socat", "telnet", "ftp", "lftp", "http",
                             "https", "xh", "axel"})
DELETE_ALWAYS = frozenset({"rm", "rmdir", "unlink", "shred", "truncate", "nix-collect-garbage", "trash-put", "trash", "trash-empty",
                           "trash-rm", "rmtrash", "srm"})
# command -> subcommands that fetch packages or code; "*" means any use (the AUR helpers sync and install by default)
DOWNLOAD_SUBS = {
    "git": frozenset({"clone", "fetch", "pull"}),
    "pip": frozenset({"install", "download"}), "pip3": frozenset({"install", "download"}),
    "pipx": frozenset({"install", "upgrade", "run"}), "uv": frozenset({"add", "sync", "install", "pip", "tool", "run"}),
    "npm": frozenset({"install", "i", "add", "ci", "update", "upgrade", "exec"}),
    "pnpm": frozenset({"install", "i", "add", "update", "up", "dlx", "exec"}),
    "yarn": frozenset({"install", "add", "up", "upgrade", "dlx"}), "bun": frozenset({"install", "i", "add", "update"}),
    "cargo": frozenset({"install", "add", "fetch", "update"}), "go": frozenset({"get", "install", "run"}),
    "deno": frozenset({"run", "install", "add", "cache", "upgrade", "compile"}),
    "svn": frozenset({"checkout", "co", "update", "up", "export", "switch", "sw"}),
    "flatpak": frozenset({"install", "update"}), "docker": frozenset({"pull"}), "podman": frozenset({"pull"}),
    "yay": frozenset({"*"}), "paru": frozenset({"*"}), "xarencode": frozenset({"pull"}),
}
REMOTE_COPY = frozenset({"scp", "rsync"})
# Any command that is not a plain reader and has one of these words among its first three non-flag arguments is tagged:
# `gh repo clone`, `apt -y install`, `docker compose up`, `make clean`, `npm uninstall`, ... Over-tagging only costs a
# prompt every time (a tagged command has no "don't ask again" key); a missed tag would let "always" cover a delete.
GENERIC_DOWNLOAD = frozenset({"install", "i", "add", "get", "clone", "download", "fetch", "pull", "update", "upgrade"})
GENERIC_DELETE = frozenset({"remove", "rm", "delete", "del", "uninstall", "purge", "prune", "clean", "drop", "destroy", "kill"})
# commands whose arguments are file names or free text, so a word like "update" there is not a subcommand
GENERIC_SKIP = PLAIN_ALLOW | frozenset({
    "git", "find", "pacman", "cp", "mv", "ln", "mkdir", "touch", "chmod", "chown", "chgrp", "tee", "sed", "awk", "cd",
    "install", "less", "more", "man", "vim", "nvim", "nano", "code", "open", "xdg-open", "sleep", "read", "test", "[",
    "export", "source", "alias", "set", "unset", "help", "history", "pushd", "popd", "diff3", "patch", "zip", "unzip", "tar"})
_PKG_DOWNLOAD = frozenset({"reinstall", "dist-upgrade", "full-upgrade", "refresh", "source", "build-dep", "sync", "in", "up", "dup",
                           "shell", "run", "develop", "build", "profile"})
_PKG_DELETE = frozenset({"autoremove", "autoclean", "erase", "cleanup", "gc"})
# command -> extra (download words, delete words) on top of the generic ones
EXTRA_SUBS: dict[str, tuple[frozenset, frozenset]] = {
    **{n: (_PKG_DOWNLOAD, _PKG_DELETE) for n in ("apt", "apt-get", "aptitude", "dnf", "yum", "zypper", "brew", "snap", "nix",
                                                  "port", "apk", "emerge", "pkg")},
    **{n: (frozenset({"run", "create", "build", "up", "load"}), frozenset({"down", "rmi", "stop-all"}))
       for n in ("docker", "podman", "docker-compose", "nerdctl")},
    "gh": (frozenset({"fork", "sync", "checkout", "co"}), frozenset()),
    **{n: (frozenset(), frozenset({"un", "unlink", "r"})) for n in ("npm", "yarn", "pnpm", "bun")},
}
AUR_HELPERS = frozenset({"yay", "paru", "pikaur", "trizen", "aura"})
_PACMAN_LONG = {"--sync": "S", "--upgrade": "U", "--remove": "R", "--query": "Q", "--files": "F", "--search": "s", "--info": "i",
                "--list": "l", "--groups": "g", "--print": "p", "--refresh": "y", "--sysupgrade": "u", "--downloadonly": "w",
                "--clean": "c"}


def _git_sub(args: list[str]) -> tuple[str, list[str]]:
    """The git subcommand and its arguments, skipping global options such as -C DIR and -c K=V."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env") else 1
    return (args[i], args[i + 1:]) if i < len(args) else ("", [])


def _pacman_risks(name: str, args: list[str]) -> set[str]:
    """pacman-style operations: -S/-U/-F -y download, -R and -Sc delete; -Q, -Si, -Ss, -Sl and the like are reads."""
    ops: set[str] = set()
    for a in args:
        if a == "--":
            break
        if a.startswith("--"):
            ops.update(_PACMAN_LONG.get(a.split("=")[0], ""))
        elif re.fullmatch(r"-[A-Za-z]+", a):
            ops.update(a[1:])
    out: set[str] = set()
    if "U" in ops:
        out.add("download")
    if "R" in ops:
        out.add("delete")
    if "S" in ops:
        if not (ops & set("silgp")) or ops & set("yuwc"):
            out.add("download")
        if "c" in ops:
            out.add("delete")
    if "F" in ops and "y" in ops:
        out.add("download")
    if not ops & set("SURQFDTVh") and name in AUR_HELPERS:
        out.add("download")  # `yay foo` searches and installs
    return out


def _git_risks(args: list[str]) -> set[str]:
    sub, rest = _git_sub(args)
    out: set[str] = set()
    flags = {a for a in rest if a.startswith("-")}
    short = {c for a in rest if re.fullmatch(r"-[A-Za-z]+", a) for c in a[1:]}
    if sub in ("clone", "fetch", "pull") or (sub == "submodule" and ("update" in rest or rest[:1] == ["add"])):
        out.add("download")
    if sub == "lfs" and rest[:1] in (["pull"], ["fetch"], ["clone"], ["checkout"]):
        out.add("download")
    if sub == "annex" and rest[:1] in (["get"], ["sync"], ["copy"], ["import"], ["addurl"], ["fetch"]):
        out.add("download")
    if (sub == "lfs" and rest[:1] == ["prune"]) or (sub == "annex" and rest[:1] in (["drop"], ["dropunused"], ["unused"])):
        out.add("delete")
    if sub == "prune" or (sub == "gc" and any(a.startswith("--prune") for a in rest)):
        out.add("delete")
    if sub == "submodule" and rest[:1] in (["foreach"], ["--recursive", "foreach"]):
        inner = [a for a in rest[1:] if a != "foreach" and a != "--recursive"]
        if inner:
            out |= risk_kinds(" ".join(inner), 2)
    if sub in ("clean", "rm", "restore") and not ({"-n", "--dry-run"} & flags):
        out.add("delete")
    if sub == "reset" and "--hard" in flags:
        out.add("delete")
    if sub == "stash" and rest[:1] in (["drop"], ["clear"]):
        out.add("delete")
    if sub == "checkout" and ("--" in rest or "." in rest or {"--force", "--ours", "--theirs"} & flags or "f" in short):
        out.add("delete")
    if sub == "switch" and ({"--discard-changes", "--force"} & flags or "f" in short):
        out.add("delete")
    if sub == "worktree" and rest[:1] in (["remove"], ["prune"]):
        out.add("delete")
    if sub == "reflog" and rest[:1] in (["expire"], ["delete"]):
        out.add("delete")
    if sub == "remote" and rest[:1] in (["remove"], ["rm"], ["prune"]):
        out.add("delete")
    if sub == "submodule" and rest[:1] == ["deinit"]:
        out.add("delete")
    if sub == "update-ref" and ("-d" in flags or "--delete" in flags):
        out.add("delete")
    if sub in ("branch", "tag") and ({"--delete"} & flags or short & {"d", "D"}):
        out.add("delete")
    if sub == "push" and ({"--delete", "--prune"} & flags or "d" in short or any(a.startswith(":") for a in rest)):
        out.add("delete")
    return out


def _gh_api_risks(args: list[str]) -> set[str]:
    """`gh api` is a read unless it names a method other than GET or sends fields (which makes it a POST)."""
    out: set[str] = set()
    for i, a in enumerate(args):
        method = None
        if a in ("-X", "--method") and i + 1 < len(args):
            method = args[i + 1]
        elif a.startswith("--method="):
            method = a.split("=", 1)[1]
        elif a.startswith("-X") and len(a) > 2:
            method = a[2:]
        if method and method.upper() != "GET":
            out.add("delete" if method.upper() == "DELETE" else "download")
        if a in ("-f", "-F", "--field", "--raw-field", "--input") or a.startswith(("--field=", "--raw-field=", "--input=")):
            out.add("download")
    return out


def _seg_risks(toks: list[str], generic: bool = True) -> set[str]:
    toks, _, _ = _strip_prefixes(toks)
    if not toks:
        return set()
    name, args = os.path.basename(toks[0]), toks[1:]
    out: set[str] = set()
    pos = [a for a in args if not a.startswith("-")]
    if name in DOWNLOAD_ALWAYS:
        out.add("download")
    if name in REMOTE_COPY and any(":" in a or "@" in a for a in args):
        out.add("download")  # a remote host on either side
    if name in DELETE_ALWAYS:
        out.add("delete")
    if name.startswith("python") and "-m" in args[:-1]:
        out |= _seg_risks(args[args.index("-m") + 1:])  # python -m pip install ...
    if name == "busybox" and pos:
        out |= _seg_risks(args[args.index(pos[0]):])  # busybox wget / busybox rm
    subs = DOWNLOAD_SUBS.get(name)
    if name == "git":
        out |= _git_risks(args)
    elif name in ("pacman", "pikaur") or name in AUR_HELPERS:
        out |= _pacman_risks(name, args)
    elif name == "dpkg" and any(a in ("-i", "--install", "-r", "--remove", "-P", "--purge") for a in args):
        out.add("delete" if any(a in ("-r", "--remove", "-P", "--purge") for a in args) else "download")
    elif name == "nix-env" and any(re.fullmatch(r"-[A-Za-z]*[ieu][A-Za-z]*|--(?:install|uninstall|upgrade)", a) for a in args):
        out |= {"download", "delete"}
    elif name == "gh" and pos[:1] == ["api"]:
        out |= _gh_api_risks(args)
    elif subs and ("*" in subs and not (args and re.fullmatch(r"-[A-Za-z]*[QFVhk][A-Za-z]*|--(?:query|files|version|help)", args[0])) or
                   any(a in subs for a in pos[:2]) or (name == "yarn" and not pos)):
        out.add("download")
    if name == "find" and "-delete" in args:
        out.add("delete")
    if name == "rsync" and any(a.startswith("--del") or a.startswith("--remove-source") for a in args):
        out.add("delete")
    if name == "rclone" and pos[:1] in (["sync"], ["delete"], ["purge"], ["deletefile"], ["move"], ["moveto"], ["cleanup"], ["dedupe"]):
        out.add("delete")
    if name == "cp" and "/dev/null" in args:
        out.add("delete")  # truncates the destination
    if name == "gio" and pos[:1] in (["trash"], ["remove"]):
        out.add("delete")
    if name in ("tar", "bsdtar") and any(a.startswith("--remove-files") for a in args):
        out.add("delete")
    if name == "zip" and ("m" in _flags(args)[0] or "--move" in args):
        out.add("delete")
    if (name == "sed" and any(a == "--in-place" or a.startswith("--in-place=") or re.fullmatch(r"-[A-Za-z]*i(?:\..*)?", a) for a in args)) \
            or (name in ("perl", "ruby") and any(re.fullmatch(r"-[A-Za-z]*i.*", a) for a in args if a.startswith("-") and not a.startswith("--"))) \
            or (name in ("awk", "gawk") and any(a == "inplace" for a in args)):
        out.add("edit")
    if name == "mv" and args and args[-1] == "/dev/null":
        out.add("delete")
    if generic and name not in GENERIC_SKIP and name != "git":
        words = set(pos[:3])
        dl_extra, del_extra = EXTRA_SUBS.get(name, (frozenset(), frozenset()))
        if words & (GENERIC_DOWNLOAD | dl_extra):
            out.add("download")
        if words & (GENERIC_DELETE | del_extra):
            out.add("delete")
    return out


def risk_kinds(cmd: str, depth: int = 0) -> set[str]:
    """Subset of {"download", "delete"} that `cmd` may do, looking inside shells, eval, xargs, find -exec and $(...)."""
    if depth > 4:
        return {"download", "delete"}
    cmd = cmd.replace("\\\n", "")
    try:
        tokens = _tokenize(cmd)
        segments, _, _, _ = _split_segments(tokens)
    except ValueError:  # cannot parse: judge by the words that appear
        words = set(re.findall(r"[A-Za-z0-9_.+-]+", cmd))
        return ({"download"} if words & (DOWNLOAD_ALWAYS | REMOTE_COPY | {"pull", "clone", "fetch"}) else set()) | \
               ({"delete"} if words & (DELETE_ALWAYS | {"clean", "-delete"}) else set())
    kinds: set[str] = {"overwrite"} if _overwrites(tokens) else set()
    for inner in _scan(cmd).subs:
        kinds |= risk_kinds(inner, depth + 1)
    for seg in segments:
        kinds |= _seg_risks(seg)
        toks, _, _ = _strip_prefixes(seg)
        if not toks:
            continue
        name, args = os.path.basename(toks[0]), toks[1:]
        if name in SHELLS | {"su", "sudo", "doas", "pkexec", "run0"}:
            for a in args:
                kinds |= risk_kinds(a, depth + 1) if " " in a else set()
        if name == "eval":
            kinds |= risk_kinds(" ".join(args), depth + 1)
        if name in ("xargs", "find", "parallel", "watch", "setsid", "flock"):
            for i in range(len(args)):  # whichever word starts the wrapped command
                kinds |= _seg_risks(args[i:], generic=False)
            if name == "find":  # only the words after -exec and friends are commands (not `-name clean`)
                for i, a in enumerate(args):
                    if a in ("-exec", "-execdir", "-ok", "-okdir"):
                        kinds |= _seg_risks(args[i + 1:])
            else:
                for i, a in enumerate(args):
                    if not a.startswith("-"):
                        kinds |= _seg_risks(args[i:])
    return kinds


def risk_tag(kinds: set[str]) -> str:
    return " and ".join(t for k, t in (("download", DOWNLOAD_TAG), ("delete", DELETE_TAG), ("edit", EDIT_TAG),
                                       ("overwrite", OVERWRITE_TAG)) if k in kinds)


def _overwrites(tokens: list[str]) -> bool:
    """True when the command truncates a file through a redirect (`> f`, `: > f`, `>| f`, `&> f`); `>> f`, `2>&1` and
    `> /dev/null` do not count."""
    for i, t in enumerate(tokens):
        if not (t and set(t) <= PUNCT):
            continue
        ops = _OP_RX.findall(t)
        for k, op in enumerate(ops):
            if op in (">", ">|", "&>"):
                nxt = tokens[i + 1] if k == len(ops) - 1 and i + 1 < len(tokens) else ""
                if nxt != "/dev/null":
                    return True
    return False


def _floor_ask(d: Decision) -> Decision:
    return d if _rank(d.verdict) >= _rank(ASK) else Decision(ASK, "runs another command")


def _expand_words(cmd: str) -> str:
    """Conservative brace expansion of the raw text: every word with {a,b} becomes the space-separated list bash makes."""
    return " ".join(" ".join(_brace_expand(w)) if "{" in w else w for w in cmd.split(" "))


def _classify_command(cmd: str, workdir: Path, cfg: dict, depth: int = 0) -> Decision:
    wd = _real(workdir)
    if depth > 3:
        return Decision(ASK, "deeply nested command")
    gate = _simple_gate(cmd)  # judged on the raw text; everything below works on the normalised copy
    cmd = cmd.replace("\\\n", "")  # bash drops backslash-newline outside single quotes; shlex keeps the newline
    if "{" in cmd:
        expanded = _expand_words(cmd)
        if expanded != cmd:
            d = classify_command(expanded, wd, cfg, depth + 1)
            if d.verdict in (BLOCK, ELEVATED):
                return d
    for m in DEV_PATH.finditer(cmd):
        if not DEV_OK.fullmatch(m.group()[5:]):
            return Decision(BLOCK, f"device path {m.group()} is blocked")
    for rx, why in RAW_BLOCKS:
        if rx.search(cmd):
            return Decision(BLOCK, f"blocked: {why}")
    try:
        tokens = _tokenize(cmd)
    except ValueError:
        return Decision(ASK, "could not parse the command")
    sudo_never = cfg.get("safety", {}).get("sudo") == "never"
    segments, write, bg, harmless_angles = _split_segments(tokens)
    scan = _scan(cmd)
    subs = scan.subs
    # a quoted string handed to a shell, eval, xargs, find -exec or su -c is parsed a second time, by bash itself
    unreliable = scan.suspicious or (scan.dq_escape and any(os.path.basename(t) in REPARSERS for t in tokens))
    decisions: list[Decision] = []
    for inner in subs:
        decisions.append(_floor_ask(classify_command(inner, wd, cfg, depth + 1)))
    elevated_any = False
    for seg in segments:
        d, el = _classify_segment(seg, wd, cfg, depth, sudo_never)
        elevated_any |= el
        decisions.append(d)
    if not decisions:
        return Decision(ASK, "empty command")
    result = _worst(decisions)
    if result.verdict == ALLOW:
        if gate:
            return Decision(ASK, f"not a plain simple command ({gate})")
        protected = _touches_protected(tokens, wd) or _recursive_root_covers_protected(segments, wd)
        if protected:
            return Decision(ASK, f"touches {protected}, which read_file refuses or asks about" if protected.startswith("/") else protected)
        if write or scan.angles > harmless_angles:
            return Decision(ASK, "uses a redirect that is not clearly harmless")
        if VAR_RX.search(scan.masked):
            return Decision(ASK, "expands a variable that could point anywhere")
        if bg:
            return Decision(ASK, "runs in the background")
        if unreliable:
            return Decision(ASK, "uses quoting ($'...', $\"...\" or odd backticks) that cannot be checked reliably")
        return result
    if result.verdict == ASK:
        single = len(segments) == 1 and not subs and not write and not bg and not unreliable and gate is None
        if not single:
            return Decision(ASK, result.reason)
    return result


def classify_command(cmd: str, workdir: Path, cfg: dict, depth: int = 0) -> Decision:
    """The verdict for a shell command. Download and delete commands are never ALLOW (not even through a
    safety.allow prefix), carry a tag, and have no approval key, so no "don't ask again" can ever cover them.
    A command that cannot be checked (a NUL byte, a path the OS rejects) asks."""
    if "\x00" in cmd:
        return Decision(ASK, "the command contains a NUL byte")
    try:
        return _classify_with_risks(cmd, workdir, cfg, depth)
    except (ValueError, OSError, RecursionError) as e:
        if depth:
            raise
        return Decision(ASK, f"could not be checked ({type(e).__name__})")


def _classify_with_risks(cmd: str, workdir: Path, cfg: dict, depth: int) -> Decision:
    d = _classify_command(cmd, workdir, cfg, depth)
    if depth or d.verdict == BLOCK:
        return d
    kinds = risk_kinds(cmd)
    if not kinds:
        return d
    tag = risk_tag(kinds)
    if d.verdict == ALLOW:
        return Decision(ASK, tag, None, tag)
    return Decision(d.verdict, d.reason, None, tag)
