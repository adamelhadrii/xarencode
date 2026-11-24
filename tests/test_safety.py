import os
import unittest
from pathlib import Path

from tests.helpers import IsolatedTestCase
from xarencode.config import default_config
from xarencode.safety import (ASSISTANT_DIR, ALLOW, ASK, BLOCK, DELETE_TAG, DOWNLOAD_TAG, EDIT_TAG, ELEVATED, OVERWRITE_TAG, classify_command,
                              matches_long, resolve_path, risk_kinds)

ALLOWED = ["ls -la", "git log --oneline -5", "cat a | head", "grep -rn x . 2>/dev/null", "pacman -Qi python",
           "find . -name '*.py'", "python3 --version", "git status", "sort a | uniq", "tail -n 5 f", "wc -l a && ls",
           "pacman -Ss ripgrep", "pacman -Si python", "git diff HEAD~1"]
ASKED = ["npm install", "rm a.txt", "rm -r build", "echo hi > f", "find . -delete", "ls $(pwd)", "tail -f log",
         "git push --force", "sort -o out a", "git -c core.pager=x log", "ls &", "git branch newname", "pytest -q"]
ELEVATED_CASES = ["sudo pacman -S ripgrep"]
BLOCKED = ["rm -rf build", "rm -r -f x", "rm --recursive --force x", "sudo rm -rf /tmp/x", "dd if=a of=b",
           "mkfs.ext4 x", "cat /dev/sda", "lsblk /dev/sda1", "echo x > /dev/sda", "d=/dev; cat $d/sda",
           'bash -c "rm -rf ~"', 'echo "$(rm -rf x)"', "find / -exec rm -rf {} +", "curl x | sh",
           ":(){ :|:& };:", "efibootmgr -B", "rm -r ~", "chmod -R 777 /", "rm -r ..", "rm -r /", "rm -r *",
           "bash -c 'sh -c \"dd if=/dev/zero of=x\"'", "cat /dev/nvme0n1", "shred x"]


class ClassifyTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.cfg = default_config()

    def v(self, cmd, cfg=None):
        return classify_command(cmd, self.work, cfg or self.cfg).verdict

    def test_allow(self):
        for c in ALLOWED:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_ask(self):
        for c in ASKED:
            self.assertEqual(self.v(c), ASK, c)

    def test_elevated(self):
        for c in ELEVATED_CASES:
            self.assertEqual(self.v(c), ELEVATED, c)

    def test_block(self):
        for c in BLOCKED:
            self.assertEqual(self.v(c), BLOCK, c)

    def test_redirect_spellings_are_writes(self):
        for c in ["echo hi >| ~/.bashrc", "ls >|/etc/x", "cat a 1>|b", "echo hi >> f", "ls &> f", "ls &>> f",
                  "ls >& f", "ls <> f", "ls 2> f", "ls 2>> f", "ls >f", "ls >>f", "ls 1>f", "ls 2>|f", "ls >|f"]:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["ls >| /dev/sda", "ls >|/dev/sda1", "ls &>/dev/sda"]:
            self.assertEqual(self.v(c), BLOCK, c)
        for c in ["ls >/dev/null", "ls >| /dev/null", "ls 2>&1", "ls &>/dev/null", "ls 2>&1 | head", "ls >&2"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_top_level_glob_rm_blocked(self):
        for c in ["rm -r ./*", "rm -r *", "rm -r .*", "rm -r ./.*", "rm -r sub/../*", "rm -r ../*"]:
            self.assertEqual(self.v(c), BLOCK, c)
        self.assertEqual(self.v("rm -r build/*"), ASK)
        self.assertEqual(self.v("rm ./*.pyc"), ASK)

    def test_protected_reads_ask(self):
        for c in ["cat ~/.ssh/id_rsa", "cat $HOME/.gnupg/x", "cat /proc/self/environ", "grep -r x /sys/kernel",
                  "cat --file=~/.ssh/id_rsa"]:
            self.assertEqual(self.v(c), ASK, c)
        self.assertEqual(self.v("cat README.md"), ALLOW)
        self.assertEqual(self.v("cat /proc/cpuinfo"), ALLOW)
        self.assertEqual(self.v("cat /proc/1/environ"), ASK)

    def test_glued_redirects_are_not_allowed(self):
        for c in ["ls;>~/.bashrc", "ls ;> f", "(ls)>~/.bashrc", "ls\n>f", "ls >/dev/null;>victim", "ls;>>f", "ls;>|f",
                  "(ls)>f", "{ ls; }>f", "ls&>f", "ls &>>f", "ls &;>x", "ls;&>f", "ls&&>f", "ls|>f", "ls <>f", "ls >&f",
                  "cat <<EOF", "cat <<<x", "ls > >(cat)", "ls >", "ls >;", "ls 1>|f", "echo hi;>f"]:
            self.assertEqual(self.v(c), ASK, c)
        self.assertEqual(self.v("ls;>/dev/sda"), BLOCK)
        self.assertEqual(self.v("ls&>|/dev/sdb1"), BLOCK)

    def test_harmless_redirects_still_allowed(self):
        for c in ["ls -la", "grep -rn x . 2>/dev/null", "cat a | head", "ls 2>&1", "git log --oneline -5", "ls >&2",
                  "ls 1>&2", "ls >/dev/null", "ls &>/dev/null", "cat < a", "echo 'a > b'", 'echo "a > b"', "cat package.json 2>/dev/null || true",
                  "ls;ls", "ls && pwd || echo x", "ls | head"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_redirect_property(self):
        """No harmless command glued to a separator and a redirect to a real file may be ALLOW."""
        cmds = ["ls", "echo hi"]
        seps = [";", "&&", "||", "|", "&", "\n", "(", ")", ";(", ");", "\n(", ")&&"]
        ops = [">", ">>", ">|", "&>", "&>>", "1>", "2>", "2>>", ">&", "<>", "&>|", "1>|"]
        spaces = ["", " "]
        seen = 0
        for cmd in cmds:
            for sep in seps:
                for op in ops:
                    for pre, mid, post in ((a, b, c) for a in spaces for b in spaces for c in spaces):
                        for text in (f"{cmd}{pre}{sep}{mid}{op}{post}victim", f"({cmd}){pre}{op}{post}victim",
                                     f"{{ {cmd}; }}{pre}{op}{post}victim"):
                            seen += 1
                            self.assertNotEqual(self.v(text), ALLOW, repr(text))
        self.assertGreater(seen, 1000)

    def test_env_prefix_is_never_allowed(self):
        for c in ["GIT_EXTERNAL_DIFF=./evil git diff", "LD_PRELOAD=x ls", "PAGER=./x git log", "env FOO=1 ls",
                  "env -S 'ls -l' ", "GIT_CONFIG_COUNT=1 git status", "git --config-env=core.fsmonitor=V status",
                  "git -c core.pager=x log", "git --exec-path=./x status", "git -p log"]:
            d = classify_command(c, self.work, self.cfg)
            self.assertEqual(d.verdict, ASK, c)
        d = classify_command("FOO=1 npm install", self.work, self.cfg)
        self.assertEqual((d.verdict, d.approval_key), (ASK, None))
        self.assertEqual(self.v("git -C . log"), ALLOW)
        self.assertEqual(self.v("git --no-pager diff"), ALLOW)

    def test_globs_cannot_reach_protected_paths(self):
        for c in ["cat ~/.ss*/id_rsa", "cat ~/.s[s]h/id_rsa", "cat /pro?/self/environ", "cat ~/.ssh*/*", "cat ~/.s{s,}h/id_rsa",
                  "cat ~/.{ssh,gnupg}/x", f"cat {self.home.parent}/*/.ssh/id_rsa", "cat /*/self/environ", "cat /sy*/kernel/x",
                  "grep -r PRIVATE ~", "grep -r PRIVATE /", "rg PRIVATE ~", "grep -rn x $HOME", "diff -r ~ /tmp"]:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["cat *.py", "ls ~/*", "ls ~", "grep -rn x .", "grep -rn x src", "cat /proc/cpuinfo", "rg x", "ls /home"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_variable_expansion_asks(self):
        for c in ["cat $FOO/x", "ls ${DIR}", 'cat "$X"']:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["ls $HOME", "echo '$FOO'", "echo $?", "ls $PWD"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_ansi_c_and_locale_quoting_ask(self):
        """bash reads \\' inside $'..' as an escaped quote; shlex does not, so later commands looked quoted."""
        for c in ["echo $'\\''; touch pwned; \\'", "echo $'\\'' >f \\'", "echo $'\\''; rm -r ~/x; \\'",
                  "echo $'\\'' >~/.bashrc \\'", "echo $'a\\nb'", 'echo $"x"', "echo $(echo $'\\''; touch f; \\')",
                  "echo \"$(echo $'\\''; touch f; \\')\"", "echo `echo $'\\''; touch f`", "bash -c \"echo \\$'\\\\'' ; touch f; \\\\'\"",
                  "eval \"echo \\`x\\`; touch f\"", "echo \"it's $(touch f)\"", "echo `echo 'a'`"]:
            d = classify_command(c, self.work, self.cfg)
            self.assertEqual(d.verdict, ASK, c)
            self.assertIsNone(d.approval_key, c)
        d = classify_command("npm install $'\\''; touch f; \\'", self.work, self.cfg)
        self.assertEqual((d.verdict, d.approval_key), (ASK, None))
        self.assertEqual(self.v("echo $'\\''; rm -rf ~; \\'"), BLOCK)

    def test_quote_scanner_does_not_crash_or_overflag(self):
        for c in ["echo 'it$'\"'\"'s'", "echo \"a'b\" 'c'", "echo \"$HOME\"", "echo '$'", "grep 'x$' a", "echo 'a' \"b\" c"]:
            self.assertEqual(self.v(c), ALLOW, c)
        for c in ["echo '", "echo \"", "echo `", "echo $(", "echo $'", "echo \\", "echo ${", "echo \"$(", "echo \"a'b\" $X 'c'"]:
            self.assertNotEqual(self.v(c), ALLOW, c)  # unterminated or expanding: never ALLOW, never an exception

    def test_long_option_abbreviations_are_dangerous_too(self):
        for c in ["sort --o=f a", "sort --outp f a", "sort --ou f a", "sort --output=f a", "sort --compress-prog=touch a",
                  "tail --foll a", "tail --f a", "date --se=now", "date --s=now", "ss --ki", "ss -D f", "ss --diag=f",
                  "file --comp -m mg", "file -C -m mg", "file --compile a", "file -m mg a", "file --magic=mg a",
                  "git diff --no-index --out=f a a", "git diff --no-index --ou f a a", "git log --ext-d", "git branch -l -D x",
                  "git tag -l -d x", "hostname -F a", "rg --pr=touch x", "tree -o f", "time -o f ls", "env --sp='touch f' ls"]:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["sort -r a", "sort --reverse a", "sort --numeric-sort a", "tail -n 3 a", "file a", "file -b a", "date +%s",
                  "date --utc", "ss -tln", "git branch --list", "git tag -l", "time -p ls", "env -i ls", "git diff --stat"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_matches_long(self):
        for opt in ["--o", "--ou", "--outp", "--output", "--output=f"]:
            self.assertTrue(matches_long(opt, "--output"), opt)
        for opt in ["--out", "-o", "--", "--oz", "-"]:
            self.assertEqual(matches_long(opt, "--output"), opt == "--out", opt)

    def test_grep_directories_recurse_is_recursive(self):
        for c in ["grep -d recurse x ~", "grep -drecurse x ~", "grep --directories=recurse x ~", "grep --directories recurse x ~",
                  "grep --dir=recurse x ~", "grep --rec x ~", "egrep -d recurse x /", "grep -d recurse x $HOME"]:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["grep -d skip x a", "grep -d recurse x .", "grep --directories=skip x ~"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_gate_refuses_what_bash_and_shlex_read_differently(self):
        nl = "\n"
        cases = [
            # comments hide quotes from bash but not from shlex
            f"echo a #'{nl}touch pwned{nl}'", f"echo #'{nl}mv a f{nl}'", f'echo #"{nl}find . -delete >"f"{nl}"',
            f"echo #'{nl}eval touch f{nl}'", f"echo #'{nl}sleep 9 &{nl}'", "echo a#b", "ls # x",
            # brace expansion smuggles in options
            "sort {-o,f} a", "find . -name a {-delete,-print}", "find . {-fprint,f}", "uniq {a,f}", "file {-C,-m,mg}",
            "git diff {--output=f,-C}", "echo {1..3}", "echo {a,b}", "echo }", "ls {a}",
            # backslash-newline is removed by bash outside single quotes
            f"find . -d\\{nl}elete", f"sort --\\{nl}output=f", f'sort "--out\\{nl}put=f" a', f'git diff -C to\\{nl}uch "--\\{nl}output=f"',
            # everything else the gate refuses
            "ls \\x", "echo \\'", "echo `ls`", "echo $(ls)", "echo ${HOME}", "echo $X", "echo $$", "echo !", "ls\ncat a",
            "ls\rcat a", "ls\tcat a", "ls\x00", "echo a~b", "echo ~root", "ls &", "ls & ls", "(ls)", "ls |& head", "ls > f",
            "ls >> f", "cat <<EOF", "cat <(ls)", "echo 'x", 'echo "x', 'echo "a\\b"', 'echo "$X"', 'echo "a$"', 'echo "!"',
            'echo "`ls`"', "echo $'a'", 'echo $"a"', "ls >&3", "ls >/dev/nullx", "ls >&2x", "ls <(a)", "ls <&0",
            "ls ;; ls", "ls &&& ls", "echo a;b"]
        for c in cases:
            d = classify_command(c, self.work, self.cfg)
            self.assertNotEqual(d.verdict, ALLOW, repr(c))
        self.assertEqual(self.v("{rm,-rf,x}"), BLOCK)
        self.assertEqual(self.v("{rm,-r,~}"), BLOCK)
        self.assertEqual(self.v("echo {dd,if=a,of=b}"), BLOCK)
        self.assertEqual(self.v("{sudo,ls}"), ELEVATED)
        self.assertEqual(self.v("r\\\nm -rf x"), BLOCK)

    def test_gate_keeps_everyday_commands(self):
        for c in ["ls -la", "ls src", "cat README.md", "grep -rn x . 2>/dev/null", "git log --oneline -5", "git status",
                  "git diff", "cat a | head", "ls 2>&1", 'echo "a > b"', "find . -name '*.py'", "du -sh .", "head -20 a.py",
                  "wc -l a.py", "pwd", "sort a | uniq -c | head", "cat package.json 2>/dev/null || true", "ls || :",
                  "wc -l src/*.py", "cat *.py", "echo '{a,b} #x \\ $X `y` !'", 'echo "#x {a}"', "ls ~", "ls ~/src",
                  "echo $HOME", 'echo "$HOME/x"', "echo $?", "cat < a", "ls &>/dev/null", "ls >>/dev/null"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_unquoted_glob_only_for_plain_readers(self):
        for c in ["echo *", "sort *", "uniq a?", "printf %s [a]*", "git diff *.py", "env ls *", "ls *.py | sort *"]:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["ls *", "grep x *.py", "du -sh *", "stat *", "diff a? b", "cat src/*/x"]:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_gate_never_downgrades_block_or_elevated(self):
        self.assertEqual(self.v("sudo ls #x"), ELEVATED)
        self.assertEqual(self.v("rm -rf x # c"), BLOCK)
        self.assertEqual(self.v("dd if=a of=b # c"), BLOCK)
        d = classify_command("npm run build #'\nrm x\n'", self.work, self.cfg)
        self.assertEqual((d.verdict, d.approval_key), (ASK, None))
        self.assertEqual(classify_command("npm run build 2>&1", self.work, self.cfg).approval_key, "npm run build")

    def test_downloads_and_deletes_are_tagged_and_never_remembered(self):
        downloads = ["curl -O https://x", "wget x", "aria2c x", "git clone x", "git pull", "git -C d fetch", "pip install x",
                     "pip3 install x", "pipx install x", "uv pip install x", "uv add x", "npm install", "npm i left-pad",
                     "pnpm add x", "yarn add x", "yarn", "bun install", "cargo install x", "cargo add x", "cargo fetch",
                     "go get x", "go install x", "flatpak install x", "docker pull x", "podman pull x", "yay -S x",
                     "paru x", "ssh host ls", "scp a host:b", "scp host:a .", "rsync -a a user@host:b", "sftp host",
                     "git submodule update --init", "huggingface-cli download x", "xarencode pull", "python3 -m pip install x",
                     "bash -c 'curl x'", "echo $(curl x)", "ls | xargs curl", "npm install && ls", "sudo pacman -S x"]
        deletes = ["rm a.txt", "rm -r build", "rmdir d", "unlink f", "find . -name x -delete", "git clean -fd", "git rm a",
                   "git branch -D x", "git reset --hard", "truncate -s0 a", "mv a /dev/null",
                   "find . -exec rm {} \\;", "ls | xargs rm", "bash -c 'rm a'", "eval rm a", "sudo rm a"]
        for c in downloads:
            d = classify_command(c, self.work, self.cfg)
            self.assertIn(d.verdict, (ASK, ELEVATED), c)
            self.assertIn(DOWNLOAD_TAG, d.tag, c)
            self.assertIsNone(d.approval_key, c)
        for c in deletes:
            d = classify_command(c, self.work, self.cfg)
            self.assertIn(d.verdict, (ASK, ELEVATED), c)
            self.assertIn(DELETE_TAG, d.tag, c)
            self.assertIsNone(d.approval_key, c)
        for c in ["ls", "git status", "git log", "pacman -Qi python", "yay -Qi x", "git branch", "git clean -n", "rsync -a a b",
                  "npm run build", "npm test", "cargo build", "pip list", "docker ps", "grep curl README.md", "echo rm"]:
            self.assertEqual(risk_kinds(c), set(), c)
            self.assertEqual(classify_command(c, self.work, self.cfg).tag, "", c)
        self.assertEqual(classify_command("npm run build", self.work, self.cfg).approval_key, "npm run build")

    def test_a_glob_that_expands_into_options_is_asked_for_find_rg_file_tail_and_sort(self):
        for name in ("-delete", "--pre=sh", "-o", "--output=x", "-exec", "-f"):
            (self.work / name).write_text("x")
        (self.work / "a").write_text("x")
        for c in ["find *", "find . -name x *", "rg x *", "file *", "tail *", "sort *", "uniq *", "tail -n1 ./*", "find -*",
                  "rg x ./*", "cat a | find *", "ls | tail *", "file ?", "tail [a-z]*"]:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["ls *", "cat *", "wc -l *", "head *", "du -sh *", "stat *", "grep x *", "diff a *"]:
            self.assertEqual(self.v(c), ALLOW, c)
        self.assertEqual(self.v("find . -name '*.py'"), ALLOW)  # a quoted glob is for find itself, not the shell

    def test_revision_syntax_with_a_tilde_is_allowed_but_tilde_expansion_is_not(self):
        for c in ["git diff HEAD~1", "git diff HEAD~", "git log main~2..HEAD", "git show v1.0~3", "git diff HEAD~1 HEAD~2",
                  "git log HEAD~1^", "git show HEAD^", "git show HEAD^2~1", "git diff HEAD~10 -- a", "git show feature_x~4:a"]:
            self.assertEqual(self.v(c), ALLOW, c)
        for c in ["ls ~", "ls ~/src"]:
            self.assertEqual(self.v(c), ALLOW, c)
        for c in ["git diff ~1", "git diff HEAD ~1", "git diff a=~/x", "git diff HEAD:~", "git diff HEAD~x", "git diff HEAD~/x",
                  "git diff HEAD~+", "git diff HEAD~-1", "git diff -~1", "git diff =~1", "git diff HEAD:~1", "echo a~b", "ls ~root",
                  "ls ~+", "ls ~-", "echo x:~", "git diff HEAD~1 ~root"]:
            self.assertNotEqual(self.v(c), ALLOW, c)

    def test_a_glob_that_reaches_ssh_keys_through_a_symlink_asks(self):
        ssh = self.home / ".ssh"
        ssh.mkdir()
        (ssh / "id_rsa").write_text("secret")
        (self.work / "keys").symlink_to(ssh)
        (self.work / "plain").mkdir()
        (self.work / "plain" / "f").write_text("x")
        for c in ["cat key?/id_rsa", "cat */*", "cat keys/*", "cat k*/id_rsa", "grep -r x *", "ls key*", "cat keys/id_rsa", "head -1 [k]eys/*"]:
            self.assertEqual(self.v(c), ASK, c)
        for c in ["cat plain/*", "ls pl*", "cat plain/f"]:
            self.assertEqual(self.v(c), ALLOW, c)
        (self.work / "keys").unlink()
        self.assertEqual(self.v("cat */*"), ALLOW)

    def test_a_glob_too_broad_to_check_asks(self):
        import xarencode.safety as safety
        for i in range(8):
            (self.work / f"f{i}").write_text("x")
        old = safety.GLOB_MATCH_LIMIT
        safety.GLOB_MATCH_LIMIT = 3
        try:
            self.assertEqual(self.v("cat f*"), ASK)
            self.assertEqual(self.v("cat f1*"), ALLOW)
        finally:
            safety.GLOB_MATCH_LIMIT = old

    RISKY = [  # (command, expected kind)
        ("gh repo clone o/r", "download"), ("gh repo fork o/r", "download"), ("gh repo delete o/r --yes", "delete"),
        ("gh release download v1", "download"), ("gh release delete v1", "delete"), ("gh api -X DELETE repos/o/r", "delete"),
        ("gh api repos/o/r --method POST", "download"), ("gh api repos/o/r -f a=b", "download"), ("gh -R o/r release download", "download"),
        ("apt install x", "download"), ("apt-get install -y x", "download"), ("apt -o X=y install x", "download"),
        ("aptitude install x", "download"), ("dnf install x", "download"), ("yum remove x", "delete"), ("zypper install x", "download"),
        ("brew install x", "download"), ("snap install x", "download"), ("nix profile install x", "download"),
        ("apt remove x", "delete"), ("apt purge x", "delete"), ("apt autoremove", "delete"), ("apt-get update", "download"),
        ("apt upgrade", "download"), ("dnf upgrade", "download"), ("brew uninstall x", "delete"), ("snap remove x", "delete"),
        ("pacman -S x", "download"), ("pacman -Sy", "download"), ("pacman -Syu", "download"), ("pacman -U x.pkg.tar.zst", "download"),
        ("pacman -U https://x/y.pkg.tar.zst", "download"), ("pacman -R x", "delete"), ("pacman -Rs x", "delete"),
        ("pacman -Rns x", "delete"), ("pacman -Rdd x", "delete"), ("pacman --noconfirm -Rns x", "delete"), ("pacman --remove x", "delete"),
        ("sudo pacman -S x", "download"), ("sudo pacman -Rns x", "delete"), ("sudo pacman -Syu", "download"), ("pacman -Sc", "delete"),
        ("docker run x", "download"), ("podman run x", "download"), ("docker pull x", "download"), ("docker rm x", "delete"),
        ("docker rmi x", "delete"), ("docker compose up", "download"), ("docker compose pull", "download"),
        ("docker-compose up -d", "download"), ("docker image prune", "delete"), ("docker system prune -a", "delete"),
        ("docker compose down", "delete"), ("docker -H h run x", "download"), ("podman rm x", "delete"),
        ("busybox wget http://x", "download"), ("busybox rm a", "delete"),
        ("git stash drop", "delete"), ("git stash clear", "delete"), ("git restore a", "delete"), ("git restore .", "delete"),
        ("git checkout -- a", "delete"), ("git checkout .", "delete"), ("git checkout -f", "delete"), ("git checkout HEAD -- a", "delete"),
        ("git switch --discard-changes main", "delete"), ("git switch -f main", "delete"), ("git worktree remove w", "delete"),
        ("git branch -d b", "delete"), ("git branch -D b", "delete"), ("git tag -d v1", "delete"), ("git reflog expire --all", "delete"),
        ("git remote remove o", "delete"), ("git push origin --delete b", "delete"), ("git push origin :b", "delete"),
        ("git -C d stash drop", "delete"), ("git submodule deinit x", "delete"),
        ("npm uninstall x", "delete"), ("npm remove x", "delete"), ("npm rm x", "delete"), ("yarn remove x", "delete"),
        ("pnpm uninstall x", "delete"), ("pnpm remove x", "delete"), ("npm cache clean --force", "delete"), ("bun remove x", "delete"),
        ("pip uninstall x", "delete"), ("pip3 uninstall -y x", "delete"), ("pipx uninstall x", "delete"), ("pip install x", "download"),
        ("python3 -m pip install x", "download"), ("python3 -m pip uninstall x", "delete"), ("python -m pip download x", "download"),
        ("flatpak uninstall x", "delete"), ("cargo uninstall x", "delete"), ("go clean", "delete"), ("go clean -cache", "delete"),
        ("make clean", "delete"), ("cargo clean", "delete"), ("nix-collect-garbage -d", "delete"),
        ("xargs apt install", "download"), ("find . -exec docker rm {} \\;", "delete"), ("bash -c 'gh repo clone x'", "download"),
        # the general fallback: any other tool whose subcommand word is one of the listed verbs
        ("sometool install x", "download"), ("sometool i x", "download"), ("sometool add x", "download"), ("sometool get x", "download"),
        ("sometool clone x", "download"), ("sometool download x", "download"), ("sometool fetch", "download"),
        ("sometool pull", "download"), ("sometool update", "download"), ("sometool upgrade", "download"),
        ("sometool remove x", "delete"), ("sometool rm x", "delete"), ("sometool delete x", "delete"), ("sometool del x", "delete"),
        ("sometool uninstall x", "delete"), ("sometool purge x", "delete"), ("sometool prune", "delete"), ("sometool clean", "delete"),
        ("sometool drop x", "delete"), ("sometool destroy x", "delete"), ("sometool kill x", "delete"), ("sometool -v --flag install x", "download"),
        ("sometool group remove x", "delete"),
        # round 7: downloads
        ("uvx ruff", "download"), ("deno run https://x/y.ts", "download"), ("deno install x", "download"), ("go run example.com/x@latest", "download"),
        ("go run .", "download"), ("pnpm exec x", "download"), ("pnpm dlx x", "download"), ("git lfs pull", "download"),
        ("git lfs fetch", "download"), ("git lfs clone x", "download"), ("git annex get f", "download"), ("yt-dlp https://x", "download"),
        ("rclone copy r:x .", "download"), ("rclone ls r:", "download"), ("nc host 80", "download"), ("ncat host 80", "download"),
        ("socat - TCP:h:1", "download"), ("telnet h", "download"), ("ftp h", "download"), ("http GET x", "download"), ("https x", "download"),
        ("xh x", "download"), ("svn checkout x", "download"), ("svn co x", "download"), ("svn update", "download"), ("aria2c x", "download"),
        ("git submodule add x y", "download"), ("sftp h", "download"), ("npx vite", "download"), ("bunx x", "download"),
        # round 7: deletes
        ("rsync --delete -a d/ e/", "delete"), ("rsync -a --delete-after d/ e/", "delete"), ("rsync --remove-source-files -a d e", "delete"),
        ("rclone sync a b", "delete"), ("rclone delete r:x", "delete"), ("rclone purge r:x", "delete"), ("cp /dev/null f", "delete"),
        ("trash-put f", "delete"), ("trash f", "delete"), ("gio trash f", "delete"), ("tar xf f --remove-files", "delete"),
        ("tar --remove-files -cf a.tar d", "delete"), ("zip -m a.zip f", "delete"), ("zip -rm a.zip d", "delete"), ("git lfs prune", "delete"),
        ("git prune", "delete"), ("git gc --prune=now", "delete"), ("git submodule foreach rm f", "delete"),
        ("git submodule foreach 'rm f'", "delete"), ("git submodule foreach --recursive rm f", "delete"), ("git annex drop f", "delete"),
        # round 7: in-place edits and overwrites
        ("sed -i s/a/b/ f", "edit"), ("sed -i.bak s/a/b/ f", "edit"), ("sed -ni 's/a/b/p' f", "edit"), ("sed --in-place s/a/b/ f", "edit"),
        ("perl -pi -e 's/a/b/' f", "edit"), ("perl -i.bak -pe 's/a/b/' f", "edit"), ("awk -i inplace '{print}' f", "edit"),
        ("> f", "overwrite"), (": > f", "overwrite"), ("echo x > f", "overwrite"), ("cat a >| f", "overwrite"), ("ls &> f", "overwrite"),
        ("echo x 2> err.log", "overwrite"), ("true > a && ls", "overwrite"),
    ]
    TAGS = {"download": DOWNLOAD_TAG, "delete": DELETE_TAG, "edit": EDIT_TAG, "overwrite": OVERWRITE_TAG}

    def test_every_download_and_delete_command_is_tagged_with_no_always_key_even_when_allowed_by_config(self):
        self.cfg["safety"]["allow"] = sorted({c.split()[0] for c, _ in self.RISKY} | {"git", "git stash drop", "docker", "gh"})
        for cmd, kind in self.RISKY:
            for cfg in (default_config(), self.cfg):
                d = classify_command(cmd, self.work, cfg)
                self.assertIn(d.verdict, (ASK, ELEVATED), cmd)
                self.assertTrue(d.tag, cmd)
                self.assertIn(self.TAGS[kind], d.tag, cmd)
                self.assertIsNone(d.approval_key, cmd)
            self.assertIn(kind, risk_kinds(cmd), cmd)

    def test_sudo_does_not_hide_a_tag(self):
        for cmd, kind in [("sudo apt install x", "download"), ("sudo docker rm x", "delete"), ("sudo gh repo delete x", "delete")]:
            d = classify_command(cmd, self.work, self.cfg)
            self.assertEqual((d.verdict, bool(d.tag), d.approval_key), (ELEVATED, True, None), cmd)

    def test_reads_that_share_a_word_with_a_risky_command_are_not_tagged(self):
        for c in ["gh repo view", "gh pr list", "gh api repos/o/r", "gh api -X GET repos/o/r", "apt list", "apt search x", "apt show x",
                  "apt-cache policy x", "pacman -Qi x", "pacman -Ss x", "pacman -Si x", "pacman -Sl", "pacman -Qdt", "pacman -F x",
                  "docker ps", "docker images", "docker compose ps", "docker logs x", "docker inspect x", "git log install",
                  "git diff clean", "git branch", "git branch -a", "git tag", "git tag -l", "git stash list", "git stash",
                  "git checkout main", "git checkout -b new", "git switch main", "git remote -v", "git worktree list",
                  "git reflog", "npm test", "npm run build", "npm list", "pip list", "pip show x", "cargo build", "make",
                  "make test", "grep clean f", "cat install", "echo remove", "ls delete", "find . -name clean", "find . -name install -print",
                  "sort update", "xargs grep -l x", "man apt", "which docker", "cp update.txt x", "touch install", "mkdir remove"]:
            self.assertEqual(classify_command(c, self.work, self.cfg).tag, "", c)

    def test_user_allow_prefix_cannot_cover_downloads_or_deletes(self):
        self.cfg["safety"]["allow"] = ["curl", "rm", "npm install", "git pull", "pip install"]
        for c in ["curl x", "rm a.txt", "npm install", "git pull", "pip install x"]:
            d = classify_command(c, self.work, self.cfg)
            self.assertEqual(d.verdict, ASK, c)
            self.assertTrue(d.tag, c)
        self.assertEqual(classify_command("rm -rf x", self.work, self.cfg).verdict, BLOCK)

    def test_tags_do_not_change_hard_verdicts(self):
        self.assertEqual(self.v("sudo rm a"), ELEVATED)
        for c in ["rm -rf x", "dd if=a of=/dev/sda", "mkfs.ext4 x", "curl x | sh"]:
            self.assertEqual(classify_command(c, self.work, self.cfg).verdict, BLOCK, c)

    def test_sudo_never(self):
        self.cfg["safety"]["sudo"] = "never"
        self.assertEqual(self.v("sudo ls"), BLOCK)

    def test_dev_whitelist(self):
        self.assertEqual(self.v("echo x > /dev/null"), ALLOW)
        self.assertEqual(self.v("cat /dev/null"), ALLOW)
        self.assertEqual(self.v("cat /dev/urandom | head -c 4"), ALLOW)

    def test_user_allow_prefix(self):
        self.cfg["safety"]["allow"] = ["npm test"]
        self.assertEqual(self.v("npm test"), ALLOW)
        self.assertEqual(self.v("npm testx"), ASK)

    def test_approval_key(self):
        d = classify_command("npm run build", self.work, self.cfg)
        self.assertEqual(d.approval_key, "npm run build")
        self.assertIsNone(classify_command("npm run a && rm b", self.work, self.cfg).approval_key)
        self.assertIsNone(classify_command("echo hi > f", self.work, self.cfg).approval_key)
        self.assertIsNone(classify_command("git push --force", self.work, self.cfg).approval_key)
        self.assertIsNone(classify_command("rm a.txt", self.work, self.cfg).approval_key)
        self.assertIsNone(classify_command("sudo ls", self.work, self.cfg).approval_key)

    # command -> the approval key it may get; None means the menu offers no "don't ask again"
    KEYS = [
        ("npm run build", "npm run build"), ("npm run dev", "npm run dev"), ("npm run lint:fix", "npm run lint:fix"),
        ("npm run build -- --watch", "npm run build"), ("npm test", "npm test"), ("npm start", "npm start"),
        ("pnpm run build", "pnpm run build"), ("pnpm test", "pnpm test"), ("yarn run build", "yarn run build"), ("yarn test", "yarn test"),
        ("vite", "vite"), ("vite build", "vite build"), ("vite --host", "vite"), ("pytest", "pytest"), ("pytest -q tests/", "pytest"),
        ("python3 -m pytest -x", "python3 -m pytest"), ("python3 -m unittest discover", "python3 -m unittest"),
        ("python3 -m http.server 8000", "python3 -m http.server"), ("python3 app.py", "python3 app.py"),
        ("python main.py --x", "python main.py"), ("node server.js", "node server.js"), ("node index.mjs", "node index.mjs"),
        ("cargo build", "cargo build"), ("cargo test", "cargo test"), ("cargo check", "cargo check"), ("cargo fmt", "cargo fmt"),
        ("cargo clippy", "cargo clippy"), ("cargo run", "cargo run"), ("cargo build --release", "cargo build"),
        ("go build ./...", "go build"), ("go test ./...", "go test"), ("go vet ./...", "go vet"), ("go fmt ./...", "go fmt"),
        ("make", "make"), ("make test", "make test"), ("make build", "make build"), ("make -j4 all", "make all"), ("tsc", "tsc"),
        ("tsc --noEmit", "tsc"), ("eslint src", "eslint"), ("prettier --check .", "prettier"), ("git add a.txt", "git add"),
        ("git add -A", "git add"), ("git commit -m x", "git commit"), ("git checkout -b feature", "git checkout -b"),
        ("git switch main", "git switch"), ("git switch -c x", "git switch"), ("git stash", "git stash"),
        ("git stash push -m x", "git stash"), ("git stash -u", "git stash"),
        # everything below must never be remembered: downloads, deletes, installs, other subcommands, odd shapes
        ("make clean", None), ("make install", None), ("make distclean", None), ("make -C ../x", None), ("make CC=evil", None),
        ("make clean-all", None), ("make deploy", None), ("npm run clean", None), ("npm run publish", None), ("npm run deploy", None),
        ("npm run --prefix ../x build", None), ("npm run", None), ("npm install", None), ("npm publish", None), ("npm exec x", None),
        ("npx vite", None), ("npx eslint", None), ("pnpm dlx x", None), ("pnpm exec x", None), ("yarn add x", None), ("yarn", None),
        ("cargo install x", None), ("cargo clean", None), ("cargo publish", None), ("cargo build --force", None), ("go get x", None),
        ("go run .", None), ("go install x", None), ("go clean", None), ("python3 -c 'print(1)'", None), ("python3 -m pip install x", None),
        ("python3 -m venv v", None), ("python3 script.sh", None), ("node -e 'x'", None), ("node app.txt", None), ("pytest && ls", None),
        ("git commit --amend", None), ("git checkout main", None), ("git checkout -- a", None), ("git stash pop", None),
        ("git stash drop", None), ("git stash clear", None), ("git add -f x", None), ("git push", None), ("git pull", None),
        ("git lfs pull", None), ("git lfs prune", None), ("git lfs track x", None), ("git submodule foreach rm f", None),
        ("git submodule update", None), ("git submodule status", None), ("git remote add o x", None), ("git branch newb", None),
        ("git rebase main", None), ("git merge x", None), ("git reset HEAD~1", None), ("git -C d add a", None), ("git tag v1", None),
        ("gh pr create", None), ("gh repo clone x", None), ("docker build .", None), ("docker run x", None), ("docker compose up", None),
        ("pacman -S x", None), ("pacman -Syu", None), ("apt install x", None), ("uvx ruff", None), ("uv run x", None), ("deno run x.ts", None),
        ("pip install x", None), ("curl x", None), ("rm a", None), ("sed -i s/a/b/ f", None), ("perl -pi -e s/a/b/ f", None),
        ("yt-dlp x", None), ("rclone sync a b", None), ("rsync --delete a b", None), ("trash f", None), ("echo hi > f", None),
        ("touch f", None), ("mkdir d", None), ("cp a b", None), ("mv a b", None), ("chmod +x f", None), ("sleep 60", None),
        ("ruby x.rb", None), ("bash script.sh", None), ("./run.sh", None), ("sudo npm run build", None), ("FOO=1 npm run build", None),
        ("npm run build && npm test", None), ("npm run build; rm x", None), ("npm run build | tee f", None), ("npm run build &", None),
        ("npm run build > out.log", None), ("(npm run build)", None), ("cd d && make", None), ("$(echo npm) run build", None),
    ]

    def test_only_a_short_list_of_developer_commands_can_be_remembered(self):
        """"Don't ask again" is fail-closed: a key exists for these well-known commands only, with the subcommand and
        target in it, and for nothing else (git lfs, gh, docker, pacman, uvx, go run, sed -i ...)."""
        self.assertGreater(len(self.KEYS), 140)
        for cmd, key in self.KEYS:
            d = classify_command(cmd, self.work, self.cfg)
            self.assertEqual(d.approval_key, key, cmd)
            if key:
                self.assertEqual((d.verdict, d.tag), (ASK, ""), cmd)  # a key never rides on an allowed or tagged command
        # approving one use covers that use only
        keys = {classify_command(c, self.work, self.cfg).approval_key for c in ("npm run build", "npm run dev", "npm run test:e2e")}
        self.assertEqual(len(keys), 3)

    def test_no_command_outside_the_list_ever_gets_a_key_even_when_it_asks(self):
        known = {k for _, k in self.KEYS if k}
        for cmd in ASKED + [c for c, _ in self.RISKY] + BLOCKED + ELEVATED_CASES:
            d = classify_command(cmd, self.work, self.cfg)
            if d.approval_key:
                self.assertIn(d.approval_key, known, cmd)

    def test_a_nul_byte_asks_instead_of_crashing(self):
        for c in ['echo "$(echo \': ~-/f -o\x00f\')"', "ls \x00", "cat a\x00b", "\x00", "git status\x00", "ls ~/\x00x", "cat $(echo \x00)"]:
            d = classify_command(c, self.work, self.cfg)
            self.assertEqual(d.verdict, ASK, repr(c))
            self.assertIsNone(d.approval_key)
        self.assertEqual(resolve_path("a\x00b", self.work, write=False, cfg=self.cfg)[1], BLOCK)
        self.assertEqual(resolve_path("a\x00b", self.work, write=True, cfg=self.cfg)[1], BLOCK)

    def test_credential_locations_ask_for_shell_reads_but_ordinary_reads_stay_allowed(self):
        home = self.home
        for rel in (".aws", ".config/gh", ".kube", ".password-store", ASSISTANT_DIR, ".mozilla", ".config/rclone", ".azure",
                    ".config/gcloud", ".local/share/keyrings", ".docker"):
            (home / rel).mkdir(parents=True, exist_ok=True)
        for rel in (".netrc", ".npmrc", ".pypirc", ".git-credentials", ".bash_history", ".zsh_history", ".aws/credentials",
                    ".config/gh/hosts.yml", ".docker/config.json", ".cargo/credentials.toml", f"{ASSISTANT_DIR}/settings.json"):
            (home / rel).parent.mkdir(parents=True, exist_ok=True)
            (home / rel).write_text("secret\n")
        (home / ".config/chromium/Default").mkdir(parents=True)
        (home / ".config/chromium/Default/Login Data").write_text("x")
        asked = ["cat ~/.aws/credentials", "cat ~/.config/gh/hosts.yml", "grep -r token ~/.config", "cat $HOME/.netrc", "head ~/.npmrc",
                 "cat ~/.pypirc", "cat ~/.git-credentials", "cat ~/.docker/config.json", "ls ~/.kube", "cat ~/.cargo/credentials.toml",
                 "cat ~/.cargo/cred*", "cat ~/.bash_history", "tail ~/.zsh_history", f"grep x ~/{ASSISTANT_DIR}/settings.json", f"cat ~/{ASSISTANT_DIR}/*",
                 "cat '~/.aws/credentials'".replace("'", ""), "cat /etc/shadow", "cat /etc/sudoers", "cat /etc/sudoers.d/x", "cat /etc/sudoers*",
                 "ls ~/.password-store", "cat ~/.config/rclone/rclone.conf", "ls ~/.local/share/keyrings", "ls ~/.mozilla",
                 "cat ~/.config/chromium/Default/Login\\ Data", "cat ~/.config/chromium/*/Login*", "cat ~/.azure/x", "cat ~/.config/gcloud/x",
                 "grep -r x ~", "cat ~/.aws/*", "cat ~/.a*/credentials", "wc -l ~/.kube/config", "diff ~/.netrc ~/.npmrc"]
        for c in asked:
            d = classify_command(c, self.work, self.cfg)
            self.assertEqual(d.verdict, ASK, c)
            self.assertIsNone(d.approval_key, c)
        allowed = ["cat /etc/os-release", "ls /etc", "cat /proc/cpuinfo", "pacman -Qi python", "df", "free", "uname -a", "cat /etc/hostname",
                   "ls ~", "ls ~/.config", "cat README.md", "cat /etc/passwd", "ls /etc/ssh", "cat /etc/hosts", "grep -n x /etc/fstab",
                   "du -sh ~/.cache", "wc -l /etc/os-release", "stat /etc", "cat ~/notes.txt"]
        for c in allowed:
            self.assertEqual(self.v(c), ALLOW, c)

    def test_read_file_asks_for_the_same_credential_locations_and_gives_no_key(self):
        from xarencode.tools import build_registry, ToolContext
        home = self.home
        (home / ".aws").mkdir()
        (home / ".aws/credentials").write_text("k\n")
        (home / ".config/chromium/Default").mkdir(parents=True)
        (home / ".config/chromium/Default/Login Data").write_text("x")
        (home / "notes.txt").write_text("hi\n")
        tool = build_registry(self.cfg).get("read_file")
        tctx = ToolContext(self.work, self.cfg)
        for p in ("~/.aws/credentials", "~/.config/chromium/Default/Login Data", "/etc/shadow", "/etc/sudoers", "~/.netrc", f"~/{ASSISTANT_DIR}/x"):
            plan = tool.plan({"path": p}, tctx)
            self.assertEqual((plan.verdict, plan.approval_key), (ASK, None), p)
        self.assertEqual(tool.plan({"path": "~/.ssh/id_rsa"}, tctx).verdict, BLOCK)
        outside = tool.plan({"path": "~/notes.txt"}, tctx)
        self.assertEqual(outside.verdict, ASK)  # an ordinary outside read still asks, and may be remembered
        self.assertTrue(outside.approval_key)
        self.assertEqual(tool.plan({"path": "/etc/os-release"}, tctx).verdict, ASK)
        wd_home = ToolContext(home, self.cfg)  # the working directory IS the home folder: the credential still asks
        self.assertEqual(tool.plan({"path": ".aws/credentials"}, wd_home).verdict, ASK)
        self.assertEqual(tool.plan({"path": "notes.txt"}, wd_home).verdict, ALLOW)

    def test_unparseable_asks(self):
        self.assertEqual(self.v("echo 'unterminated"), ASK)

    def test_rm_recursive_inside_workdir_asks(self):
        (self.work / "build").mkdir()
        self.assertEqual(self.v("rm -r build"), ASK)
        self.assertEqual(self.v(f"rm -r {self.work}"), BLOCK)
        self.assertEqual(self.v("rm -r ../other"), BLOCK)


class PathTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.cfg = default_config()

    def test_inside_outside(self):
        _, v, _ = resolve_path("a.txt", self.work, write=True, cfg=self.cfg)
        self.assertEqual(v, ALLOW)
        _, v, _ = resolve_path("../x", self.work, write=True, cfg=self.cfg)
        self.assertEqual(v, BLOCK)
        _, v, _ = resolve_path("../x", self.work, write=False, cfg=self.cfg)
        self.assertEqual(v, ASK)
        self.cfg["safety"]["write_outside_workdir"] = True
        _, v, _ = resolve_path("../x", self.work, write=True, cfg=self.cfg)
        self.assertEqual(v, ASK)

    def test_symlink_escape(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        os.symlink(outside, self.work / "link")
        _, v, _ = resolve_path("link/f", self.work, write=True, cfg=self.cfg)
        self.assertEqual(v, BLOCK)

    def test_forbidden(self):
        for p in ("/proc/cpuinfo", "/dev/sda", "/sys/x", "~/.ssh/id_rsa", "~/.gnupg"):
            _, v, _ = resolve_path(p, self.work, write=False, cfg=self.cfg)
            self.assertEqual(v, BLOCK, p)

    def test_git_dir(self):
        (self.work / ".git").mkdir()
        _, v, _ = resolve_path(".git/config", self.work, write=True, cfg=self.cfg)
        self.assertEqual(v, BLOCK)
        _, v, _ = resolve_path(".gitignore", self.work, write=True, cfg=self.cfg)
        self.assertEqual(v, ALLOW)


if __name__ == "__main__":
    unittest.main()
