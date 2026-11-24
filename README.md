# XarenCode

A lean terminal coding agent for a local model. It talks to `llama-server` (llama.cpp),
fills your terminal with a full-window view (streaming replies, tool cards, diffs, an approval
menu, a status bar), has a friendly persona called Xaren and a small blue cat mascot.

Built for a small machine: CPU only, 4096 tokens of context, about 2.6 GB on disk with the
Qwen3-4B-Instruct Q4_K_M model.

## Install

One-line install (uses `pipx`, `uv`, or an isolated venv in `~/.local/share/xarencode/venv`):

```sh
curl -fsSL https://raw.githubusercontent.com/adamelhadrii/xarencode/main/install.sh | bash
```

Or install directly with `pipx` / `uv` / `pip`:

```sh
pipx install git+https://github.com/adamelhadrii/xarencode.git
# or: uv tool install git+https://github.com/adamelhadrii/xarencode.git
```

System dependencies (Arch Linux) & first-time setup:

```sh
sudo pacman -S --needed llama-cpp ripgrep
xarencode pull                           # downloads the model (about 2.5 GB, resumable)
xarencode doctor                         # checks everything
xarencode                                # start chatting in the current folder
```

Local development install:

```sh
git clone https://github.com/adamelhadrii/xarencode.git && cd xarencode
python -m venv --system-site-packages .venv && .venv/bin/pip install -e .
```

## Use

| input | what it does |
|---|---|
| plain text | ask Xaren to read, search, edit files, run commands, start dev servers |
| `/help` | list slash commands (`/compact`, `/context`, `/resume`, `/model`, `/server`, `/dev`, `/vm`, ...) |
| `!cmd` | run a shell command yourself; the output is added to the conversation |
| `@file` | attach an excerpt of a file |
| Esc | interrupt a reply |
| Shift+Tab | auto-accept edits on or off |
| Alt+Enter | new line |
| `/copy [code\|all]` | copy the last reply, its last code block, or the whole session |
| `/pet` | pet the cat |
| `/mascot [on\|off]` | show or hide the cat for this session |
| `/header [full\|small\|off]` | change the header size for this session |
| `/theme [dark\|light\|contrast\|mono]` | switch the UI colour theme |
| `/sidebar [on\|off]` | toggle the session/context sidebar (`Ctrl+B`) |
| `/timestamps [on\|off]` | show or hide `[HH:MM]` timestamps on messages |

## Full-window view

XarenCode takes over the terminal like vim or htop: a fixed header with the cat and the
XARENCODE wordmark, a scrolling chat, an input bar and a status bar.

- PgUp/PgDn or the mouse wheel scroll; Ctrl+Home and Ctrl+End jump to the top and the live end.
- Alt+Enter adds a new line, Esc interrupts, Shift+Tab toggles auto-accept for edits.
- F3 pets the cat, and clicking the cat or the wordmark plays an animation.
- Tradeoff: there is no native scrollback while it runs, and mouse capture blocks native text
  selection. Press F2 to toggle mouse capture, or hold Shift and drag to select. `/copy` copies replies.
- `xarencode -p "prompt"` gives plain one-shot output for scripts. With no real terminal, plain
  `xarencode` exits with code 2 and a message pointing to `-p`.

One-shot mode: `xarencode -p "list the files here and count them"`.
Other options: `--resume [ID]`, `--workdir DIR`, `--config FILE`, `xarencode server start|stop|status|log`.

## Social

- Xaren greets you by time of day, picks playful spinner words, and celebrates passing tests,
  a dev server coming up and applied edits. `/vibe friendly|hype|chill|focused` changes the tone.
  Set `persona.user_name` in the config to be greeted by name.
- `/share md|svg|html` saves the session as a Markdown transcript or as a terminal-screenshot
  card (SVG or HTML) in `~/Documents/XarenCode/`. Your home folder path and anything that looks
  like a secret is redacted first.
- The blue cat reacts to approvals, blocked commands and celebrations, falls asleep after 3 idle
  minutes and wears sunglasses in YOLO mode. `/mascot off` or `NO_COLOR` hides the pixel cat and
  leaves a text cat `=^.^=`.

## Safety

- Read-only shell commands run freely. Everything else asks first.
- Known limit: the read-only check is a classifier, not a sandbox. Git can run programs named in a
  repository's own config, so do not run XarenCode inside a git repo you do not trust. The approval
  prompt before every non-read-only command is the real safety gate.
- `rm -rf`, `dd`, `mkfs`, any block device path (such as `/dev/sda`) and downloads piped into a
  shell are always blocked.
- `sudo` runs only after you type `sudo` to confirm (or never, with `safety.sudo = "never"`).
- File writes stay inside the working directory, and never inside `.git/`.
- A command counts as read-only only if it is a plain chain of simple commands (no comments, braces,
  backslashes, `$(...)`, backticks or `$'...'`); anything fancier asks.
- A read-only `git` command (status, diff, log, show, ...) runs freely only when the repository git will use has
  nothing in its own `.git/config` that could run a program (`core.fsmonitor`, `gpg.program`, textconv and filter
  drivers, `include.path`, `alias.*`, ...). Otherwise it asks. Free runs also override those keys as a second line
  of defence.
  `git -C DIR` and `--git-dir` are followed, and ask when they point outside the working directory.
- An unquoted `*` or `?` is only trusted for `ls`, `cat`, `wc`, `head`, `du`, `stat`, `grep` and `diff`: a file named
  `-delete` or `--pre=sh` must not turn into an option for `find`, `rg`, `file` or `tail`. Globs that reach `~/.ssh`
  through a symlink ask.
- The command classifier is a speed bump, not a sandbox: read the approval prompt.

### Normal mode and yolo mode

In normal mode, commands that download (`curl`, `wget`, `git clone/pull/fetch`, `pip/npm/cargo/go install`,
`pacman -S`, `ssh`, `scp`, `docker pull`, `xarencode pull`, ...) or delete (`rm`, `rmdir`, `find -delete`,
`git clean`, `git reset --hard`, `truncate`, ...) ask **every time**. They show a tag in the approval card, have no
"don't ask again" option, and are not covered by auto-edits (Shift+Tab) or `safety.allow`. The `dev_server` tool
follows the same rule.

`xarencode --yolo` (or `-yolo`, also with `-p`) or `/yolo on` answers yes to every ask for that run or session:
downloads, deletes, file writes inside the working directory. The tool cards still print what ran, and the status bar
and banner show a `YOLO` badge. `/yolo off` goes back to asking. Yolo can only be switched on for a single run
(the flag) or a single session (`/yolo on`). It is never read from the config file or from environment variables, and
a `[safety] yolo` line in the config is ignored with a warning.

What stays the same in yolo mode: hard blocks (`rm -rf`, `dd`, `mkfs`, block devices, piping a download into a
shell) are still blocked, `sudo` still needs you to type `sudo` (and `safety.sudo = "never"` still blocks it), file
tools still cannot write outside the working directory (when you allow that in the config, each such write still
asks), and the classifier gives the same verdicts as in normal mode. The model cannot turn yolo on: it is not a tool
argument.

## Config

`~/.config/xarencode/config.toml` is created on first run, fully commented. Swapping the model is
the one `model.gguf` line. Dev server presets and VM machines are shown there as commented examples.
The llama-server flags live in `server.args`.

Data lives in `~/.local/share/xarencode` (models, sessions) and `~/.local/state/xarencode` (logs, pids).

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
```

Core code and tests use only the standard library; the UI tests skip if `rich` is missing.
A mock llama-server lives in `tests/mock_llama_server.py` (`--demo` for a manual UI try-out with
`xarencode --config tests/fixtures/mock.toml`).

MIT licensed.
