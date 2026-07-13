# Git, in plain language

Git is a tool that saves **snapshots** of your project over time, so you can
always go back and see (or restore) an earlier version. Think of it like a
video game's save points — each "commit" is a save point you can return to.

For this project, I (Claude) will run the git commands for you at the end of
each phase (and sometimes mid-phase) to create a checkpoint. This guide
explains what those commands actually do, so it's not a black box.

## The three core concepts

1. **Repository ("repo")** — the project folder itself, once it's being
   tracked by git. Ours was created with `git init`.
2. **Staging area** — a holding area where you pick *which* changed files
   you want to include in the next snapshot. You add files to it with
   `git add`.
3. **Commit** — an actual saved snapshot of whatever is in the staging area,
   along with a short message describing what changed. Made with
   `git commit`.

So the normal flow is always: change some files → `git add` the ones you
want to save → `git commit` to actually save them.

## Commands used in this project

### `git init`
Turns the current folder into a git repository. Only needs to happen once
per project — already done in Phase 1.

### `git status`
Shows what's changed since the last commit: new files, edited files,
deleted files. **Safe to run any time** — it never changes anything, it
only reports. Good habit: run this before doing anything else, so you know
what state you're in.

### `git add <file>` (or `git add <file1> <file2> ...`)
Stages specific files — marks them as "include this in the next commit".
We deliberately add files **by name** rather than using `git add .` (which
would add *everything*, including files we might not want, like a stray
secret or a huge generated file we forgot to ignore).

### `git commit -m "message"`
Takes everything currently staged and saves it as a permanent snapshot in
the project's history, labeled with the message. The message should say
*why*/*what*, e.g. `"Phase 1: project scaffolding, config loader, tests"`.

### `git log`
Shows the history of commits — like a list of all your save points, newest
first. Add `--oneline` (`git log --oneline`) for a compact one-line-per-commit
view.

### `git diff`
Shows the exact line-by-line changes that haven't been committed yet. Useful
if you want to see *exactly* what changed before staging/committing it.

### `.gitignore`
Not a command — a file that tells git "never track these". We use it for
things that shouldn't be snapshotted: your `venv/` folder (everyone
generates their own), your `.env` file (contains your real secret API key),
and generated data/index files (they're reproducible from code, so
versioning them just bloats the repo).

## A typical checkpoint, end to end

```bash
git status                          # see what changed
git add src/data_gen/generate.py    # stage the specific file(s) you want
git commit -m "Phase 2: synthetic data generator"   # save the snapshot
git log --oneline                   # confirm it's there
```

## Things we are *not* doing (on purpose, for now)

- **No GitHub/remote push yet.** Everything above is 100% local to your
  machine — there's no "undo my local history" risk to anything shared.
  If/when you want a GitHub backup, that's a separate, deliberate step
  we'll do together (creating a remote and `git push`), not something
  that happens silently.
- **No branches yet.** Branches let multiple lines of work happen in
  parallel; this project is linear (phase by phase), so we're just
  committing straight to the main line (called `main`) to keep things
  simple while you're learning.
- **No force-pushes, resets, or history rewrites.** Those commands can
  permanently discard work, so they're out of scope for normal checkpoint
  use — I won't run them without stopping to explain exactly what would be
  lost and confirming with you first.

## If you want to poke around yourself

- `git status` and `git log --oneline` are both 100% safe (read-only) —
  run them any time you're curious what state the repo is in.
- `git show <commit-hash>` shows exactly what changed in one specific
  commit (get the hash from `git log --oneline`).
