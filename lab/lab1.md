# Lab 1 - git/dvc and data preparation

**Repo:** https://github.com/alialakbarr1/MLOps

## Solution adopted for the data remote

I used **solution 1: a local folder as the dvc remote** instead of DagsHub.
The remote points to a directory that lives completely outside the git repo:

```bash
dvc remote add --default origin C:/Users/user/dvcstore
```

`dvc push` / `dvc pull` copy the cache objects to and from that folder exactly the
way they would with an HTTP remote, so the whole "git holds the pointer, dvc holds
the bytes" behaviour is preserved without any credentials or upload quota.

---

## Question 1 - Observe the files created by `uv init`, what do you think they contain

| File | What it holds |
|---|---|
| `pyproject.toml` | The project manifest: name, version, description, authors, `requires-python = ">=3.13"`, the `dependencies` list, the `[project.scripts]` entry point and the build backend (`uv_build`). This is the file `uv add` edits. |
| `.python-version` | A single line, `3.13`. It pins the interpreter uv picks for this project so everyone on the team resolves against the same Python. |
| `README.md` | Empty placeholder, referenced by `readme = "README.md"` in the manifest. |
| `src/mlops/__init__.py` | The package skeleton (src-layout). It contains a `main()` that prints `Hello from mlops!`, which is what the `mlops` console script points at. |

Two more files appear the first time you run `uv add` / `uv run`:

- `uv.lock` - the fully resolved, cross-platform lock file (exact versions + hashes). **Commit it**, it is what makes the environment reproducible.
- `.venv/` - the actual virtual environment. **Not committed** - uv drops a `.venv/.gitignore` containing `*` so git ignores it automatically.

## Question 2 - What are the files created by `dvc init`, what are they for, which ones go to git

`dvc init` creates:

```
.dvc/.gitignore
.dvc/config
.dvc/tmp/          (+ .dvc/cache/ once data is added)
.dvcignore
```

| Path | Purpose | Push to git? |
|---|---|---|
| `.dvc/config` | The project-level dvc configuration (remote urls, cache settings). Shared with the team. | **Yes** |
| `.dvc/.gitignore` | Written by dvc itself; it ignores `/config.local`, `/tmp` and `/cache`. | **Yes** |
| `.dvcignore` | Same idea as `.gitignore`, but tells dvc which paths to skip when it walks the workspace (speeds up hashing). Starts as comments only. | **Yes** |
| `.dvc/cache/` | The local content-addressable store - the real file contents, keyed by md5. This is the data, it must never go to git. | No (already ignored) |
| `.dvc/tmp/` | State DB, lock files, run cache. Machine-local scratch. | No (already ignored) |
| `.dvc/config.local` | Created later by `dvc config --local`; this is where **secrets** belong. | No (already ignored) |

So `git add .dvc .dvcignore` is safe precisely because dvc pre-wrote a `.gitignore`
that keeps the cache, tmp and local config out.

## Question 3 - Where are the credentials stored, what are the options other than `--global`, should they be pushed to github

`dvc remote modify origin --global ...` writes to the **user-level config file**,
outside the repository. On this machine that is:

```
C:\Users\user\AppData\Local\iterative\dvc\config
```

(on Linux it is `~/.config/dvc/config`). Because it sits outside the repo, git never sees it.

The four scopes accepted by `dvc config` / `dvc remote modify`:

| Flag | File | Scope |
|---|---|---|
| `--system` | machine-wide dvc config dir | every user on the machine |
| `--global` | user config dir (`%LOCALAPPDATA%\iterative\dvc\config`) | this user, all repos |
| `--project` (default) | `.dvc/config` | this repo, **tracked by git** |
| `--local` | `.dvc/config.local` | this repo, **gitignored** |

**Credentials must never be pushed to GitHub.** Put the remote *url* in `--project`
(`.dvc/config`) so teammates inherit it, and put the user / password / token in
`--local` or `--global`. `--local` is usually the better fit for a per-project
token: it stays next to the project but is already listed in `.dvc/.gitignore`.

You can check what would leak with:

```bash
dvc config --list --show-origin
```

## Question 4 - Take a look at the `.gitignore` file, explain what happened

`dvc add data` appended a `/data` line to the repository `.gitignore`:

```
.venv/
__pycache__/
*.pyc
/data          <-- added by dvc add
```

dvc took ownership of the `data` directory and told git to stop looking at it, so
the two tools cannot fight over the same files. From now on git tracks only the
small `data.dvc` pointer while dvc tracks the contents. If the folder had already
been committed to git, `dvc add` refuses:

```
ERROR: output 'data' is already tracked by SCM (e.g. Git).
    git rm -r --cached 'data'
```

Git and dvc are not allowed to track the same output.

## Question 5 - Do you see a `.dvc` file? What does it contain?

Yes, `data.dvc` in the repo root. It is a small YAML pointer:

```yaml
outs:
- md5: a3a457d03c51ff8b037a833440f6ad13.dir
  size: 1188442712
  nfiles: 16643
  hash: md5
  path: data
```

- `path` - the workspace path it stands for
- `md5` ending in `.dir` - the hash of a *directory listing object*, not of a single file
- `size` / `nfiles` - total bytes and file count, used for progress and for `dvc status`

The `.dir` object itself lives in the cache / remote and is a JSON array mapping
every file to its own md5:

```json
[{"md5": "835878c085ad8f08ea6cdd959fe67463", "relpath": "food11_raw/evaluation/0_0.jpg"},
 {"md5": "0b2cc3493d8f057449223ce5cd036b71", "relpath": "food11_raw/evaluation/0_1.jpg"}, ...]
```

That indirection is what lets one ~120 byte pointer in git represent 1.2 GB of images.

After `data.py` ran and the processed datasets were added, the same file became:

```yaml
outs:
- md5: 311eb5c8da889ca29c6cd6ffb2e2a4c7.dir
  size: 1316401637
  nfiles: 36578
  hash: md5
  path: data
```

Same four lines, a different directory hash - 36578 files instead of 16643, because
`food11_processed` and `food11_processed_mini` now sit beside `food11_raw`. That one
changed hash is the entire diff git sees for a 128 MB change on disk.

## Question 6 - What is on the GitHub main branch, and what is on the dvc remote?

On GitHub (`git ls-tree -r --name-only origin/main`):

```
.dvc/.gitignore
.dvc/config
.dvcignore
.gitignore
data.dvc
pyproject.toml
uv.lock
src/food11/data.py
src/mlops/__init__.py
```

- **Code: yes.**
- **Data: no.** Not a single image.
- **A file that points to the data: yes** - `data.dvc`, plus `.dvc/config` which says
  *where* the remote is. Together they are enough to locate and restore the exact dataset.

On the dvc remote you see the mirror image: no source code, just the
content-addressed cache, one file per md5:

```
dvcstore/files/md5/a3/a457d03c51ff8b037a833440f6ad13.dir
dvcstore/files/md5/00/009cbd0012549df615ac915fbec3ba
dvcstore/files/md5/00/0246d03ce4222d0ed8aebe0a087c2c
dvcstore/files/md5/00/02ae8b30b5199ef73aacfe6cb93877
...
```

The file names are hashes, not `0_0.jpg`, and the first two characters become the
directory, which is what keeps any single folder from holding tens of thousands of
entries.

Storing by hash also deduplicates. Pushing the raw dataset reported:

```
16021 files pushed
```

for a folder holding **16643** images - about 620 of them are byte-identical
duplicates somewhere in Food-11, so they are stored once and simply referenced twice
in the `.dir` listing. On DagsHub the "Data" tab does the reverse mapping for you and
shows the human-readable tree instead of the hash names.

## Question 7 - Clone the repo in a fresh folder. Is the data folder there? What command brings it back?

After `git clone`, the working tree contains:

```
.dvc/  .dvcignore  .gitignore  data.dvc  pyproject.toml  src/  uv.lock
```

There is **no `data/` folder** - `ls data` fails. `dvc status` says:

```
data.dvc:
        changed outs:
                not in cache:       data
```

The command that materialises it is:

```bash
dvc pull
```

which fetches the objects named by `data.dvc` from the remote into `.dvc/cache` and
then links them into `data/`:

```
A       data\
7 files fetched and 6 files added
```

(`dvc fetch` only fills the cache, `dvc checkout` only links cache -> workspace,
`dvc pull` is the two of them together.)

## Question 8 - After checking out an old commit, do you still see `food11_processed` and `food11_processed_mini`?

The commits touching the pointer:

```bash
$ git log --oneline -- data.dvc
964e0b3 Add food11_processed and food11_processed_mini
5b5d356 Track data folder with dvc
```

Two steps, and the order matters:

1. `git checkout 5b5d356` alone - the folders are **still on disk**. Git rewound
   `data.dvc`, but git does not manage `data/` at all, so the workspace is now out of
   sync with the pointer. `dvc status` reports `modified: data`.
2. `dvc checkout` - **now the two processed folders are gone**, only
   `data/food11_raw` remains, which is exactly the state that old pointer describes.

```
after git checkout only:   data/food11_processed  data/food11_processed_mini  data/food11_raw
after dvc checkout:        data/food11_raw
```

Going back with `git checkout main && dvc checkout` restores all three folders
instantly - the content was never deleted, it was still sitting in `.dvc/cache`, so
the second checkout is just a relink with no download.

**Takeaway:** a git commit hash now identifies *code + data* together, but you have
to move both tools. `git checkout` without `dvc checkout` leaves you running new
code against old data.

---

## Commands used, in order

```bash
# project
uv init
uv add pillow

# dvc
dvc init
git add .dvc .dvcignore && git commit -m "Initialize git and dvc"

# local folder remote, outside the repo
mkdir -p /c/Users/user/dvcstore
dvc remote add --default origin C:/Users/user/dvcstore
git add .dvc/config && git commit -m "Configure local folder as dvc remote"

# raw data
dvc add data
git add data.dvc .gitignore && git commit -m "Track data folder with dvc"
git push
dvc push

# processing
uv run python ./src/food11/data.py
dvc add data
git add data.dvc && git commit -m "Add food11_processed and food11_processed_mini"
git push
dvc push

# time travel
git log --oneline -- data.dvc
git checkout <old-commit-hash> && dvc checkout
git checkout main && dvc checkout
```

## The preparation script

`src/food11/data.py` reads `data/food11_raw/{training,evaluation,validation}`, where
the label is the part of the file name before the underscore, and writes two
`torchvision.datasets.ImageFolder`-shaped trees - the layout ResNet expects, one
sub-folder per class:

```
data/food11_processed/training/Bread/0_0.jpg
data/food11_processed/training/Dairy product/1_0.jpg
...
data/food11_processed_mini/...      same, capped at 100 images per class per split
```

Every image is converted to RGB and resized to **128x128** before being written back
as JPEG. Both output trees are rebuilt from scratch on each run, so a rerun never
leaves stale files behind.

```bash
uv run python ./src/food11/data.py
```
