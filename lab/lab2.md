# Lab 2 - Model training and experiment tracking with MLflow

**Repo:** https://github.com/alialakbarr1/MLOps
**Tracking server:** `http://127.0.0.1:5000`, backend `sqlite:///mlflow.db`, artifacts `./mlruns`
**Experiment:** `food11` (experiment_id `1`)

All runs below use the `food11_processed_mini` dataset built in lab 1
(1100 training / 1096 validation / 1096 evaluation images, 11 classes), a
pretrained `resnet18` with an 11-way head, Adam, 5 epochs, `--seed 42`, on CPU.

---

## Question 1 - Look at `pyproject.toml` and `uv.lock`. What changed?

**`pyproject.toml`** gained the four libraries in `dependencies`:

```toml
dependencies = [
    "mlflow",
    "pillow>=12.3.0",
    "scikit-learn",
    "torch",
    "torchvision",
]
```

plus a block I added *before* running `uv add`, because this machine has an Intel
UHD GPU and no NVIDIA card, so the default CUDA wheels would have been a multi-GB
download that could never be used:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

`explicit = true` means that index is used *only* for packages that name it in
`[tool.uv.sources]`, so everything else still resolves from PyPI.

**`uv.lock`** is where the real change is:

```
 pyproject.toml      |   15 +
 src/food11/train.py |  205 +++++
 uv.lock             | 2468 ++++++++++++++++++++++++++++++++++++++++++++++++++-
```

It went from **2 locked packages to 101**. `pyproject.toml` records the *intent*
("I want mlflow"); `uv.lock` records the *resolution* - every transitive dependency
pinned to an exact version with its hashes and the index it came from. That is why
the lockfile is the file that makes the environment reproducible, and why both belong
in git.

The pin that proves the index override worked:

```
torch==2.14.0+cpu
torchvision==0.29.0+cpu
```

The `+cpu` local version label is the CPU-only build.

## Question 2 - What is `--backend-store-uri` for, what is `--default-artifact-root` for, metadata vs artifacts

`--backend-store-uri sqlite:///mlflow.db` is the **metadata store**: a real database
holding the structured, queryable record of every experiment and run. After the four
runs in this lab, `mlflow.db` is **948 KB**, with tables including:

```
experiments, runs, params, metrics, latest_metrics, tags,
experiment_tags, logged_models, logged_model_metrics, model_versions, ...
```

`--default-artifact-root ./mlruns` is the **artifact store**: a plain directory tree
for files that are too big or too opaque for a database - model weights, plots,
images, arbitrary output. After the same four runs, `mlruns/` is **228 MB**.

| | metadata (backend store) | artifacts (artifact root) |
|---|---|---|
| holds | params, metrics + their full step history, tags, run status, timings | model weights, `MLmodel`, env files, plots, any file you log |
| shape | relational rows | opaque blobs on a filesystem / S3 / etc. |
| size here | 948 KB | 228 MB |
| who reads it | the server, on every UI query | fetched by URI when you open or load an artifact |
| what it enables | sorting, filtering, comparing, the parallel-coordinates plot | `mlflow.pytorch.load_model(...)`, serving |

The split exists so the UI can sort 10 000 runs by `val_accuracy` with one SQL query
without ever touching a gigabyte of weights. The DB stores only a *URI* pointing at
the artifact.

## Question 3 - Why shouldn't `mlflow.db` and `mlruns/` be tracked by git, and why not by dvc either?

**Not git**, for three separate reasons:

1. `mlflow.db` is a **binary SQLite file that changes on every single run**. Git would
   store a full new copy each time and could never show a meaningful diff.
2. `mlruns/` holds **46 MB per logged model**. Four runs already put 228 MB in there.
   That is exactly the kind of payload git repositories are bad at.
3. They are **outputs, not sources**. The reproducible inputs are the code (git) and
   the data (dvc). Run history is produced *by* those two, so committing it duplicates
   derived state.

**Not dvc either**, which is the less obvious half:

- dvc tracks things you want **pinned to a commit** - "this code, with exactly this
  data". Run history is not an input to anything; nothing is reproduced *from* it.
- dvc versions **snapshots you explicitly `dvc add`**. Experiment tracking is
  continuous and append-only: every run would need another `dvc add` + `git commit`,
  and the pointer would be stale seconds later.
- **MLflow is already the system of record** for this data. Putting it under dvc means
  two tools owning the same state, with no gain.

The clean division of labour across the three tools:

| tool | owns |
|---|---|
| git | code, config, and pointers |
| dvc | datasets |
| mlflow | runs, params, metrics, models |

So `.gitignore` gained:

```
mlflow.db
mlruns/
```

## Question 4 - What happens the first time you call `set_experiment` with a name that doesn't exist?

**MLflow creates it silently** - no error, no flag needed. Before the first run the
server only had `Default`; after it, the experiments API returns:

```
0  Default  file:C:/Users/user/Desktop/uni-git/MLOps/mlruns/0
1  food11   file:C:/Users/user/Desktop/uni-git/MLOps/mlruns/1
```

`food11` was assigned `experiment_id = 1` and given its own artifact directory under
the artifact root, and it appears in the UI sidebar next to `Default`. `set_experiment`
is get-or-create: later runs with the same name attach to experiment `1` rather than
making a second one.

## Question 5 - `log_param` vs `log_metric`, and why only `log_metric` takes `step`

A **param** is an input, fixed before training and constant for the whole run. It is
written once, lands in the `params` table as a single row, and is immutable - logging a
different value for the same key in the same run is an error.

A **metric** is an output that is measured repeatedly. Each call appends a row of
`(value, timestamp, step)` to the `metrics` table, so a metric is a **time series**,
not a value. That is precisely what `step` is for: it is the x-axis. Without it MLflow
would only know "val_accuracy was 0.74 at some point", and could not draw a curve.

The history MLflow stored for the best run shows why this matters:

| step (epoch) | train_loss | val_loss | val_accuracy |
|---|---|---|---|
| 0 | 1.7839 | 1.1367 | 0.6505 |
| 1 | 0.6031 | 0.9210 | 0.7071 |
| 2 | 0.2655 | 0.8381 | 0.7381 |
| 3 | 0.1469 | 0.7917 | **0.7509** |
| 4 | 0.0963 | 0.7975 | 0.7436 |

A single final number would have hidden the most important fact in this table - see Q7.

`log_param` needs no `step` because a param has no history: the learning rate was
`0.0001` for the entire run, at every epoch.

(MLflow also keeps a `latest_metrics` table caching the most recent value per key, so
the runs list can sort by `val_accuracy` without scanning the full history.)

## Question 6 - Find the params, the metric charts and the model artifact. Where does the model live on disk?

In the UI: **Parameters** and **Metrics** panels on the run page, per-metric charts from
the Model metrics tab, and the model under the run's **Artifacts**.

On disk the answer is **not** what the lab text implies, because MLflow 3 changed it.
The model is *not* stored under the run's own artifact directory - that directory is
empty. It is a first-class **LoggedModel** entity with its own location:

```
mlruns/1/models/m-cb59fdb92f904408a2ae8d6169d075f0/artifacts/
├── MLmodel                      # flavour + signature + metadata
├── data/model.pt2               # the serialised weights/graph
├── input_example.json
├── serving_input_example.json
├── conda.yaml
├── python_env.yaml
└── requirements.txt
```

That is **46 MB per run**, and the entity records `source_run_id` pointing back at the
run that produced it (`888e2d8e…`). Path shape:
`<artifact-root>/<experiment_id>/models/m-<model_id>/artifacts/`.

### Two changes the lab's code needs on MLflow 3.16 + Windows

Both were real failures here, worth writing down:

1. **`mlflow.pytorch.log_model(model, "model")` fails outright.** MLflow 3 defaults to
   `serialization_format='pt2'`, a traced-graph format, which cannot save without a
   concrete example to trace `forward` with:

   ```
   MlflowException: If `serialization_format` is set to 'pt2', then input_example
   is required.
   ```

   Fix - hand it one real batch element, which doubles as the logged signature:

   ```python
   example_images, _ = next(iter(test_loader))
   mlflow.pytorch.log_model(model, "model", input_example=example_images[:1].cpu().numpy())
   ```

2. **The run never closes on a Windows console.** MLflow ends a run by printing
   `🏃 View run … `; stdout defaults to cp1252, which cannot encode that emoji, and the
   `UnicodeEncodeError` propagates out of `mlflow.end_run()`. Params, metrics and the
   model all logged fine, but the run was left stuck in `RUNNING`. Fix:

   ```python
   sys.stdout.reconfigure(encoding="utf-8", errors="replace")
   ```

   These two interact nastily: failure 2 fires while failure 1 is unwinding, so the
   traceback shows only the encoding error and hides the real cause. The tell was
   `LOGGED_MODEL_UPLOAD_FAILED` on the logged-models API.

## Question 7 - Which learning rate gave the best `val_accuracy`? Is higher always better?

Runs sorted by `val_accuracy` descending:

| lr | batch_size | val_accuracy | val_loss | train_loss | test_accuracy | secs | run_id |
|---|---|---|---|---|---|---|---|
| **0.0001** | 32 | **0.7436** | 0.7975 | 0.0963 | **0.7728** | 261 | `888e2d8e…` |
| 0.001 | 64 | 0.6597 | 1.2739 | 0.3423 | 0.6880 | 237 | `b54907f2…` |
| 0.001 | 32 | 0.5703 | 1.5536 | 0.6738 | 0.6204 | 291 | `8a17401b…` |
| 0.01 | 32 | 0.1487 | 2.4377 | 2.2897 | 0.1478 | 376 | `13ce3720…` |

**`lr = 0.0001` is clearly best**, and the ordering is monotone over the range tested:
`0.0001 > 0.001 > 0.01`.

**Higher is not better - and the failure is dramatic.** At `lr = 0.01` the run does not
merely underperform, it **diverges**: `train_loss` ends at 2.2897, essentially where it
started, and 0.1487 accuracy on 11 balanced classes is barely above the 1/11 = 0.0909
chance line. The steps are so large that fine-tuning destroys the pretrained features
instead of adapting them.

Two caveats worth stating:

- **Lower is not unconditionally better either.** This is a U-shape, and we only sampled
  the right-hand slope. Push to `1e-6` and 5 epochs would not be enough to converge -
  underfitting instead of divergence.
- **The best epoch was not the last one.** In the winning run `val_accuracy` peaked at
  epoch 3 (**0.7509**) and fell to 0.7436 by epoch 4, while `val_loss` ticked back up
  (0.7917 → 0.7975) even as `train_loss` kept dropping (0.1469 → 0.0963). That is
  overfitting beginning. Reading only the final number costs ~0.7 points, which is
  exactly the mistake `step`-wise logging exists to prevent.

## Question 8 - Parallel coordinates on `lr`, `batch_size` and `val_accuracy`

The plot separates on one axis and only one: **`lr` dominates, `batch_size` is
second-order.**

- Every line entering the `lr` axis at **0.01** drops to the bottom of the
  `val_accuracy` axis. The 0.01 line is visually isolated from the other three -
  a cliff, not a gradient.
- Lines entering at **0.0001** land at the top. The `lr` axis alone almost fully
  orders the outcome.
- The **`batch_size`** axis does *not* separate cleanly: its lines cross. Both 32 and
  64 appear among the good runs, so no ordering can be read off that axis by itself.

`batch_size` only becomes legible once `lr` is held constant. At `lr = 0.001`:

| batch_size | val_accuracy |
|---|---|
| 32 | 0.5703 |
| 64 | **0.6597** |

so 64 was better *here* - with the caveat that batch size and learning rate are coupled
(a bigger batch gives less noisy gradients per step but fewer steps per epoch: 18/epoch
at bs=64 versus 35 at bs=32), and one pair of runs at one learning rate is not enough to
call that a general rule.

The practical reading: **spend your tuning budget on `lr` first.** Nothing you do to
`batch_size` rescues `lr = 0.01`.

## Question 9 - Sort by `val_accuracy` descending. Which run is best? Note its run ID.

Best run:

```
run_id       888e2d8ef879462ba98253b7c54d9a73
params       lr=0.0001  batch_size=32  epochs=5  dataset=mini
             architecture=resnet18  pretrained=True  optimizer=adam  seed=42
metrics      val_accuracy=0.7436   val_loss=0.7975
             test_accuracy=0.7728  test_loss=0.6790
             train_loss=0.0963     training_seconds=261
model        m-cb59fdb92f904408a2ae8d6169d075f0  (LOGGED_MODEL_READY)
```

**Run ID for lab 3: `888e2d8ef879462ba98253b7c54d9a73`**

Note that `test_accuracy` (0.7728) is *higher* than `val_accuracy` (0.7436). The
evaluation split is simply a slightly easier sample - it was never used for any
decision, so this is not leakage.

---

## Commands used

```bash
# environment (CPU-only torch index added to pyproject.toml first)
uv add mlflow torch torchvision scikit-learn

# tracking server, left running in its own terminal
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# keep local run outputs out of git
echo "mlflow.db" >> .gitignore
echo "mlruns/"   >> .gitignore
git add .gitignore && git commit -m "Ignore local mlflow tracking files"

# the sweep, one run at a time
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.01   --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001  --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001  --batch-size 64

git add src/food11/train.py pyproject.toml uv.lock
git commit -m "Add training script with mlflow tracking"
git push
```

## The training script

`src/food11/train.py`:

- loads the three splits with `ImageFolder` + `DataLoader`, normalised with the ImageNet
  statistics `resnet18` was pretrained on, with a horizontal flip on the training split only
- takes `resnet18(weights=ResNet18_Weights.DEFAULT)` and replaces `model.fc` with
  `nn.Linear(model.fc.in_features, 11)`
- takes `--dataset {processed,mini}`, `--epochs`, `--lr`, `--batch-size`, plus
  `--num-workers` and `--seed`
- inside `with mlflow.start_run():` logs all hyperparameters once with `log_params`,
  then `train_loss` / `val_loss` / `val_accuracy` with `log_metric(..., step=epoch)`
  after every epoch, and finally `test_accuracy`, `test_loss`, `training_seconds` and
  the model itself

`training` fits the weights, `validation` is scored every epoch, and `evaluation` is
held out and scored exactly once at the end.
