# Lab 3 - Containerizing the model with Docker

**Repo:** https://github.com/alialakbarr1/MLOps
**Registered model:** `food11`, alias `champion` → **v2**
**Image:** `food11-api:latest` (multi-stage), `food11-api:naive` (single-stage, for comparison only)

Everything below was run against the lab 2 tracking server
(`sqlite:///mlflow.db`, artifacts under `./mlruns`) on Docker 27.5.1 / Docker
Desktop on Windows.

---

## Question 1 - What version number was your model given? Run artifact vs registered model

The best run from lab 2 (`888e2d8ef879462ba98253b7c54d9a73`, `val_accuracy` 0.7436)
registered as **version 1**:

```
Successfully registered model 'food11'.
Created version '1' of model 'food11'.
```

Registration printed a warning that is worth reading closely:

```
Run with id 888e2d8e... has no artifacts at artifact path 'model',
registering model based on models:/m-cb59fdb92f904408a2ae8d6169d075f0 instead
```

That is the MLflow 3 LoggedModel indirection from lab 2 showing through: `runs:/<id>/model`
no longer resolves to anything, and MLflow silently followed the link to the real
LoggedModel entity. The lab's `runs:/<run-id>/model` URI still works, but only because
of that fallback.

A **second version** was created later in this lab for a reason that had nothing to do
with the weights - see Q7 - so the registry now holds:

```
v2  source=models:/m-a30cca875cc1477eac14ab8ec7fdc083  run=48a1d7a3   <- champion
v1  source=models:/m-cb59fdb92f904408a2ae8d6169d075f0  run=888e2d8e
```

Both versions are the same trained weights.

### Logged model artifact vs registered model

| | logged model artifact | registered model |
|---|---|---|
| created by | `mlflow.pytorch.log_model(...)` during a run | `mlflow.register_model(...)`, a deliberate promotion |
| identity | tied to the run that produced it | a **name** (`food11`) that outlives any run |
| how many | one per `log_model` call, thousands across an experiment | a curated handful of versions |
| addressed as | `models:/m-<model_id>` | `models:/food11/2`, or `models:/food11@champion` |
| meaning | "this is what that experiment produced" | "this is a candidate we might serve" |

The important practical difference: the registered name is a **stable address**. The
serving container asks for `models:/food11@champion` and never learns which run,
version or file path is behind it. Registration is the boundary between
experimentation and deployment.

## Question 2 - Which aliases replaced stages? Why version separately from the run, why are aliases more flexible?

The deprecated built-in stages were `None`, `Staging`, `Production` and `Archived` - a
fixed, closed set with one model per stage. They are replaced by **aliases**:
user-defined, mutable, named pointers to a specific version. There is no built-in list;
the conventional ones are `champion` (currently serving) and `challenger` (being
evaluated against it), plus anything else a team wants - `baseline`, `canary`, `eu-prod`.
Free-form **tags** carry the remaining metadata.

Here `champion` points at v2:

```
aliases on registered model: {'champion': '2'}
champion -> v2 (run 48a1d7a3)
```

**Why version separately from the run?** A run is an immutable historical fact - this
code, this data, these metrics, at this time. A model version is a *decision* about
that artifact. Separating them means:

- the run stays reproducible and untouched no matter what you later decide
- one run can yield several versions (re-packaged, re-serialised - exactly what
  happened here between v1 and v2)
- versions can come from entirely different runs, experiments, or even be imported
  from outside, yet still live under one name
- consumers depend on the name, not on a run ID that means nothing to them

**Why is an alias more flexible than a fixed stage?**

1. **Promotion is a pointer move, not a state change.** Reassigning `champion` from v2
   to v3 is one API call. Nothing about v2 changes; it stays available for instant
   rollback.
2. **Not limited to one per stage.** You can have `champion`, `challenger`,
   `canary-eu` and `shadow` simultaneously, which is how real A/B and canary rollouts
   work. The old stages could not express that.
3. **Names fit your workflow**, rather than forcing every team into the same three words.
4. **Deployments stop caring about version numbers.** The container in this lab hardcodes
   `models:/food11@champion`. Promoting a new model needs no rebuild, no redeploy and no
   config change - only a restart, as Q8 shows.

## Question 3 - Why load through a model URI instead of a `.pth` on disk? What changes to serve a newer version?

`serve.py` loads:

```python
mlflow.pyfunc.load_model("models:/food11@champion")
```

Pointing at a `.pth` would be worse in five distinct ways:

1. **A `.pth` is only weights.** It has no architecture, so the serving code would have
   to rebuild `resnet18` with an 11-way head and stay in sync with training forever. The
   MLflow model bundles the flavour, the signature, the input example and the
   environment (`MLmodel`, `conda.yaml`, `python_env.yaml`, `requirements.txt`).
2. **The path is host-specific.** The real path here is
   `mlruns/1/models/m-cb59fdb92f904408a2ae8d6169d075f0/artifacts/data/model.pt2` -
   meaningless inside a container, on a CI runner, or on a teammate's laptop.
3. **No provenance.** A filename cannot tell you which run, metrics or data produced it.
   The registry keeps that link.
4. **No promotion or rollback story.** With a path baked into the image, shipping a new
   model means rebuilding and redeploying the image. With an alias it is a registry
   operation.
5. **`pyfunc` gives a uniform interface.** If the model were later replaced by
   scikit-learn or ONNX, `predict()` would not change.

**To serve a newer version: move the alias, restart the container.**

```python
client.set_registered_model_alias("food11", "champion", 3)
```

```bash
docker restart food11
```

No rebuild, no new image, no code edit, no environment variable change. Rolling back is
the same call with the old version number. The only reason a restart is needed at all is
that `serve.py` resolves the alias once in its `lifespan` startup hook - the deliberate
trade of picking up promotions instantly for not doing a registry lookup on every
request.

## Question 4 - Why `COPY pyproject.toml`/`uv.lock` and `uv sync` before the source? What happens to the cache when `serve.py` changes?

Docker caches per instruction and invalidates **every layer after the first one that
changes**. A layer's cache key for `COPY` is the content of the files copied.

Dependencies change rarely; source changes constantly. Copying the two manifests first
means the expensive `uv sync` layer keys only on `pyproject.toml` + `uv.lock`:

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
```

**Editing a line in `serve.py`** invalidates only the `COPY src ./src` layer and the
trivial metadata layers after it. Everything above is reused - observed in the rebuild
after I changed the Dockerfile's user handling:

```
#13 [builder 4/5] COPY pyproject.toml uv.lock ./
#13 CACHED
```

That is a **45 kB** layer rebuilt instead of a **1.76 GB** dependency install: seconds
instead of ~8 minutes, and no network at all.

Invert the order - `COPY . .` before `uv sync`, as `Dockerfile.naive` deliberately does -
and every source edit changes the context, so `uv sync` re-runs and re-downloads torch
every single time.

`--no-install-project` matters too: without it `uv sync` builds the local `mlops`
package, which needs `src/` present, which would force the source copy to come first and
defeat the whole layout. The app runs from `src/` and is never imported as the installed
package, so skipping it is free.

## Question 5 - Size difference between naive single-stage and multi-stage

| image | size |
|---|---|
| `food11-api:naive` (single-stage) | **3.6 GB** |
| `food11-api:latest` (multi-stage, first attempt) | 4.66 GB |
| `food11-api:latest` (multi-stage, after the fix below) | **2.42 GB** |

Final answer: **the multi-stage image is 2.42 GB vs 3.6 GB naive - 1.18 GB smaller,
about 33%.**

But the first multi-stage attempt came out *bigger* than the naive one, which was not
the expected result and was worth chasing. `docker history` found the cause immediately:

```
1.76GB  RUN /bin/sh -c useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
1.76GB  COPY /app/.venv /app/.venv
```

`chown -R` **rewrites the metadata of every file in the venv**, and Docker's copy-on-write
layering stores the entire modified tree as a second full-size layer. One convenience
line silently doubled the image. The `chown` was also pointless: the venv is read-only
at runtime and root-owned files are world-readable. Fix - create the user *before* the
copy and drop the `chown` entirely:

```dockerfile
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
USER appuser
```

Layer-by-layer comparison from `docker history`:

| layer | naive | multi-stage (fixed) |
|---|---|---|
| base image | **2.04 GB** (`python:3.13`) | **133 MB** (`python:3.13-slim`) |
| `pip install uv` | 82.9 MB | 0 - uv stays in the builder |
| dependency install | 1.48 GB | 1.76 GB (`COPY /app/.venv`) |
| source | 573 kB (`COPY . .`) | 45.1 kB (`COPY src`) |
| user creation | - | 69.6 kB (was 1.76 GB with `chown -R`) |
| **total** | **3.6 GB** | **2.42 GB** |

Where the multi-stage win actually comes from, in order:

1. **~1.9 GB** from `python:3.13-slim` instead of `python:3.13` - the full base carries
   gcc, make, headers and git that only the builder ever needs.
2. **83 MB** from leaving `uv` behind in the builder stage.
3. The uv download cache is a `--mount=type=cache`, so it never enters any layer.

The dominant remaining cost is torch itself (~1.5-1.8 GB), which no amount of staging
removes - it is the actual dependency. A genuinely small image would need a
`torch`-free runtime (ONNX Runtime, or TorchServe's slim base).

> Note: `docker images` reports uncompressed on-disk size. Pushed-to-registry size is
> smaller because layers are gzipped.

## Question 6 - What happens without a `.dockerignore`? Which excluded folders would break the build?

**Build speed.** The whole context is tarred and streamed to the daemon *before the
first instruction runs*. Measured sizes in this repo:

| path | size |
|---|---|
| `data/` | 1.3 GB |
| `.venv/` | 1.1 GB |
| `mlruns/` | 228 MB |
| `.git/` | 365 KB |
| `src/` | **29 KB** |

Without `.dockerignore` the context is **~2.6 GB**; with it, roughly **100 KB** - about
a 25,000x reduction, paid on *every* build, including ones that hit the cache for every
layer.

**Image size.** The naive `COPY . .` would bake all 2.6 GB into a layer.

**Cache.** Worse than either: the context hash would then include `mlruns/` and
`mlflow.db`, which change on every training run, so an unrelated experiment would
invalidate the Docker cache and force a full torch reinstall.

### Which would actually break the build

- **`.venv/`** - the real breaker. It holds Windows executables, `Scripts/` rather than
  `bin/`, and absolute host paths in `pyvenv.cfg`. Copied into a Linux image over
  `/app/.venv`, `PATH` would find a `uvicorn` that cannot execute. Not a size problem,
  a correctness one.
- **`data/`, `mlruns/`, `.git/`** - wasteful, not fatal.

### The inverse trap, which did break the build here

`.dockerignore` is equally dangerous in the other direction. I had excluded `README.md`
as "docs, not needed to run the service". The naive build then failed:

```
error: Failed to build `mlops @ file:///app`
  cause: failed to open file `/app/README.md`: No such file or directory (os error 2)
```

`pyproject.toml` declares `readme = "README.md"`, so the package metadata cannot be
built without it. The multi-stage build was immune only because `--no-install-project`
skips building the project at all. **Anything `pyproject.toml` references is a build
input, regardless of whether it looks like documentation.** `README.md` is now
explicitly kept, with a comment saying why.

## Question 7 - Why can't the container use `127.0.0.1:5000`? What does `host.docker.internal` resolve to?

`127.0.0.1` is **per network namespace**. Each container gets its own loopback, so inside
the container `127.0.0.1:5000` means "port 5000 of *this container*", where nothing is
listening. The host's loopback is a different, unreachable interface.

`host.docker.internal` is a DNS name Docker Desktop injects into the container's resolver
that resolves to the host's address **as seen from the container network** - on Docker
Desktop, the gateway IP of the VM's virtual switch, not a real public address. On Linux
there is no Docker Desktop VM, hence the lab's separate `--network host` instruction,
which puts the container directly in the host's namespace so `127.0.0.1` genuinely is
the host.

Run command used (Windows):

```bash
docker run -d --name food11 -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

### Two failures that DNS alone did not solve

Reaching the server was necessary but not sufficient. Both of these are worth recording
because the lab text does not mention either.

**1. MLflow rejected the request with HTTP 403.**

```
API request to endpoint /api/2.0/mlflow/registered-models/alias failed with
error code 403 != 200. Response body:
'Invalid Host header - possible DNS rebinding attack detected'
```

MLflow's security middleware validates the `Host` header, and the container sends
`host.docker.internal:5000`, which is not in the default allow-list. The fix is
`--allowed-hosts` - **but the flag replaces the permissive default with exact string
matching that ignores ports**, so a first attempt with
`--allowed-hosts 'localhost,127.0.0.1,host.docker.internal'` started 403ing my own
browser too. Entries must be listed with their ports:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts --artifacts-destination ./mlruns \
  --allowed-hosts 'localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,host.docker.internal,host.docker.internal:5000'
```

**2. The artifacts were unreachable even once the API answered.**

```
mlflow.exceptions.MlflowException: No such artifact: ''
```

The registry recorded v1's location as a **host filesystem path**,
`file:C:/Users/user/Desktop/uni-git/MLOps/mlruns/1/models/...`. MLflow clients download
artifacts *directly from that URI*, not through the tracking server, so the container
tried to open a Windows path that does not exist inside it. The tracking API is proxied;
artifact storage is not, unless you ask for it.

Adding `--serve-artifacts` was not enough on its own: **an experiment's artifact location
is fixed when the experiment is created**, and `food11` predated the flag. The fix was a
new experiment created under proxied storage, and re-registering the same weights there:

```
food11-serving  artifact_location: mlflow-artifacts:/2
new model       mlflow-artifacts:/2/models/m-a30cca875cc1477eac14ab8ec7fdc083/artifacts
registered      food11 v2, alias champion -> v2
```

`mlflow-artifacts:` makes the tracking server proxy the bytes over HTTP, so any client
that can reach the API can fetch the model. This is the generalisable lesson: **a local
`file:` artifact root works only while every client shares the filesystem.** The moment
serving moves into a container it stops working, and the same applies to CI runners and
Kubernetes pods.

With both fixed, the container serves:

```
health: {"status":"ok"}
model:  {"model_uri":"models:/food11@champion","tracking_uri":"http://host.docker.internal:5000"}
```

and predictions match the pre-container local API to three decimals:

| image | local API | container |
|---|---|---|
| validation/Bread | Fried food 0.738 | Fried food 0.738 |
| validation/Soup | Soup 1.000 | Soup 1.000 |
| validation/Rice | Egg 0.372 | Egg 0.372 |
| validation/Dessert | Dessert 0.519 | Dessert 0.519 |

(3 of 4 correct is expected from a 74%-accuracy model; scored over 110 validation
images through the API the service gets **0.7727**, matching training.)

## Question 8 - Stop the container, start a new one from the same image. Does the model still load?

**Yes, with no rebuild.** Same image ID, fresh container:

```
$ docker stop food11 && docker rm food11
$ docker images food11-api:latest --format '{{.ID}} {{.Size}}'
508105e969c3 2.42GB
$ docker run -d --name food11b -p 8000:8000 \
    -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
$ curl http://127.0.0.1:8000/health
{"status":"ok"}
$ # predict on validation/Soup
Soup conf=1.000
```

### What that says about baked-in versus fetched-at-runtime

| baked into the image | fetched at runtime |
|---|---|
| Python 3.13 runtime and OS libraries | the model weights |
| the locked dependency tree (torch, mlflow, fastapi…) | which *version* `champion` resolves to |
| `src/food11/serve.py` | the tracking server's address (env var) |
| the default `MLFLOW_TRACKING_URI` | |

The image contains **no model at all** - `.dockerignore` excludes `mlruns/`, and nothing
copies weights in. Each container start resolves `models:/food11@champion` and downloads
it fresh. Three consequences:

1. **The image is stateless and interchangeable.** Containers are disposable; scaling to
   ten replicas gives ten identical ones.
2. **Promotion needs no rebuild** - exactly the Q3 property, now demonstrated. Move the
   alias, restart, new model.
3. **The tracking server is a hard startup dependency.** If it is unreachable the
   container exits rather than serving stale weights - as seen repeatedly above, exit
   code 3. That is safe but not resilient; production would either bake a known-good
   model in as a fallback or keep a warm local cache.

## Question 9 - What's still missing before CI or Kubernetes could pull and run this exact image?

The Dockerfile is in git; the image exists only in this laptop's local Docker daemon.
Gaps, roughly in order of how badly each one bites:

1. **A registry.** Nothing can `docker pull food11-api:latest` - it was never pushed.
   It needs to live in GHCR, Docker Hub, ECR or similar.
2. **An immutable tag.** `latest` is a mutable pointer; two pulls a day apart can differ.
   Deployments should reference an immutable identifier - ideally the digest
   (`food11-api@sha256:…`), or at minimum a tag derived from the git SHA.
3. **Reachable infrastructure.** The image hardcodes
   `MLFLOW_TRACKING_URI=http://host.docker.internal:5000`, which is meaningless in a
   cluster. It needs a real tracking server URL, and - per Q7 - that server must serve
   artifacts over `mlflow-artifacts:` rather than local `file:` paths, or use shared
   object storage (S3/GCS/Azure) both sides can read.
4. **Credentials.** A shared tracking server and object store need auth: tokens or
   service accounts, injected as secrets, never baked into the image.
5. **A build pipeline.** The image should be built by CI from a known commit, not from a
   laptop with uncommitted changes. That also gives reproducibility: the same Dockerfile
   + `uv.lock` + git SHA.
6. **Multi-architecture builds.** This is `linux/amd64`. An arm64 node or an Apple
   Silicon laptop would need a `buildx` manifest list.
7. **Health probes and resource limits.** `/health` exists but nothing declares it as a
   readiness probe, and torch on CPU needs explicit CPU/memory requests. Kubernetes would
   otherwise route traffic to a pod still downloading the model, or OOM-kill it.
8. **Provenance.** No OCI labels linking image → git SHA → model version, and no SBOM or
   signature. When a prediction looks wrong six months from now, nothing ties the running
   container back to a commit and a run.

The theme: git versions the *recipe*, and the image is the *build output*. Until that
output is pushed, immutably addressed and reproducibly rebuildable, "it works on my
machine" is still the only guarantee.

---

## Commands used

```bash
# register and alias
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.register_model('runs:/888e2d8ef879462ba98253b7c54d9a73/model', 'food11')"

uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.MlflowClient().set_registered_model_alias('food11', 'champion', 2)"

# serving deps and a local smoke test before containerizing
uv add fastapi uvicorn python-multipart
uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
curl -X POST -F "file=@data/food11_processed_mini/validation/Soup/9_0.jpg" \
  http://127.0.0.1:8000/predict

# build and compare
docker build -t food11-api:latest .
docker build -f Dockerfile.naive -t food11-api:naive .
docker images food11-api
docker history food11-api:latest

# run (Windows / Docker Desktop)
docker run -d --name food11 -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
docker logs -f food11

git add Dockerfile Dockerfile.naive .dockerignore src/food11/serve.py pyproject.toml uv.lock
git commit -m "Containerize model serving with Docker"
git push
```

## Files added

- **`src/food11/serve.py`** - FastAPI app. `GET /health`, `GET /model` (reports the
  resolved URI, handy after a promotion) and `POST /predict` (multipart upload →
  category, confidence, top-3). The model loads once in a `lifespan` startup hook from
  `models:/food11@champion`; `MLFLOW_TRACKING_URI` and `MODEL_URI` are both
  environment-overridable. Preprocessing mirrors training exactly: RGB, 128x128
  bilinear, ImageNet mean/std, NCHW float32.
  One subtlety: `ImageFolder` assigns class indices by **sorted folder name**, so the
  category list in `serve.py` must stay in sorted order. For Food-11 the sorted order
  happens to equal the original numeric labels, which makes an easy bug easy to miss.
- **`Dockerfile`** - two-stage build described above.
- **`Dockerfile.naive`** - deliberately bad single-stage build, kept only as the Q5
  measurement. Not for deployment.
- **`.dockerignore`** - excludes `.venv/`, `data/`, `mlruns/`, `mlflow.db`, `.git/`,
  caches and `lab/`; deliberately **keeps** `README.md`.
