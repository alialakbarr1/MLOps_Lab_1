# Lab 4 - Orchestrating the app with Docker Compose

**Repo:** https://github.com/alialakbarr1/MLOps_Lab_1
**Stack:** `mlflow` (tracking + registry) · `inference` (lab 3 API) · `frontend` (Streamlit)
**Versions:** Docker 27.5.1, Docker Compose v2.32.4

```
mlflow      Up (healthy)   0.0.0.0:5000->5000/tcp
inference   Up (healthy)   8000/tcp
frontend    Up             0.0.0.0:8501->8501/tcp
```

Two deviations from the lab text, both deliberate:

- **Aliases, not stages.** The lab says to move a version to `Staging`, but registry
  stages are deprecated and lab 3 already put the model behind
  `models:/food11@champion`. Everything below uses that alias.
- **The mlflow image needs two extra flags.** The Dockerfile as given in the lab cannot
  actually serve this stack - see Q3's note and Q7.

---

## Question 1 - What happens to `/mlflow-data` with no volume mounted?

**Everything written there lives in the container's writable layer and dies with the
container.** Tested directly with no volume:

```
$ docker run -d --name q1test -p 5000:5000 mlops-mlflow
$ curl -X POST .../experiments/create -d '{"name":"written-without-a-volume"}'
{"experiment_id": "1"}

$ curl .../experiments/search
    1 written-without-a-volume
    0 Default

$ docker exec q1test ls -l /mlflow-data/
-rw-r--r-- 1 root root 876544 Oct 13 09:16 mlflow.db      <- 876 KB, really written
```

Then remove the container and start a fresh one from the **same image**:

```
$ docker rm -f q1test
$ docker run -d --name q1test2 -p 5000:5000 mlops-mlflow
$ curl .../experiments/search
    0 Default                                              <- the experiment is gone
```

The data was genuinely written - it is not that the path was read-only or missing. A
container's writable layer is a real filesystem, it is just **bound to the lifetime of
that container**. `docker stop`/`start` on the *same* container would keep it; `docker
rm` destroys it. Since Compose recreates containers routinely (any `docker compose up`
after an image or config change), anything that must outlive a container needs to be on
a volume.

That is the whole point of putting the mount at `/mlflow-data` rather than scattering the
db and artifacts around the filesystem: one directory, one mount, both the sqlite
database and the artifact tree survive.

## Question 2 - Why a named volume instead of a bind mount? Would a bind mount work?

**A bind mount would work functionally** - mlflow would read and write happily either
way. So this is a design choice, not a correctness one. Named volumes win here for:

| | named volume | bind mount |
|---|---|---|
| location | Docker-managed (`mlops_mlflow-data`) | a path in your repo |
| portability | identical on Windows, Mac, Linux | host-path and OS specific |
| in git's way? | no | yes, needs `.gitignore` |
| permissions | Docker owns them | host uid/gid vs container uid clashes |
| Windows/WSL2 I/O | native Linux filesystem, fast | crosses the 9p boundary, slow |
| lifecycle | `down -v` removes it deliberately | survives everything, easy to forget |
| backup | `docker volume` commands, `--volumes-from` | ordinary file copy |

The deciding reasons in this project specifically:

1. **It is sqlite.** A database file over a Windows→WSL2 bind mount is the classic
   source of locking and corruption problems; file locking semantics do not cross that
   boundary cleanly. A named volume lives on the Linux side.
2. **The repo already has `mlruns/` and `mlflow.db` in `.gitignore` and
   `.dockerignore`.** A bind mount would put a second, parallel copy of the same kind of
   state back inside the repo.
3. **`down -v` is an explicit, scriptable reset** - see Q10. With a bind mount you would
   delete a folder by hand and hope you got the right one.

When a bind mount *is* the right call: when a human needs to read or edit the files
directly - live-reloading source code in dev, or inspecting artifacts without
`docker cp`.

## Question 3 - Why does `http://mlflow:5000` resolve now when `host.docker.internal` was needed before?

Because the containers are now **on the same user-defined network**, and Compose runs an
embedded DNS server on it.

In lab 3 there was one container on the default bridge network, and mlflow was a process
on the *host* - outside Docker entirely. The only way across that boundary was a name
Docker Desktop injects for the host's address (`host.docker.internal`), or
`--network host` to abolish the boundary.

Now Compose creates a network per project (`mlops_default`) and attaches all three
services to it. Every container gets its resolver pointed at Docker's embedded DNS at
`127.0.0.11`, which resolves **each service name to that service's container IP** on
that network. So `mlflow` is a real DNS name, and `http://mlflow:5000` reaches the
container's port 5000 directly - not the published host port, the container port.

Three consequences worth noting:

- The name is the *service* name from `docker-compose.yml`, not the container name
  (`mlops-mlflow-1`), and not a hostname anyone configured.
- It resolves **only inside the project network**. From your host shell,
  `curl http://mlflow:5000` fails; you need `127.0.0.1:5000`, the published port.
- If the service scales to several replicas, the same name round-robins across them -
  which is the seed of the load-balancing answer in Q11.

`serve.py` needed no change at all, exactly as the lab says: it reads
`MLFLOW_TRACKING_URI`, and Compose supplies `http://mlflow:5000` instead of
`http://host.docker.internal:5000`.

> **But DNS alone was not enough.** Reaching mlflow by name still produced a 403:
> `Invalid Host header - possible DNS rebinding attack detected`, because requests now
> arrive with `Host: mlflow:5000`, which is not in mlflow's default allow-list. And once
> that was fixed, artifact download failed, because a plain `--default-artifact-root`
> hands clients a `file:/mlflow-data/...` path that exists only inside the *mlflow*
> container. Both are fixed in `mlflow/Dockerfile` with `--allowed-hosts` and
> `--serve-artifacts`; full detail in Q7.

## Question 4 - Why read `INFERENCE_URL` from the environment instead of hardcoding `http://inference:8000`?

Because `http://inference:8000` is **only meaningful inside this Compose project**. The
hostname `inference` exists because of `docker-compose.yml`; nothing in the frontend
image makes it true.

Hardcode it and the image works in exactly one context. Read it from the environment and
the same image runs:

| context | `INFERENCE_URL` |
|---|---|
| this Compose stack | `http://inference:8000` |
| `docker run` alone, API on the host | `http://host.docker.internal:8000` |
| local dev, no containers at all | `http://127.0.0.1:8000` (the default) |
| Kubernetes | `http://inference.default.svc.cluster.local:8000` |
| staging | `https://inference.staging.example.com` |

This is the same principle as `MLFLOW_TRACKING_URI` in `serve.py`, and the general rule:
**configuration that varies by environment does not belong in the image.** An image is
built once and should be promotable from dev to staging to prod unchanged; anything
baked in that differs per environment forces a rebuild per environment, and then the
thing you tested is not the thing you shipped.

The default matters too. `os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000")`
means the bare `streamlit run app.py` still works for local development with no
container at all - the variable is an override, not a requirement.

## Question 5 - Why doesn't `inference` publish a port, and how does the frontend still reach it?

**It does not need one.** Publishing maps a container port onto a host interface, and the
only reason to do that is to let something *outside* the Docker network in. Nothing
outside needs the inference API: the only client is the frontend, which is on the same
network.

The frontend reaches it over the project network by service name, `http://inference:8000`
- the container port, never the host. Verified from inside the frontend container:

```
$ docker compose exec frontend python -c "import requests; print(requests.get('http://inference:8000/health').json())"
{'status': 'ok'}
```

Not publishing is the better default, for three reasons:

1. **Attack surface.** A published port is reachable from the host and, depending on
   firewall rules, the LAN. The model API has no authentication; the fewer ways in, the
   better.
2. **No host port collisions.** Port 8000 stays free. This matters immediately when you
   scale - three replicas cannot all publish 8000.
3. **It documents intent.** Reading the compose file tells you which services are entry
   points (mlflow's UI, the frontend) and which are internal.

`EXPOSE 8000` in the Dockerfile is *not* publishing - it is metadata saying "this image
listens here". That is why `docker compose ps` shows `8000/tcp` with no `->` mapping.

## Question 6 - `depends_on` waits for the process, not readiness. What happens to `inference`?

**The container exits with code 3 and Compose does not bring it back.**

Demonstrated by stopping mlflow and restarting inference:

```
$ docker compose stop mlflow
$ docker compose restart inference
$ docker compose ps -a
SERVICE     STATUS
frontend    Up 6 minutes
inference   Exited (3) 11 seconds ago
mlflow      Exited (0) 5 minutes ago

$ docker compose logs inference
mlflow.exceptions.MlflowException: API request to
  http://mlflow:5000/api/2.0/mlflow/registered-models/alias
  failed with exception HTTPConnectionPool...
ERROR:    Application startup failed. Exiting.
```

`serve.py` resolves `models:/food11@champion` in its FastAPI `lifespan` startup hook. If
that call fails, startup fails, uvicorn exits, and with no `restart:` policy the
container stays dead - while `frontend` cheerfully stays `Up` and shows a connection
error to anyone who visits.

### The honest part: the race did not actually fire on a cold start

Running `docker compose up` from scratch with the lab's plain `depends_on: [mlflow]`, the
stack came up fine. Not because `depends_on` worked, but because of timing luck:

```
09:07:05  Started server process [1]
09:07:39  Application startup complete.        <- 34 seconds later
```

Importing torch and downloading the model takes ~34 s, and mlflow becomes ready well
inside that window. **The race is real but masked by a slow import.** That is the worst
kind of bug - it passes locally and fails on a faster machine, a warm page cache, or a
slower mlflow start. Relying on it would be relying on an accident.

### The fix

A healthcheck on mlflow plus `condition: service_healthy`:

```yaml
  mlflow:
    healthcheck:
      # python:3.11-slim has no curl, so probe with the interpreter
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()"]
      interval: 5s
      timeout: 3s
      retries: 20
      start_period: 10s

  inference:
    depends_on:
      mlflow:
        condition: service_healthy
    restart: unless-stopped
```

Now the ordering is explicit rather than lucky:

```
$ docker compose up -d --wait
 Container mlops-mlflow-1     Healthy
 Container mlops-inference-1  Healthy
 Container mlops-frontend-1   Healthy
STARTUP_SECONDS=54
```

`restart: unless-stopped` is still worth having: the healthcheck only closes the
*startup* race, and mlflow can go away later. A belt-and-braces version would also make
`serve.py` retry with backoff rather than dying on first failure - a healthcheck fixes
ordering, not resilience.

## Question 7 - `docker compose ps`: which services publish a port?

```
SERVICE     STATUS                        PORTS
frontend    Up 1 second                   0.0.0.0:8501->8501/tcp
inference   Up 13 seconds (healthy)       8000/tcp
mlflow      Up About a minute (healthy)   0.0.0.0:5000->5000/tcp
```

**Yes, it matches the compose file exactly.** `mlflow` and `frontend` declare `ports:`
and show a `0.0.0.0:host->container/tcp` mapping. `inference` declares none and shows a
bare `8000/tcp` - the port is *exposed* (image metadata, reachable on the project
network) but not *published* (not bound to any host interface).

The practical test: `curl http://127.0.0.1:8000/predict` from the host fails, while the
same request from inside the frontend container succeeds.

### Two things the lab's `mlflow/Dockerfile` is missing

Neither is optional - the stack cannot work without both, and each produced a different
failure.

**1. `--allowed-hosts`.** mlflow validates the `Host` header. Requests from the Compose
network arrive as `Host: mlflow:5000`, which is not in the default list:

```
403 Invalid Host header - possible DNS rebinding attack detected
```

Entries are matched **exactly and must include the port** - listing bare hostnames is not
enough, and passing the flag at all *replaces* the permissive default, so forgetting
`127.0.0.1:5000` locks out your own browser too.

**2. `--serve-artifacts` / `--artifacts-destination`.** With only
`--default-artifact-root /mlflow-data/mlruns`, mlflow records artifact locations as
`file:/mlflow-data/mlruns/...` and every client reads that path **off its own
filesystem**. `/mlflow-data` exists only in the mlflow container, so the inference
container fails with `No such artifact: ''`. Proxied serving streams the bytes over HTTP
instead, and artifact locations become `mlflow-artifacts:/1/...`, which any client that
can reach the API can fetch.

```dockerfile
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"]
```

The general lesson, same as lab 3: **a local `file:` artifact root only works while every
client shares one filesystem.** Containerising mlflow breaks that assumption immediately.

### Seeding the registry

A related consequence: the volume starts empty, so the containerised mlflow has no
experiments, no runs and no registered model - and `inference` cannot start without
`models:/food11@champion`. The lab 2 model lives in the *host's* `mlruns/`, which is not
in the volume.

`scripts/seed_registry.py` re-logs those local weights into whatever tracking server
`MLFLOW_TRACKING_URI` points at, registers them as `food11` and sets the alias:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python scripts/seed_registry.py
```

```
artifact uri    : mlflow-artifacts:/1/models/m-bbf9995ac65344ee90e4bba0d9fc6109/artifacts
registered      : food11 v1
alias           : champion -> v1
```

Predictions through the full stack match lab 3 to three decimals, which confirms the same
model is being served through a completely different topology:

| image | lab 3 (host mlflow) | lab 4 (Compose) |
|---|---|---|
| validation/Soup | Soup 1.000 | Soup 1.000 |
| validation/Bread | Fried food 0.738 | Fried food 0.738 |
| validation/Dessert | Dessert 0.519 | Dessert 0.519 |
| validation/Rice | Egg 0.372 | Egg 0.372 |

## Question 8 - After promoting a new version, does the prediction change? What single command picks it up?

**No - the old version keeps serving until the container restarts.** The command is:

```bash
docker compose restart inference
```

To make this verifiable rather than a matter of faith, I promoted a **genuinely
different** model: the `lr=0.01` run from lab 2, which diverged at 14.87% validation
accuracy (chance on 11 classes is 9.09%). If the promotion silently did nothing, the
predictions would be unchanged; if it took effect, they would visibly collapse.

I also extended `/model` to report the version actually held in memory:

```python
def resolve_version(uri: str) -> str | None:
    """Turn 'models:/food11@champion' into the concrete version it points at."""
```

**Before the restart** - registry already moved, container unaware:

```
champion -> v2                    (registry)
loaded_version = 1                (container)
  actual=Soup      -> Soup             conf=1.000
  actual=Bread     -> Fried food       conf=0.738
  actual=Dessert   -> Dessert          conf=0.519
```

**After `docker compose restart inference`:**

```
loaded_version = 2
  actual=Soup      -> Dessert          conf=0.233
  actual=Bread     -> Fried food       conf=0.237
  actual=Dessert   -> Bread            conf=0.204
```

Predictions collapsed to ~0.2 confidence and wrong classes - the diverged model, exactly
as expected. The promotion is unambiguous.

**Rolling back is the same two steps**, and took seconds:

```python
client.set_registered_model_alias("food11", "champion", 1)
```
```bash
docker compose restart inference
```
```
loaded_version = 1
  actual=Soup -> Soup conf=1.000
```

Why the staleness happens: `serve.py` resolves the alias **once**, in its startup hook,
and holds the model in memory. That is a deliberate trade - no registry round-trip per
request, at the cost of not noticing a promotion. The alternatives are a TTL re-check, a
webhook from the registry, or a `/reload` endpoint; all add complexity to buy something a
restart already gives you.

Worth flagging as a production concern: with several replicas, a rolling restart means
some replicas serve v1 and others v2 **at the same time**, so identical requests get
different answers until the roll completes. `loaded_version` in `/model` exists to make
that visible.

## Question 9 - Why does `restart` alone work, with no rebuild?

Because **the model was never in the image.** `restart` stops and starts the same
container from the same image; the only thing that changes is that the startup hook runs
again, re-resolves `models:/food11@champion`, and downloads whatever it now points at.

| baked into the image at build time | fetched at container startup |
|---|---|
| Python 3.13 and OS libraries | the model weights |
| the locked dependency tree (torch, mlflow, fastapi) | which version `champion` resolves to |
| `src/food11/serve.py` | the tracking server address (env var) |

`.dockerignore` excludes `mlruns/` and `mlflow.db`, and nothing copies weights in - so
the image is **model-agnostic**. It is a *runtime for Food-11 models*, not a container
holding one.

That separation is what makes the promotion workflow work:

- changing the **model** → registry operation + restart (seconds)
- changing the **code** → rebuild + `up -d` (the `COPY src` layer only; measured at
  **6 seconds** thanks to the layer ordering from lab 3's Q4, versus ~8 minutes if the
  dependency layer were invalidated)
- changing the **dependencies** → full rebuild (minutes)

The flip side is Q6's dependency: because nothing is baked in, the container is useless
without a reachable registry, and it fails hard rather than serving stale weights.

## Question 10 - Does the model survive `down`/`up`? What changes with `down -v`?

**`docker compose down` then `up`: everything survives.**

```
  model: food11 | aliases: [('champion', '1')]
  versions: ['2']
  inference loaded: 1
$ docker volume ls --filter name=mlops
mlops_mlflow-data local
```

Both registered versions, the `champion` alias and all artifacts came back untouched.
`down` removes containers and the network but **not** named volumes - the sqlite db and
artifact tree sit on `mlops_mlflow-data`, which is a separate object with its own
lifecycle. The containers were destroyed and recreated; the state was not in them.

**`docker compose down -v` then `up`: everything is gone.**

```
$ docker compose down -v
 Volume mlops_mlflow-data  Removed
 Network mlops_default     Removed
volumes left: 0

$ docker compose up -d mlflow --wait
$ curl .../registered-models/search
{}
$ curl .../experiments/search
  experiment: 0 Default
```

Empty registry, only the auto-created `Default` experiment - the same state as the
Q1 no-volume test. `-v` means "also delete the named volumes declared in this compose
file", which is the deliberate, scriptable "give me a clean slate" switch.

Three things worth drawing out:

1. **`down` vs `down -v` is the whole difference between a restart and a data loss.**
   `-v` is three keystrokes and there is no confirmation prompt.
2. **The inference service recovers automatically after `down`/`up`, but not after
   `down -v`** - there is no longer a `food11@champion` to load, so it exits 3 as in Q6.
   Recovery needs the seed script re-run.
3. **The volume is not a backup.** It is on this machine's disk, and `down -v` or a disk
   failure takes it. Real deployments point `--backend-store-uri` at a managed Postgres
   and artifacts at object storage - which is also most of the Q11 answer.

## Question 11 - What would three replicas behind a load balancer, or a failure-surviving mlflow, require?

Compose is a **single-host** tool. It can start `inference` three times
(`docker compose up --scale inference=3`) and its embedded DNS will round-robin the name,
which looks like load balancing - but everything underneath is missing.

### For replicas behind a real load balancer

| need | why Compose cannot give it |
|---|---|
| **a real load balancer** | DNS round-robin has no health awareness, no connection draining, no retries; clients cache DNS and pin to one replica. You need nginx/Traefik/Envoy or a cloud LB. |
| **readiness vs liveness** | Compose healthchecks restart a container; they cannot *remove it from rotation* while it is still starting. With a 34 s torch import, a new replica would receive traffic and fail. |
| **rolling updates** | `up` recreates containers with downtime and no surge/unavailable controls, no automatic rollback on failure. |
| **autoscaling** | `--scale` is a manual number. Nothing watches load. |
| **spread across machines** | all replicas land on one host - one kernel panic takes all three. Needs a scheduler (Kubernetes, Nomad, ECS). |
| **per-replica resource limits** | `deploy.resources` in compose files is ignored outside Swarm. |

Three replicas also expose the Q8 problem: each resolves `champion` independently at
startup, so a rolling restart serves two different models simultaneously.

### For mlflow to survive a machine failure

The current design has three single points of failure, all on one disk:

1. **sqlite** - one file, one writer, no replication. Replace with managed
   Postgres/MySQL (RDS, Cloud SQL): replicated, backed up, point-in-time recovery.
2. **artifacts on a local volume** - replace `--artifacts-destination /mlflow-data/mlruns`
   with S3/GCS/Azure Blob. Then mlflow becomes **stateless** and can itself run as
   several replicas behind a LB.
3. **the named volume** - not a backup. It dies with the host, and with `down -v`.

Once state is externalised, the mlflow container holds nothing and the whole failure
story changes: a dead host is replaced, the new one connects to the same database and
bucket, and nothing is lost.

### What Compose genuinely cannot express

Service discovery across hosts, secret management (secrets here would be plain
environment variables in a committed file), ingress and TLS termination, network
policies, persistent-volume claims with replication, pod disruption budgets, node
affinity, and declarative reconciliation - Compose runs commands, it does not
continuously drive the system toward a declared state.

**What Compose is genuinely good at:** exactly what this lab used it for - a reproducible
multi-service environment on one machine, for development, integration testing and CI.
The `docker-compose.yml` here is a precise, readable description of how three services
fit together, and that description is most of the input to a Kubernetes manifest later.

---

## Commands used

```bash
# build and bring up
docker compose build
docker compose up -d --wait
docker compose ps

# seed a fresh registry (the volume starts empty)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python scripts/seed_registry.py

# promote / roll back, then pick it up
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  LOCAL_MODEL_PATH=".../mlruns/1/models/m-3353e42ddb60435c85af8fd73f8f9599/artifacts" \
  uv run python scripts/seed_registry.py
docker compose restart inference

# service-to-service, no published port involved
docker compose exec frontend python -c \
  "import requests; print(requests.get('http://inference:8000/health').json())"

# persistence
docker compose down      && docker compose up -d   # state survives
docker compose down -v   && docker compose up -d   # state destroyed

git add mlflow/Dockerfile frontend/ docker-compose.yml scripts/seed_registry.py src/food11/serve.py
git commit -m "Orchestrate mlflow, inference and frontend with Docker Compose"
git push
```

## Files added

| file | purpose |
|---|---|
| `mlflow/Dockerfile` | tracking server, pinned to mlflow 3.16.1, with `--serve-artifacts` and `--allowed-hosts` |
| `frontend/app.py` | Streamlit upload page; sidebar shows the resolved model and tracking URI |
| `frontend/Dockerfile` | streamlit + requests on `python:3.11-slim`, 184 MB |
| `docker-compose.yml` | three services, one named volume, healthcheck-gated startup |
| `scripts/seed_registry.py` | re-logs the lab 2 champion into a fresh registry; `LOCAL_MODEL_PATH` promotes a different one |

Changed: `src/food11/serve.py` gained `resolve_version()` so `/model` reports the version
actually in memory - which is what makes Q8 verifiable rather than assumed.

Image sizes: `mlops-mlflow` 1.29 GB, `mlops-inference` 2.42 GB, `mlops-frontend` 184 MB.

Open the UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) and the frontend at
[http://127.0.0.1:8501](http://127.0.0.1:8501).
