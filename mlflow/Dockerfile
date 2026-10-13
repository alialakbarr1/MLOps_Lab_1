# Tracking server + model registry for the Compose stack.
#
# mlflow is pinned to the version that logged the model in lab 2; a server
# older than its clients can reject newer registry concepts (LoggedModel).
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.16.1

EXPOSE 5000

# /mlflow-data is the mount point of the named volume declared in
# docker-compose.yml, so both the sqlite database and the artifacts land on the
# volume rather than the container's writable layer.
#
# Two flags beyond the lab's version, both learned the hard way in lab 3:
#
#   --serve-artifacts / --artifacts-destination
#       Without these, mlflow hands clients a bare `file:/mlflow-data/mlruns/...`
#       URI and each client reads it off its OWN filesystem. /mlflow-data exists
#       only in this container, so the inference container would fail with
#       "No such artifact: ''". Proxied serving streams artifacts over HTTP
#       (mlflow-artifacts:/) so any client that can reach the API gets the bytes.
#
#   --allowed-hosts
#       mlflow validates the Host header against an allow-list. Requests from
#       inside the Compose network arrive as Host: mlflow:5000, which is not in
#       the default list, and are rejected with
#       "403 Invalid Host header - possible DNS rebinding attack detected".
#       Entries are matched exactly and must include the port.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"]
