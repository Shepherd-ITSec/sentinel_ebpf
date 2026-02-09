FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends bpfcc-tools python3-bpfcc linux-headers-amd64 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY probe /app/probe
COPY detector /app/detector
COPY events_pb2.py events_pb2_grpc.py /app/
# Use system Python (the one that has bcc from python3-bpfcc) for the venv so bcc is available
RUN uv venv --python /usr/bin/python3 /app/.venv && \
    uv sync --no-dev --frozen

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
# bcc is in system dist-packages; ensure it is found when running the venv python
ENV PYTHONPATH=/app:/usr/lib/python3/dist-packages
RUN mkdir -p /etc/sentinel-ebpf

ENTRYPOINT ["python", "-m", "probe.probe_runner"]
