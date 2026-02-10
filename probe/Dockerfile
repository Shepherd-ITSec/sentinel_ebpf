# Stage 1: build venv with system Python (has bcc) and build tools for any source builds (e.g. numpy on 3.13)
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends bpfcc-tools python3-bpfcc linux-headers-amd64 ca-certificates curl \
        build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY probe /app/probe
COPY detector /app/detector
COPY events_pb2.py events_pb2_grpc.py /app/
RUN uv venv --python /usr/bin/python3 /app/.venv && uv sync --no-dev --frozen

# Stage 2: runtime image with bcc, no compilers
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends bpfcc-tools python3-bpfcc linux-headers-amd64 ca-certificates xz-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY probe /app/probe
COPY detector /app/detector
COPY events_pb2.py events_pb2_grpc.py /app/

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app:/usr/lib/python3/dist-packages
RUN mkdir -p /etc/sentinel-ebpf

ENTRYPOINT ["python", "-m", "probe.probe_runner"]
