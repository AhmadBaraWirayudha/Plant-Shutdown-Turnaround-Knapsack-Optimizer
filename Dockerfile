# ============================================================================
#  Turnaround Knapsack Optimizer — Multi-stage Docker build
#
#  Stage 1 (builder): install dependencies into a virtualenv
#  Stage 2 (runtime):  copy only the venv + source → small final image
#
#  Build:  docker build -t turnaround-optimizer .
#  Run:    docker run --rm -v "$(pwd)/reports:/app/reports" \
#                      -v "$(pwd)/dashboard:/app/dashboard" \
#                      -v "$(pwd)/database:/app/database" \
#                      turnaround-optimizer --budget 5000000
# ============================================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to build wheels for scipy / ortools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

LABEL maintainer="Reliability Engineering"
LABEL description="Plant Shutdown Turnaround Knapsack Optimizer — OR-Tools CP-SAT ILP"

# Minimal runtime deps (libgomp1 required by OR-Tools' parallel solver)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash optimizer

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=optimizer:optimizer . .

# Pre-create output dirs with correct ownership so volume mounts don't fail
RUN mkdir -p data/raw data/processed data/external reports/audit_logs dashboard database \
    && chown -R optimizer:optimizer /app

USER optimizer

# Healthcheck: confirm every hard dependency imports cleanly (catches dependency rot)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD python -c "import ortools, pandas, numpy, scipy, pyarrow, sqlalchemy, plotly, openpyxl, dotenv, requests" || exit 1

ENTRYPOINT ["python", "run_optimizer.py"]
CMD ["--budget", "5000000"]
