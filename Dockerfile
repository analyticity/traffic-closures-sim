# ---- Stage 1: Python base with system deps ----
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        libspatialindex-dev \
        libsqlite3-mod-spatialite \
        libgeos-dev \
        libproj-dev \
        gdal-bin \
        libgdal-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/
COPY run.py ./
COPY config/ config/
COPY scripts/ scripts/
COPY scenarios/ scenarios/
COPY experiments/ experiments/

RUN pip install --no-cache-dir -e ".[geo]"

# ---- Stage 2: Run the pipeline to bake data into the image ----
FROM base AS pipeline

ARG CITY=brno
ENV PYTHONDONTWRITEBYTECODE=1

ENV CFG="config/${CITY}/sim.yaml"

# Pre-downloaded sources from build context (CI pre-fetches & caches them).
# For local builds: mkdir -p data/sources before docker build.
COPY data/sources/ /app/data/sources/

# One RUN per step: when a step fails, the build log names it directly instead
# of only reporting that a 14-command chain exited 1.
RUN python run.py --config "$CFG" check
RUN python run.py --config "$CFG" build-network
RUN python run.py --config "$CFG" fetch-data
RUN python run.py --config "$CFG" normalize-network
RUN python run.py --config "$CFG" build-zones
RUN python scripts/check_gateways.py --config "$CFG"
RUN python run.py --config "$CFG" build-supernetwork
RUN python run.py --config "$CFG" build-demand
RUN python run.py --config "$CFG" assign-warm-skims
RUN python run.py --config "$CFG" distribute
RUN python run.py --config "$CFG" assign
RUN python run.py --config "$CFG" audit-supply
RUN python run.py --config "$CFG" calibrate
RUN python run.py --config "$CFG" validate
RUN python run.py --config "$CFG" learn-profile

# Experiments take a while and are not needed for a baseline/scenario run.
#   docker build --build-arg RUN_EXPERIMENTS=0 ...
ARG RUN_EXPERIMENTS=1
RUN if [ "$RUN_EXPERIMENTS" = "1" ]; then \
        python experiments/run_all.py --config "$CFG"; \
    else \
        echo "RUN_EXPERIMENTS=0 — experiments skipped"; \
    fi

# ---- Stage 3: Final image ----
FROM base AS final

ARG CITY=brno
ENV SIM_CONFIG="config/${CITY}/sim.yaml"
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=pipeline /app/data/ /app/data/
COPY --from=pipeline /app/outputs/ /app/outputs/
COPY --from=pipeline /app/project/ /app/project/

EXPOSE 8000
CMD ["python", "run.py", "serve"]
