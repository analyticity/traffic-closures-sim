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

RUN pip install --no-cache-dir -e ".[geo]"

# ---- Stage 2: Run the pipeline to bake data into the image ----
FROM base AS pipeline

ARG CITY=brno
ENV PYTHONDONTWRITEBYTECODE=1

# Pre-downloaded sources from build context (CI pre-fetches & caches them).
# For local builds: mkdir -p data/sources before docker build.
COPY data/sources/ /app/data/sources/

RUN python run.py --config "config/${CITY}/sim.yaml" check \
    && python run.py --config "config/${CITY}/sim.yaml" build-network \
    && python run.py --config "config/${CITY}/sim.yaml" normalize-network \
    && python run.py --config "config/${CITY}/sim.yaml" build-zones \
    && python run.py --config "config/${CITY}/sim.yaml" fetch-data \
    && python run.py --config "config/${CITY}/sim.yaml" build-supernetwork \
    && python run.py --config "config/${CITY}/sim.yaml" build-demand \
    && python run.py --config "config/${CITY}/sim.yaml" assign-warm-skims \
    && python run.py --config "config/${CITY}/sim.yaml" distribute \
    && python run.py --config "config/${CITY}/sim.yaml" assign \
    && python run.py --config "config/${CITY}/sim.yaml" audit-supply \
    && python run.py --config "config/${CITY}/sim.yaml" calibrate \
    && python run.py --config "config/${CITY}/sim.yaml" validate \
    && python run.py --config "config/${CITY}/sim.yaml" learn-profile

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
