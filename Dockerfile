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
    && python run.py --config "config/${CITY}/sim.yaml" calibrate \
    && python run.py --config "config/${CITY}/sim.yaml" validate \
    && python run.py --config "config/${CITY}/sim.yaml" learn-profile

# ---- Stage 3: Build frontend ----
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV VITE_API_URL=""
RUN npm run build

# ---- Stage 4: Final image ----
FROM base AS final

ARG CITY=brno
ENV SIM_CONFIG="config/${CITY}/sim.yaml"
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=pipeline /app/data/ /app/data/
COPY --from=pipeline /app/outputs/ /app/outputs/
COPY --from=pipeline /app/project/ /app/project/

COPY --from=frontend-build /app/frontend/dist/ /app/frontend/dist/

EXPOSE 8000
CMD ["python", "run.py", "serve"]
