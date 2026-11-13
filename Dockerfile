# One image, two entrypoints: the shopdesk service and the proofbench runner.
# They share every dependency, and building them separately would mean two
# layers to keep in step for no gain.
FROM python:3.12.15-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip==25.3 && pip install ".[otel]"

COPY suites ./suites

# Runs as nobody: the runner writes only to a mounted volume, and the service
# writes nothing at all.
USER 65534:65534

EXPOSE 8080
CMD ["uvicorn", "shopdesk.app:app", "--host", "0.0.0.0", "--port", "8080"]
