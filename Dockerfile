# One image, two entrypoints: the shopdesk service and the proofbench runner.
# They share every dependency, and building them separately would mean two
# layers to keep in step for no gain.
FROM python:3.12.15-slim-trixie AS runtime

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


# The published dashboard is built here, and only if everything that vouches for
# its numbers passes first: the test suite, and the replay gate that re-grades the
# recorded baseline and fails if any grader's verdicts moved. A dashboard built
# from code that fails its own gate is not published.
FROM runtime AS site

USER root
RUN pip install ".[dev]"
COPY tests ./tests
COPY eval ./eval
RUN python -m pytest -q -p no:cacheprovider \
 && proofbench run suites/support --replay eval/baseline/artifact.json --require-fresh \
      --baseline eval/baseline/summary.json --out /tmp/gate --quiet \
 && proofbench dashboard --out /site/index.html


# `docker build --target export --output type=local,dest=out .` writes just the
# page to ./out, with nothing else from the image.
FROM scratch AS export
COPY --from=site /site/index.html /index.html
