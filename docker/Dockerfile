FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV MODE=${MODE} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.0.5

WORKDIR /workspace

RUN pip install -U pip && \
    pip install --no-cache-dir poetry==${POETRY_VERSION}

COPY poetry.lock pyproject.toml calbert /workspace/

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install $(test "$MODE" == production && echo "--no-dev") --no-interaction --no-ansi
