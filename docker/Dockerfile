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

RUN mkdir -p /workspace/calbert /workspace/config /workspace/dist

COPY poetry.lock pyproject.toml /workspace/
COPY calbert/* /workspace/calbert/
COPY dist/* /workspace/dist/
COPY config/* /workspace/config/

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install $(test "$MODE" == production && echo "--no-dev") --no-interaction --no-ansi

