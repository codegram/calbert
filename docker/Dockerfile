FROM codegram/apex-pytorch1.4-cuda10.1:latest

WORKDIR /workspace

COPY pyproject.toml /workspace
COPY poetry.lock /workspace

RUN pip install poetry
RUN poetry install
