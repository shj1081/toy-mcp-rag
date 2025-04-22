# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv


WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY . /app

# Create a virtual environment first
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv

# Install the project's dependencies using the requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e .

FROM python:3.12-slim-bookworm

# Install dependencies needed for document processing (PDF, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
 
COPY --from=uv /root/.local /root/.local
COPY --from=uv --chown=root:root /app/.venv /app/.venv
COPY --from=uv /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Create mount points for data directories
RUN mkdir -p /app/docs
RUN mkdir -p /app/vector_store

# Set environment variables
ENV PYTHONUNBUFFERED=1


# Default command entrypoint
ENTRYPOINT ["mcp-server-rag"]