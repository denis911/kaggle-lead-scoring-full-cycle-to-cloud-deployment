# Use a slim Python 3.13 image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen to lockfile)
RUN uv sync --frozen --no-dev --no-install-project

# Final stage
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Set environment path to use the virtualenv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application files
COPY app.py lead_scoring_model.joblib ./

# Expose port (FastAPI default)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
