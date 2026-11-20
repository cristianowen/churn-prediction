# Serves the trained model over HTTP. Train first: `make train`
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

COPY api ./api
COPY config.yaml ./
COPY models/churn_model.joblib ./models/churn_model.joblib
ENV CHURN_MODEL_PATH=/app/models/churn_model.joblib

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
