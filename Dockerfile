# Private demo image: local model + redacted index, no API key, no data egress.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[ner,dense,local,demo]" \
    && python -m spacy download en_core_web_sm

EXPOSE 7860
CMD ["contract-assistant", "demo", "--config", "configs/default.json", "--host", "0.0.0.0", "--port", "7860"]
