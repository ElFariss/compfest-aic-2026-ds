FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8088 \
    ARTIFACT_DIR=/app/artifacts

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY policy_runtime.py service.py ./
COPY artifacts/ ./artifacts/

USER 65532:65532
EXPOSE 8088
ENTRYPOINT ["python", "service.py"]
