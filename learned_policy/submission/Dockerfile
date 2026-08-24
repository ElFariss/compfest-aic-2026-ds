FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backhaul_runtime.py service.py demo_input.json ./
COPY artifacts ./artifacts

USER 65532:65532
EXPOSE 8088
CMD ["python", "service.py", "--host", "0.0.0.0", "--port", "8088", "--artifacts", "artifacts"]
