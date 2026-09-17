FROM python:3.12-slim

WORKDIR /app

COPY workload/workload.py .

ENTRYPOINT ["python", "workload.py"]
