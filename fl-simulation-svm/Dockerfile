FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/fl-simulation-svm

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements-docker.txt

COPY . .

ENV MPLBACKEND=Agg

RUN mkdir -p results
