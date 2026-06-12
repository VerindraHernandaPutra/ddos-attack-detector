FROM tensorflow/tensorflow:2.11.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/fl-simulation-cnn

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements-docker.txt

COPY . .

ENV MPLBACKEND=Agg

RUN mkdir -p results
