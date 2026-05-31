FROM python:3.11

# Install system tools used by the course notebooks and Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    graphviz \
    pkg-config \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install requirements
RUN python -m pip install --upgrade pip setuptools wheel

RUN pip download \
    --default-timeout=1000 \
    --retries=20 \
    --progress-bar off \
    -r requirements.txt \
    -d /tmp/wheels

RUN pip install \
    --no-index \
    --find-links=/tmp/wheels \
    --no-cache-dir \
    -r requirements.txt

RUN rm -rf /tmp/wheels

# Register jupyter kernel
RUN python3 -m ipykernel install --user --name=container_env --display-name "Python (Container)"

# Expose Jupyter port
EXPOSE 8888
