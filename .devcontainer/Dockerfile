FROM python:3.11


# Install Node.js/npm so Codex CLI can be installed
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && npm install -g @openai/codex \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install requirements
RUN pip install -r requirements.txt

# Register jupyter kernel
RUN python3 -m ipykernel install --user --name=container_env --display-name "Python (Container)"

# Expose Jupyter port
EXPOSE 8888

