FROM python:3.11

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install requirements
RUN pip install -r requirements.txt

# Register jupyter kernel
RUN python3 -m ipykernel install --user --name=container_env --display-name "Python (Container)"

# Expose Jupyter port
EXPOSE 8888

