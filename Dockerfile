FROM python:3.13-slim

WORKDIR /app

# Install dependencies first, separately from app code — Docker caches this
# layer, so code-only changes won't force a full dependency reinstall on
# every rebuild.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code and everything predict.py needs at runtime
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# --host 0.0.0.0 is mandatory in a container: binding to 127.0.0.1 (the
# local-machine-only default) makes the server unreachable from OUTSIDE
# the container, even with the port exposed/mapped correctly.
# --app-dir src tells uvicorn to look for api.main:app inside src/,
# without us needing to change WORKDIR or restructure the app itself.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
