# Builds a container that runs the Streamlit app.
#
# The image bundles the app, the source, and the saved run results in data/, so
# it starts up ready to browse. You don't need an API key just to read the numbers.
#
# The API key is never built into the image. It is read at runtime from the
# DO_INFERENCE_KEY environment variable, which you set in the App Platform
# dashboard. The .dockerignore file keeps the local .env out of the image.

FROM python:3.12-slim

# Don't write .pyc files; make logs flush immediately (nice for container logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (this layer caches unless requirements change).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the data (issues, ground truth, cached runs).
COPY src/ ./src/
COPY app.py .
COPY data/ ./data/

# App Platform provides the port to listen on via $PORT (defaults to 8080 here).
# Streamlit must bind 0.0.0.0 and run headless inside a container.
ENV PORT=8080
EXPOSE 8080

# Shell form so ${PORT} is expanded at runtime.
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
