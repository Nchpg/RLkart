# Backend Dockerfile
FROM python:3.10-slim

# System dependencies for PyBullet and FastAPI
RUN apt-get update && apt-get install -y \ 
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY ./backend ./backend
COPY ./rl ./rl
COPY ./models ./models

# Expose port 8000
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
