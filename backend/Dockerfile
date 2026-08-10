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

# torch is installed FIRST, from PyTorch's CPU-only index.
#
# The default wheel on PyPI is the CUDA build: it drags in the whole NVIDIA
# runtime and turns this image into several gigabytes — on a host with no GPU,
# where PPO inference runs on the CPU anyway. Installing it first means the
# `-r requirements.txt` below finds torch already satisfied and leaves it alone.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY ./backend ./backend
COPY ./rl ./rl
COPY ./models ./models

# Expose port 8000
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
