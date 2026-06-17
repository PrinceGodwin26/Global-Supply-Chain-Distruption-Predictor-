# ── Stage 1: Base image ───────────────────────────────────────
# FROM tells Docker: start with this pre-built image from Docker Hub
# python:3.11-slim is an official Python image — "slim" means minimal size
# Think of this as: start with a fresh laptop that already has Python 3.11 installed
FROM python:3.11-slim

# ── Stage 2: Set environment variables ───────────────────────
# These are like .env but baked into the container itself
# PYTHONDONTWRITEBYTECODE: stops Python creating .pyc cache files (keeps container clean)
# PYTHONUNBUFFERED: forces Python to print logs immediately (important for debugging)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Stage 3: Set working directory ───────────────────────────
# WORKDIR is like doing "cd" inside the container
# All future commands run from this folder
# Think of it as: this is where our project lives inside the container
WORKDIR /app

# ── Stage 4: Install system dependencies ─────────────────────
# RUN executes a shell command during the build process
# These are OS-level packages PostgreSQL driver needs to compile
# apt-get is the Linux package manager (like pip but for the OS)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 5: Install Python dependencies ─────────────────────
# COPY copies files FROM your computer INTO the container
# We copy requirements.txt first (before the rest of the code)
# Why? Docker caches each step — if requirements.txt didn't change,
# it skips re-installing all libraries. Huge time saver.
COPY requirements.txt .

# Now install all Python libraries listed in requirements.txt
# --no-cache-dir: don't store the download cache (keeps container size small)
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 6: Copy project code ───────────────────────────────
# Now copy everything else (all our Python code) into the container
# The "." means: copy everything from current folder on your PC
# into /app inside the container
COPY . .

# ── Stage 7: Expose port ──────────────────────────────────────
# EXPOSE tells Docker: this container will receive traffic on port 8000
# This is where our FastAPI server will listen for requests
# Think of it as: opening a specific door in the container for communication
EXPOSE 8000

# ── Stage 8: Default command ──────────────────────────────────
# CMD is what runs when the container starts
# This starts our FastAPI server using uvicorn
# --host 0.0.0.0 means: accept connections from outside the container
# --port 8000 matches the EXPOSE above
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
