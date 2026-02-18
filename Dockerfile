# Use official Python 3.12 slim image
FROM python:3.12-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_DIR=/app/venv \
    PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Install minimal system dependencies for building CatBoost/scikit-learn
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv $VENV_DIR

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install pinned versions from requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Install your project in editable mode
RUN pip install --no-cache-dir -e .

# Expose port and run Flask
ENV PORT=8080
EXPOSE 8080
CMD ["python", "app.py"]
