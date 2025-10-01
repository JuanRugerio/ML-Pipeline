# Use official Python 3.11 image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole project
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make run.sh executable
RUN chmod +x run.sh

# Default command to execute
CMD ["./run.sh"]

