# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any needed, e.g. for building packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirement.txt .

# Install any needed packages specified in requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose ports (8000 for backend, 8501 for frontend)
EXPOSE 8000
EXPOSE 8501

# Default command (can be overridden in docker-compose)
CMD ["python3", "main.py"]
