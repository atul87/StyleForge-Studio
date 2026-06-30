# Use a lightweight python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for Pillow and OpenCV/image libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker cache layers
COPY requirements.txt .

# Install dependencies.
# We download the CPU version of PyTorch to keep the Docker image small (~1.5GB vs ~5GB+ for CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose target port
EXPOSE 5000

# Set running directory to the Flask app folder where app.py and gunicorn.conf.py are located
WORKDIR /app/NST_Code

# Start server using Gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
