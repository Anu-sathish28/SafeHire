FROM python:3.12-slim

# Install system packages required for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the complete SafeHire project
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn for production
RUN pip install --no-cache-dir gunicorn

# Move into the Flask backend folder
WORKDIR /app/backend

# Start SafeHire Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]