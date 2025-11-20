# Use Python 3.13 as base image
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching layers)
COPY server/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY server/ .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app/app.py"]