# Use Python 3.13 as base image
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Copy backend requirements first (for caching layers)
# The project stores Python requirements under `backend/requirements.txt`.
COPY backend/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app (path corrected)
CMD ["python", "backend/app/app.py"]