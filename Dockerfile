FROM python:3.11-slim

WORKDIR /app

# Copy requirement file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Generate background dirty and ground-truth data sets inside the image
RUN python data_generator.py

# Expose API port
EXPOSE 7860

# Start server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
