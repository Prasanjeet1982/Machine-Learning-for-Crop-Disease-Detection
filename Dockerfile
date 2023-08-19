# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables for Docker image
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /app

# Copy the requirements files
COPY requirements_app.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy the application files into the container
COPY app.py /app/
COPY prediction.py /app/
COPY model_checkpoint.h5 /app/

# Expose the application port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]