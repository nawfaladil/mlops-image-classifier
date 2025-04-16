# Use an official Python base image.
FROM python:3.8-slim

# Set a working directory.
WORKDIR /app

# Copy requirements.txt and install dependencies.
COPY ./requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Expose the port that uvicorn will run on.
EXPOSE 8000

# Command to run the application using uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]