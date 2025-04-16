# Use an official Python base image.
FROM python:3.8-slim

# Set working directory.
WORKDIR /app

# Copy dependencies and install.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Gradio application code.
COPY . .

# Expose the port that Gradio runs on.
EXPOSE 7860

# Run the Gradio app.
CMD ["python", "gradio_app.py"]