# Use the official MLflow image as the base
FROM ghcr.io/mlflow/mlflow:v2.21.2

# Switch to root to install packages
USER root

# Update package lists and install awscli
RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir boto3

# Use the default command of the MLflow image (or redefine it as needed)
# For instance:
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "s3://mlops-bucket/mlflow-artifacts", "--host", "0.0.0.0"]
