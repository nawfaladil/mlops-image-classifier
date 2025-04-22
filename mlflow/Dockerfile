FROM ghcr.io/mlflow/mlflow:v2.21.2

USER root

RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir boto3


CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "s3://mlops-bucket/mlflow-artifacts", "--host", "0.0.0.0"]
