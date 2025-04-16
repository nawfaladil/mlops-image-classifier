# Use the official Apache Airflow image as base.
FROM apache/airflow:2.10.5

# Switch back to the airflow user.
USER airflow

# Install additional Python libraries as root.
RUN pip install --no-cache-dir \
    pymysql \
    mysql-connector-python \
    boto3 \
    requests \
    torch \
    torchvision \
    Pillow==10.1 \
    mlflow



# Copy DAGs to the Airflow DAGs folder.
USER root
COPY dags/ /opt/airflow/dags/

# (Optional) Add additional code or update PYTHONPATH if needed.
# For example:
# COPY ./src/ /opt/mlops/
# ENV PYTHONPATH="/opt/mlops:${PYTHONPATH}"

USER airflow

# Start Airflow standalone (includes webserver and scheduler).
ENTRYPOINT ["airflow", "standalone"]
