FROM apache/airflow:2.10.5

USER airflow

RUN pip install --no-cache-dir \
    pymysql \
    mysql-connector-python \
    boto3 \
    requests \
    torch \
    torchvision \
    Pillow==10.1 \
    mlflow



USER root
COPY dags/ /opt/airflow/dags/


USER airflow

ENTRYPOINT ["airflow", "standalone"]
