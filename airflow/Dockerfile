FROM apache/airflow:2.6.0

USER airflow

# Installer des bibliothèques supplémentaires (MySQL, Minio, Requests)
RUN pip install --no-cache-dir pymysql mysql-connector-python boto3 requests

# Copier les DAGs dans le dossier Airflow
USER root
COPY dags/ /opt/airflow/dags/

USER airflow

# Démarrer Airflow standalone (Web UI + Scheduler)
ENTRYPOINT ["airflow", "standalone"]