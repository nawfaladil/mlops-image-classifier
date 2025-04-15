FROM apache/airflow:2.6.0

USER airflow

# Installer des bibliothèques supplémentaires (MySQL, Minio, Requests)
RUN pip install --no-cache-dir pymysql mysql-connector-python boto3 requests torch torchvision Pillow==9.3.0

# Copier les DAGs dans le dossier Airflow
USER root
COPY dags/ /opt/airflow/dags/

# # Copier le code source dans un dossier, par exemple /opt/mlops/
# COPY ./src/ /opt/mlops/

# # Ajouter /opt/mlops au PYTHONPATH pour qu'il soit visible par Python
# ENV PYTHONPATH="/opt/mlops:${PYTHONPATH}"

USER airflow

# Démarrer Airflow standalone (Web UI + Scheduler)
ENTRYPOINT ["airflow", "standalone"]