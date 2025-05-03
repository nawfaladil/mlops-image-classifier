FROM apache/airflow:slim-2.10.5 AS airflow

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      default-libmysqlclient-dev \
      pkg-config && \
    rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
      "apache-airflow[webserver]" \
      apache-airflow-providers-mysql \
      boto3 \
      requests \
      mlflow \
      torch==2.2.1+cpu -f https://download.pytorch.org/whl/torch_stable.html\
      torchvision==0.17.1+cpu \
      Pillow==10.1.0 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"

COPY airflow/dags/ /opt/airflow/dags/

ENTRYPOINT ["airflow", "standalone"]
