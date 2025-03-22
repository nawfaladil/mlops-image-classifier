from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import requests
from io import BytesIO
import boto3

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='plants_data_pipeline_grouped_boto3',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    def fetch_urls_from_db(**context):
        """
        Récupère (id, url_source, label) depuis la table plants_data.
        Seules les lignes dont url_source n'est pas NULL sont sélectionnées.
        """
        mysql_hook = MySqlHook(mysql_conn_id='mysql_conn_id')
        sql = """
            SELECT id, url_source, label
            FROM plants_data
            WHERE url_source IS NOT NULL
        """
        records = mysql_hook.get_records(sql)
        results = []
        for row in records:
            results.append({
                'id': row[0],
                'url_source': row[1],
                'label': row[2]
            })
        return results

    def upload_images_to_minio(**context):
        """
        Pour chaque ligne (id, url_source, label) :
          - Télécharge l'image en mémoire depuis url_source.
          - Upload l'image vers MinIO dans un dossier selon le label
            (ex. images/dandelion/ pour le label "dandelion" et images/grass/ pour "grass").
          - Construit l'URL S3 et retourne une liste (id, url_s3) pour la mise à jour en DB.
        """
        ti = context['ti']
        data = ti.xcom_pull(task_ids='fetch_urls_from_db')
        if not data:
            raise ValueError("Aucune donnée récupérée depuis fetch_urls_from_db")

        # Créez un client boto3 pour MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',  # Depuis le conteneur Airflow, utilisez le nom de service "minio"
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        bucket_name = 'mlops-bucket'
        updated_rows = []

        for item in data:
            img_id = item['id']
            url_source = item['url_source']
            label = item['label']

            response = requests.get(url_source)
            if response.status_code != 200:
                raise Exception(f"Erreur lors du téléchargement de {url_source}")

            # Prépare le contenu de l'image en mémoire
            file_obj = BytesIO(response.content)

            # Nommer le fichier avec l'id et placer dans un dossier selon le label
            file_name = f"{img_id}.jpg"
            key = f"images/raw/{label}/{file_name}"

            # Upload direct vers MinIO avec boto3
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=file_obj.getvalue()
            )

            # Construit l'URL S3 (format s3://...)
            url_s3 = f"s3://{bucket_name}/{key}"

            updated_rows.append({'id': img_id, 'url_s3': url_s3})
        return updated_rows

    def update_db_with_s3_urls(**context):
        """
        Met à jour la colonne url_s3 dans la table plants_data pour chaque image traitée.
        """
        ti = context['ti']
        updated_rows = ti.xcom_pull(task_ids='upload_images_to_minio')
        if not updated_rows:
            raise ValueError("Aucune donnée à mettre à jour")

        mysql_hook = MySqlHook(mysql_conn_id='mysql_conn_id')
        conn = mysql_hook.get_conn()
        cursor = conn.cursor()

        for row in updated_rows:
            sql_update = """
                UPDATE plants_data
                SET url_s3 = %s
                WHERE id = %s
            """
            cursor.execute(sql_update, (row['url_s3'], row['id']))
        conn.commit()
        cursor.close()
        conn.close()

    # Déclaration des tâches
    fetch_urls = PythonOperator(
        task_id='fetch_urls_from_db',
        python_callable=fetch_urls_from_db,
        provide_context=True
    )

    upload_images = PythonOperator(
        task_id='upload_images_to_minio',
        python_callable=upload_images_to_minio,
        provide_context=True
    )

    update_db = PythonOperator(
        task_id='update_db_with_s3_urls',
        python_callable=update_db_with_s3_urls,
        provide_context=True
    )

    # Ordonnancement des tâches
    fetch_urls >> upload_images >> update_db
