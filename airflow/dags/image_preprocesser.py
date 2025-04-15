from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import boto3
from custom_modules.preprocessing_script import list_keys, preprocess_image_from_key, augment_image_from_key, tensor_to_serialized_bytes

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

BUCKET_NAME = 'mlops-bucket'

# Configuration du client boto3 (adapter l'endpoint si Airflow tourne dans Docker)
s3_client_minio = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

def process_new_images(**context):
    """
    Fonction qui :
      - Liste les clés dans les dossiers raw (grass et dandelion)
      - Applique le prétraitement et augmentation
      - Réuploade les images traitées dans un préfixe 'processed' et 'augmented'
      - Affiche les clés traitées
    """
    # Exemple pour "grass" et "dandelion"
    prefixes = {
        'grass': 'images/raw/grass',
        'dandelion': 'images/raw/dandelion'
    }

    for label, prefix in prefixes.items():
        keys = list_keys(prefix, s3_client_minio, BUCKET_NAME)
        for key in keys:
            # for preprocessing images
            tensor = preprocess_image_from_key(key, s3_client_minio, BUCKET_NAME)
            serialized_bytes = tensor_to_serialized_bytes(tensor)
            new_key = key.replace('raw', 'processed')
            s3_client_minio.put_object(Bucket=BUCKET_NAME, Key=new_key, Body=serialized_bytes)
            # for augmenting images
            tensor = augment_image_from_key(key, s3_client_minio, BUCKET_NAME)
            serialized_bytes = tensor_to_serialized_bytes(tensor)
            new_key = key.replace('raw', 'augmented')
            s3_client_minio.put_object(Bucket=BUCKET_NAME, Key=new_key, Body=serialized_bytes)
            print(f"Preprocessing and augmenting {key} -> {new_key}")

with DAG(
    dag_id='image_processing_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    process_images = PythonOperator(
        task_id='process_new_images',
        python_callable=process_new_images,
        provide_context=True
    )

    process_images  # Une seule tâche pour cet exemple