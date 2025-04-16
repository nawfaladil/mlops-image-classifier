"""
This script is an Airflow DAG that automates the process of retraining a machine learning model
"""
from datetime import datetime, timedelta
from custom_modules.preprocessing_script import list_keys
from custom_modules.model_retraining_methods import fine_tune_existing_model, move_new_data_to_treated
import boto3
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from functools import partial
import os

os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"

MODEL_URI = 'runs:/de315fec012c4251833d2051e10a36da/model'
BUCKET_NAME = 'mlops-bucket'
PREFIX_GRASS = 'images/raw/new_data/grass'
PREFIX_DANDELION = 'images/raw/new_data/dandelion'

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
}

dag = DAG(
    dag_id='retraining_pipeline',
    default_args=default_args,
    schedule_interval='@daily',  # or set to None for manual triggering
    catchup=False
)

# Task 1: Vérifier la présence de nouvelles données dans le bucket MinIO.
s3_client = boto3.client(
    's3',
    endpoint_url='http://minio:9000',  # Use the service name if in the same Docker network
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

# A simple downstream task to log that new data has been detected
def new_data_detected(**kwargs):
    grass_keys = list_keys(PREFIX_GRASS, s3_client, BUCKET_NAME)
    dandelion_keys = list_keys(PREFIX_DANDELION, s3_client, BUCKET_NAME)
    if grass_keys or dandelion_keys:
        print("New  data detected")
        return True
    else:
        print("No new data detected")
        return None

notify_new_data = PythonOperator(
    task_id="notify_new_data",
    python_callable=new_data_detected,
    provide_context=True,
    dag=dag
)

def retrain_model_task(**kwargs):
    """
    Task to retrain the model if new data is detected.
    """
    additional_params = {
        'learning_rate': 0.0001,
        'num_epochs': 10,
        'batch_size': 32
    }
    grass_keys = list_keys(PREFIX_GRASS, s3_client, BUCKET_NAME)
    if grass_keys:
        print("Starting retraining on new data keys")
        metrics = fine_tune_existing_model(
            existing_model_uri=MODEL_URI,
            bucket_name=BUCKET_NAME,
            s3_client=s3_client,
            labels=0,
            additional_params=additional_params
        )
        print("Retraining metrics:", metrics)
        return metrics

    else :
        print("No new data detected for retraining.")
        return None


retrain_task = PythonOperator(
    task_id="retrain_model",
    python_callable=retrain_model_task,
    provide_context=True,
    dag=dag
)

def move_data(**kwargs):
    """
    Task to move new data to the treated folder after retraining.
    """
    grass_keys = list_keys(PREFIX_GRASS, s3_client, BUCKET_NAME)
    dandelion_keys = list_keys(PREFIX_DANDELION, s3_client, BUCKET_NAME)
    a = None
    b = None
    if grass_keys:
        a = move_new_data_to_treated(
            bucket_name=BUCKET_NAME,
            source_prefix=PREFIX_GRASS,
            target_prefix='images/treated/grass',
            s3_client=s3_client
        )
    if dandelion_keys:
        b = move_new_data_to_treated(
            bucket_name=BUCKET_NAME,
            source_prefix=PREFIX_DANDELION,
            target_prefix='images/treated/dandelion',
            s3_client=s3_client
        )
    if not grass_keys and not dandelion_keys:
        print("No new data to move.")
        return None
    print("Moved new data to treated folder.")
    return a, b
             

move_data_task = PythonOperator(
    task_id="move_new_data",
    python_callable=move_data,
    provide_context=True,
    dag=dag
)


notify_new_data >> retrain_task >> move_data_task
