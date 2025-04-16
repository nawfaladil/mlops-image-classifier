import os
import boto3
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from training_methods import create_model, train_model, evaluate_model, MinIODataset


os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

# Définir quelques paramètres
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0

BUCKET_NAME = 'mlops-bucket'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2  # Adjust based on your dataset


# Configurer MLflow pour pointer vers le serveur MLflow en Docker
RUN_NAME = f'Learning rate={LEARNING_RATE}'
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('Resnet18_finetune')


s3_client_minio = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )


with mlflow.start_run(run_name=RUN_NAME) as run:

    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("weight_decay", WEIGHT_DECAY)
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("model", "resnet18_finetune")

    dataset_processed_grass = MinIODataset(s3_client_minio,
                                       bucket_name=BUCKET_NAME,
                                       prefix='images/processed/grass', label=0)

    dataset_processed_dandelion = MinIODataset(s3_client_minio,
                                            bucket_name=BUCKET_NAME,
                                            prefix='images/processed/dandelion', label=1)

    dataset_augmented_grass = MinIODataset(s3_client_minio,
                                        bucket_name=BUCKET_NAME,
                                        prefix='images/augmented/grass', label=0)

    dataset_augmented_dandelion = MinIODataset(s3_client_minio,
                                            bucket_name=BUCKET_NAME,
                                            prefix='images/augmented/dandelion', label=1)

    full_dataset = ConcatDataset([
        dataset_processed_grass,
        dataset_processed_dandelion,
        dataset_augmented_grass,
        dataset_augmented_dandelion
    ])

    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mlflow.log_metric("train_size", train_size)
    mlflow.log_metric("val_size", val_size)
    mlflow.log_metric("test_size", test_size)

    model = create_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    model, best_val_acc = train_model(model,
                                      train_loader,
                                      val_loader,
                                      criterion,
                                      optimizer, NUM_EPOCHS, device)

    mlflow.log_metric("best_val_accuracy", best_val_acc.item())

    test_loss, test_accuracy = evaluate_model(model,
                                              test_loader,
                                              criterion, device)


    mlflow.log_metric("test_accuracy", test_accuracy.item())
    mlflow.log_metric("test_loss", test_loss)
    mlflow.pytorch.log_model(model, artifact_path="model")
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="MyModel")
    
    mlflow.end_run()
