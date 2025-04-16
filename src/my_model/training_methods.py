import io
import boto3
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset


class MinIODataset(Dataset):
    """
    Dataset that retrieves images from a MinIO bucket.
    
    Args:
        s3_client: A boto3 client configured for MinIO.
        bucket_name: The name of the bucket in MinIO.
        prefix: The prefix (path) in the bucket where images are stored.
        label: Optionally, the label for all images in this dataset.
    """
    def __init__(self, s3_client, bucket_name, prefix, label=None):
        self.s3_client = s3_client  # This client will be excluded from pickling.
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.label = label
        self.keys = self._list_keys()

    def _list_keys(self):
        # Ensure the s3_client is initialized before usage.
        if self.s3_client is None:
            self._initialize_s3_client()
        keys = []
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith(('.jpg', '.png', '.jpeg')):
                    keys.append(obj['Key'])
        return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # On each worker, ensure s3_client is available
        if self.s3_client is None:
            self._initialize_s3_client()
        key = self.keys[idx]
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        data = obj['Body'].read()
        # Here, we assume your images have been serialized via torch.save
        tensor = torch.load(io.BytesIO(data))
        return tensor, self.label

    def _initialize_s3_client(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpickleable s3_client from the state
        state['s3_client'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the s3_client if it's missing
        if self.s3_client is None:
            self._initialize_s3_client()

def create_model(num_classes):
    """
    Create a ResNet model with a specified number of output classes.
    """
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train the model and validate it on the validation set.
    """
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase:
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_train += inputs.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = running_corrects.double() / total_train

        # Validation phase:
        model.eval()
        running_corrects_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects_val += torch.sum(preds == labels.data)
                total_val += inputs.size(0)

        val_acc = running_corrects_val.double() / total_val

        # Logging metrics via MLflow
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc.item(), step=epoch)
        mlflow.log_metric("val_accuracy", val_acc.item(), step=epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return model, best_val_acc

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_accuracy = running_corrects.double() / total_samples

    return avg_loss, avg_accuracy
