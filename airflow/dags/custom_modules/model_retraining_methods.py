import os
import io
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from custom_modules.preprocessing_script import list_keys
from PIL import Image


PREFIX_GRASS = 'images/raw/new_data/grass'
PREFIX_DANDELION = 'images/raw/new_data/dandelion'


preprocess = transforms.Compose([
    transforms.Resize(256),          # Redimensionner le côté le plus court à 256 pixels
    transforms.CenterCrop(224),      # Recadrage central pour obtenir 224x224
    transforms.ToTensor(),           # Conversion en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalisation selon ImageNet
                         std=[0.229, 0.224, 0.225])
])

class NewDataDataset(torch.utils.data.Dataset):
    """
    Custom dataset class to load images from S3 bucket and apply transformations.
    """
    def __init__(self, s3_client, bucket_name, prefix, label=None, transform=None):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.label = label
        self.transform = transform
        self.keys = list_keys(prefix, s3_client, bucket_name)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        data = obj['Body'].read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = preprocess(img)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, self.label

def fine_tune_existing_model(existing_model_uri, bucket_name, s3_client, labels, additional_params):
    """
    Loads an existing model from MLflow, fine-tunes it on new data, and logs the updated model.
    
    Args:
      - existing_model_uri (str): MLflow model URI, e.g., "runs:/<RUN_ID>/model".
      - new_data_prefix (str): Prefix where new data is stored in the bucket (e.g., "images/raw/new_data/").
      - bucket_name (str): Name of the MinIO bucket.
      - s3_client: A boto3 S3 client instance.
      - additional_params (dict): Hyperparameters such as learning rate, number of epochs, and batch size.
    
    Returns:
      - A dictionary with retraining metrics (e.g., best validation accuracy).
    """
    # Load the existing model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    print("Loading model from:", existing_model_uri)
    model = mlflow.pytorch.load_model(existing_model_uri)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract hyperparameters
    lr = additional_params.get("learning_rate", 0.0001)
    num_epochs = additional_params.get("num_epochs", 10)
    batch_size = additional_params.get("batch_size", 32)

    # Create a transform if needed (here we assume the tensors are already sized [3, 224, 224])
    transform = None  # or define any additional transforms if needed

    # Create dataset for new data
    new_data_grass = NewDataDataset(
        s3_client=s3_client,
        bucket_name=bucket_name,
        prefix=PREFIX_GRASS,
        label=0,  # Set label if supervised fine-tuning is needed.
        transform=transform
    )
    
    new_data_dandelion = NewDataDataset(
        s3_client=s3_client,
        bucket_name=bucket_name,
        prefix=PREFIX_DANDELION,
        label=1,  # Set label if supervised fine-tuning is needed.
        transform=transform
    )
    
    full_dataset = ConcatDataset([
        new_data_grass,
        new_data_dandelion
    ])

    # Optionally split new data into training and validation sets; here we use 80/20 split:
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    mlflow.set_experiment('Resnet18_finetune')
    RUN_NAME = "retrain_model"

    # Start a new MLflow run for the retraining
    with mlflow.start_run(run_name=RUN_NAME, nested=True):
        mlflow.log_params({"learning_rate": lr,
                           "num_epochs": num_epochs,
                           "batch_size": batch_size,
                           "existing_model_uri": existing_model_uri})

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total_train = 0

            # Training Loop
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

            # Validation Loop
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
            print(f"Retrain Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc.item(), step=epoch)
            mlflow.log_metric("val_accuracy", val_acc.item(), step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        mlflow.log_metric("best_val_accuracy", best_val_acc.item())

        mlflow.pytorch.log_model(model, artifact_path="model")
        
        mlflow.register_model(model_uri=existing_model_uri, name="MyModel")
        

        print("Retraining complete, best validation accuracy:", best_val_acc.item())
        
        mlflow.end_run()

    return {"best_val_accuracy": best_val_acc.item()}

def move_new_data_to_treated(source_prefix, target_prefix, bucket_name=None, s3_client=None, **kwargs):
    """
    Move all files from the new_data folder to a treated folder in the same bucket.
    This involves copying each object to a new key (with treated/ instead of new_data/)
    and then deleting the original object.
    """

    
    # List keys in the new_data folder using your existing function
    keys = list_keys(source_prefix, s3_client, bucket_name)
    
    if not keys:
        print("No new data to move.")
        return "No data moved"
    
    for key in keys:
        # Derive the new key by replacing the source prefix with the target prefix.
        target_key = key.replace(source_prefix, target_prefix, 1)
        
        # Copy the object to the new location
        copy_source = {'Bucket': bucket_name, 'Key': key}
        s3_client.copy_object(CopySource=copy_source, Bucket=bucket_name, Key=target_key)
        print(f"Copied {key} to {target_key}")
        
        # Delete the original object
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        print(f"Deleted original key {key}")
    
    return f"Moved {len(keys)} objects from {source_prefix} to {target_prefix}"
