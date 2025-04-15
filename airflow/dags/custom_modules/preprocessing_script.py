import io
from PIL import Image
import torchvision.transforms as transforms
import torch

# Définir le pipeline de transformation pour ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),          # Redimensionner le côté le plus court à 256 pixels
    transforms.CenterCrop(224),      # Recadrage central pour obtenir 224x224
    transforms.ToTensor(),           # Conversion en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalisation selon ImageNet
                         std=[0.229, 0.224, 0.225])
])

# Pipeline d'augmentation pour l'entraînement
data_augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),       # Flip horizontal aléatoire
    transforms.RandomRotation(degrees=15),        # Rotation aléatoire jusqu'à 15°
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),          # Modification aléatoire des couleurs
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Recadrage aléatoire et redimensionnement
    transforms.ToTensor(),                        # Conversion en tenseur PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def list_keys(prefix, s3_client, bucket):
    """list bucket objects keys"""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents']]
    return []


# Exemple d'utilisation sur une image chargée depuis votre bucket
def preprocess_image_from_key(key, s3_client, bucket):
    """preprocessing pipeline"""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj['Body'].read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_tensor = preprocess(img)
    return img_tensor

def augment_image_from_key(key, s3_client, bucket):
    """augmentation pipeline"""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj['Body'].read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_tensor = data_augmentation_transforms(img)
    return img_tensor

def tensor_to_serialized_bytes(tensor):
    """Sérialise le tensor et retourne des bytes."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return buffer.read()
