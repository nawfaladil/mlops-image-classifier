from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import torchvision.transforms as transforms
import torch
import io
import mlflow
import os

# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")

# Configure MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
MODEL_NAME = "MyModel"
MODEL_VERSION = "latest"

# Define a response model.
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

app = FastAPI(title="Image Prediction API")

# Global variable to store the model.
model = None

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.pytorch.load_model(model_uri)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

# Define the preprocessing transform.
preprocess = transforms.Compose([
    transforms.Resize(256),                # Resize the shorter side to 256
    transforms.CenterCrop(224),            # Center crop to 224x224
    transforms.ToTensor(),                 # Convert PIL Image to PyTorch tensor [C x H x W]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # ImageNet mean
        std=[0.229, 0.224, 0.225]           # ImageNet std
    )
])

def load_and_preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Load an image from bytes and preprocess it.
    """
    try:
        # Use BytesIO to load the image from bytes.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to open image: {e}")
    image_tensor = preprocess(image)
    # Add a batch dimension: shape becomes [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    # Read the image file bytes.
    try:
        file_bytes = await file.read()
        input_tensor = load_and_preprocess_image_bytes(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the image: {e}")

    # Check if the model is loaded.
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Get prediction.
    try:
        # Depending on how the model was saved using MLflow,
        # the loaded model might not be a standard PyTorch nn.Module.
        # If it is a mlflow.pyfunc, it usually expects a pandas DataFrame.
        # However, if you've logged a standard PyTorch model, you may be able to call eval()
        # on it. For example, if you registered and loaded it with mlflow.pytorch.log_model,
        # the object is a PyTorch nn.Module.
        #
        # Here, we check if the model has the "eval" attribute:
        model.eval()
        with torch.no_grad():
            # Model output should be a tensor of logits.
            output = model(input_tensor)
        # Apply softmax to obtain probabilities.
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
        labels = {0: "grass", 1: "dandelion"}
        predicted_class = labels.get(predicted_index.item(), "unknown")
        return PredictionResponse(predicted_class=predicted_class, confidence=confidence.item())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the image prediction API! POST your image to /predict."}
