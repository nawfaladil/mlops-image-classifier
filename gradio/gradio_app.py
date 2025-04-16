import gradio as gr
import requests
from PIL import Image
import io

# URL of your FastAPI prediction endpoint.
# Adjust the host and port if necessary (e.g., if running via Docker, use the container name).
API_URL = "http://fastapi:8000/predict"

def predict_via_api(image: Image.Image) -> str:
    """
    This function receives an image from Gradio, converts it to bytes,
    and makes an HTTP POST request to the FastAPI /predict endpoint.
    It returns the prediction as a string.
    """
    # Convert the PIL image to bytes.
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Prepare the file to be sent via a POST request.
    files = {"file": ("image.png", img_bytes, "image/png")}

    try:
        response = requests.post(API_URL, files=files)
    except Exception as e:
        return f"Error: Could not connect to API. {e}"

    if response.status_code != 200:
        return f"Error from API: {response.status_code}, {response.text}"

    # Parse the API JSON response.
    result = response.json()
    predicted_class = result.get("predicted_class", "unknown")
    confidence = result.get("confidence", 0)

    return f"Predicted: {predicted_class} (Confidence: {confidence:.2f})"

# Create a Gradio interface that calls our predict_via_api function.
iface = gr.Interface(
    fn=predict_via_api,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs="text",
    title="Image Prediction Frontend",
    description="This interface uses Gradio as a frontend to call our FastAPI prediction service. "
                "Upload an image and get the prediction (e.g., 'grass' or 'dandelion')."
)

if __name__ == "__main__":
    # Launch Gradio on port 7860; available at http://localhost:7860
    iface.launch(server_name="0.0.0.0", server_port=7860)
