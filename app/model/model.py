import tensorflow as tf
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load TensorFlow model
model = tf.keras.models.load_model(f"{BASE_DIR}/model/1")

def predict_image(image):
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence
