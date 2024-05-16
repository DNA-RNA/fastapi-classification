from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from tensorflow.keras.models import load_model
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)


MODEL = load_model("./model/strawberry-model.h5")
CLASS_NAMES = ["Strawberry Late Blight ","Strawberry Healty"]



def read_file_as_image(data) -> np.ndarray:
    # Open the image file
    image = Image.open(BytesIO(data))
    # Resize the image
    image = image.resize((256, 256))
    # Convert the image to a NumPy array
    return np.array(image)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image( await file.read())
    img_batch = np.expand_dims(image,0)

    predicts =  MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predicts[0])]
    confidence = np.max(predicts[0])
    
    response_data = {
        'class': predicted_class,
        'confidence': float(confidence)
    }

    # FastAPI'nin özel encoder'ını kullanarak numpy veri tiplerini uygun şekilde işle
    return JSONResponse(content=jsonable_encoder(response_data, custom_encoder={
        np.float32: lambda x: float(x),
        np.ndarray: lambda x: x.tolist()
    }))

@app.get("/ping")
async def ping():
    return "Hello worldd"

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 8000))
	run(app, host="0.0.0.0", port=port)
