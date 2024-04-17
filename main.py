from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("./saved_models/1")
CLASS_NAMES = ["Strawberry Late Blight ","Strawberry Healty"]



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

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
    uvicorn.run(app, host='localhost', port=8000)
