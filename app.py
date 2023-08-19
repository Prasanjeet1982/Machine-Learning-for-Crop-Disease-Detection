from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from prediction import predict_image

app = FastAPI()

class ImageUpload(BaseModel):
    file: UploadFile

@app.post("/predict/")
async def predict_endpoint(upload: ImageUpload):
    result = predict_image(upload.file)
    return result
