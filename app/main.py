# Modified main.py approach
print("==== Starting Iris model ====")
import os
learning_rate = float(os.getenv("LEARNING_RATE", 0.01))
print("Using learning rate:", learning_rate)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris Model API")

# Don't load model at startup
model = None

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris Prediction API is up.", "model_loaded": model is not None}

@app.post("/predict")
def get_prediction(input: IrisInput):
    global model
    if model is None:
        from app.model_utils import load_model
        print("Loading model on first request...")
        model = load_model()
        print("Model loaded successfully")
    
    from app.model_utils import predict
    features = [input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]
    pred_class = predict(model, features)
    return {"prediction": pred_class}
