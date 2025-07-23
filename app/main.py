from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import load_model, predict

app = FastAPI(title="Iris Model API")

model = load_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris Prediction API is up."}

@app.post("/predict")
def get_prediction(input: IrisInput):
    features = [
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width,
    ]
    pred_class = predict(model, features)
    return {"prediction": pred_class}
