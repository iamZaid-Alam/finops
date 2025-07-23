import joblib
import numpy as np

MODEL_PATH = "app/iris_model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def predict(model, features: list):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)
    return int(prediction[0])
