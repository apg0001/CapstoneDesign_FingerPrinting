from fastapi import FastAPI
from pydantic import BaseModel
from .predict import load_latest_resources, predict_location

app = FastAPI()

class WifiData(BaseModel):
    data: dict  # {mac: rssi}

# 모델 & 인코더 로딩
model, location_encoder, mac_encoder = load_latest_resources()

@app.post("/predict")
def predict(wifi_data: WifiData):
    result = predict_location(wifi_data.data, model, location_encoder, mac_encoder)
    return {"predicted_location": result}