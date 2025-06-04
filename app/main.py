from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict
import logging
import traceback
from .predict import Predictor
# detect_new_macs 함수 별도로 구현 필요
from .online_trainer import OnlineTrainer, detect_new_macs

app = FastAPI()

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
# ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250423_181508.pkl"
# NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250423_181508.pkl"
# CONFIG_PATH = "./finger_printing/config/hyperparameters_20250423_215818.yaml"
# MODEL_PATH = "./app/checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
# ENCODER_PATH = "./app/checkpoints/encoders_20250423_181508.pkl"
# NORM_PATH = "./app/checkpoints/norm_20250423_181508.pkl"
# CONFIG_PATH = "./app/config/hyperparameters_20250423_181508.yaml"
# MODEL_PATH = "./checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
# ENCODER_PATH = "./checkpoints/encoders_20250423_181508.pkl"
# NORM_PATH = "./checkpoints/norm_20250423_181508.pkl"
# CONFIG_PATH = "./config/hyperparameters_20250423_181508.yaml"
MODEL_PATH = "./app/checkpoints/fp_model_CNNTransformer_20250520_172109.pt"
ENCODER_PATH = "./app/checkpoints/encoders_20250520_172109.pkl"
NORM_PATH = "./app/checkpoints/norm_20250520_172109.pkl"
CONFIG_PATH = "./app/config/hyperparameters_20250520_172109.yaml"

predictor = Predictor(MODEL_PATH, ENCODER_PATH, NORM_PATH, CONFIG_PATH)

online_trainer = OnlineTrainer(
    model=predictor.model,
    mac_encoder=predictor.mac_encoder,
    location_encoder=predictor.location_encoder,
    rssi_mean=predictor.rssi_mean,
    rssi_std=predictor.rssi_std,
    max_ap=predictor.config["num_ap"]
)


class InputData(BaseModel):
    mac_rssi: Dict[str, int]


def detect_new_macs(input_mac_rssi, mac_encoder):
    new_macs = []
    for mac in input_mac_rssi.keys():
        try:
            mac_encoder.transform([mac])
        except ValueError:
            new_macs.append(mac)
    return new_macs


def background_online_training(input_mac_rssi):
    new_macs = detect_new_macs(input_mac_rssi, online_trainer.mac_encoder)
    known_mac_rssi = {mac: rssi for mac,
                      rssi in input_mac_rssi.items() if mac not in new_macs}
    pred_location, _ = predictor.predict(known_mac_rssi)
    label = online_trainer.location_encoder.transform([pred_location])[0]
    online_trainer.add_new_sample(input_mac_rssi, label)
    online_trainer.train_incrementally(epochs=1, batch_size=1)
    online_trainer.save_model_and_encoders()


@app.post("/predict")
async def predict_api(input_data: InputData, background_tasks: BackgroundTasks):
    try:
        location, _ = predictor.predict(input_data.mac_rssi)
        # background_tasks.add_task(
        #     background_online_training, input_data.mac_rssi)
        return {"status_code": 200, "message": "Prediction Success!", "predicted_location": str(location)}
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test")
async def predict_api(input_data: InputData):
    logger.info(f"{input_data.mac_rssi}")
    first_mac = list(input_data.mac_rssi.keys())[0]
    first_rssi = input_data.mac_rssi[first_mac]
    return {"input_data": f"first_mac: {first_mac}, first_rssi: {first_rssi}"}