from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Dict, Optional
import logging
import traceback
from .predict_one_input import Predictor
# detect_new_macs 함수 별도로 구현 필요
from .online_trainer import OnlineTrainer, detect_new_macs

app = FastAPI()

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 5층 데이터만
# MODEL_PATH = "./app/checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
# ENCODER_PATH = "./app/checkpoints/encoders_20250423_181508.pkl"
# NORM_PATH = "./app/checkpoints/norm_20250423_181508.pkl"
# CONFIG_PATH = "./app/config/hyperparameters_20250423_181508.yaml"

# 6층 데이터 포함
# MODEL_PATH = "./app/checkpoints/fp_model_CNNTransformer_20250520_172109.pt"
# ENCODER_PATH = "./app/checkpoints/encoders_20250520_172109.pkl"
# NORM_PATH = "./app/checkpoints/norm_20250520_172109.pkl"
# CONFIG_PATH = "./app/config/hyperparameters_20250520_172109.yaml"

# # 6층 데이터 파인튜닝 후 증강
# MODEL_PATH = "./app/checkpoints/fp_model_CNNTransformer_20250608_143837.pt"
# ENCODER_PATH = "./app/checkpoints/encoders_20250608_143837.pkl"
# NORM_PATH = "./app/checkpoints/norm_20250608_143837.pkl"
# CONFIG_PATH = "./app/config/hyperparameters_20250608_143837.yaml"

# 데이터 증강 후
# MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250605_194159.pt"
# ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250605_194159.pkl"
# NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250605_194159.pkl"
# CONFIG_PATH = "./finger_printing/config/hyperparameters_20250605_194159.yaml"

# 일부 데이터 제거 후 증강
MODEL_PATH = "./app/checkpoints/fp_model_CNNTransformer_20250613_213544.pt"
ENCODER_PATH = "./app/checkpoints/encoders_20250613_213544.pkl"
NORM_PATH = "./app/checkpoints/norm_20250613_213544.pkl"
CONFIG_PATH = "./app/config/hyperparameters_20250613_213544.yaml"


predictor = Predictor(MODEL_PATH, ENCODER_PATH, NORM_PATH, CONFIG_PATH)

online_trainer = OnlineTrainer(
    model=predictor.model,
    mac_encoder=predictor.mac_encoder,
    location_encoder=predictor.location_encoder,
    rssi_mean=predictor.rssi_mean,
    rssi_std=predictor.rssi_std,
    max_ap=predictor.config["num_ap"]
)


# class InputData(BaseModel):
#     mac_rssi: Optional[Dict[str, int]] = None
#     mac_rssi2: Optional[Dict[str, int]] = None
#     mac_rssi3: Optional[Dict[str, int]] = None

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
async def predict_api(input_data: InputData, background_tasks: BackgroundTasks, request: Request):
    # print(request.body)
    # print(input_data)
    # logger.info(input_data)
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
    logger.info(f"{input_data.mac_rssi1}")
    first_mac = list(input_data.mac_rssi1.keys())[0]
    first_rssi = input_data.mac_rssi1[first_mac]
    return {"input_data": f"first_mac: {first_mac}, first_rssi: {first_rssi}"}