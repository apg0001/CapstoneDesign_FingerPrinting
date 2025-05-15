from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
import traceback
import numpy as np
from .predict import Predictor

app = FastAPI()


class InputData(BaseModel):
    mac_rssi: Dict[str, int]  # MAC 주소를 키로 하고, RSSI 값을 값으로 갖는 딕셔너리


# 로그 설정 (파일로 저장하거나 콘솔에 출력)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 모델 경로 설정
MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250423_181508.pkl"
NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250423_181508.pkl"
CONFIG_PATH = "./finger_printing/config/hyperparameters_20250423_215818.yaml"

# Predictor 객체 생성 (서버 시작시 1회만 로드)
predictor = Predictor(MODEL_PATH, ENCODER_PATH, NORM_PATH, CONFIG_PATH)


@app.post("/predict")
async def predict_api(input_data: InputData):
    try:
        location, _ = predictor.predict(input_data.mac_rssi)
        return {"status_code": 200, "message": "Prediction Success!", "predicted_location": str(location)}

    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())  # 오류가 발생한 스택 트레이스를 출력
        raise HTTPException(
            status_code=500, message=f"An error occurred while processing the request.: {e}")


@app.post("/test")
async def predict(input_data: InputData):
    # 요청 받은 데이터 확인
    mac_rssi_data = input_data.mac_rssi
    print(mac_rssi_data)
    return {"message": "Data received", "data": mac_rssi_data}
