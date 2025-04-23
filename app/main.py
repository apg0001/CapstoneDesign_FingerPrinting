import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from .predict import predict_rssi

app = FastAPI()


class WifiData(BaseModel):
    input: Dict[str, int]  # {mac: rssi}
    
class InputData(BaseModel):
    mac_rssi: Dict[str, int]  # MAC 주소를 키로 하고, RSSI 값을 값으로 갖는 딕셔너리


# 로그 설정 (파일로 저장하거나 콘솔에 출력)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/predict")
async def predict(input_data: InputData):
    mac_rssi_data = input_data.mac_rssi
    print(mac_rssi_data)
    
    # 예시: 모델 경로, 인코더 경로, 정규화 파라미터 경로 및 예측을 위한 데이터
    model_path = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
    encoder_path = "./finger_printing/checkpoints/encoders/encoders_20250423_181508.pkl"
    norm_path = "./finger_printing/checkpoints/norm/norm_20250423_181508.pkl"
    config_path = "./finger_printing/config/hyperparameters_20250423_215818.yaml"

    try:
        # 예측 수행
        predicted_locations = predict_rssi(
            model_path, encoder_path, norm_path, mac_rssi_data, config_path
        )
        return {"predicted_location": str(predicted_locations)}

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"File not found: {str(e)}")

    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())  # 오류가 발생한 스택 트레이스를 출력
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the request.")


@app.post("/test")
async def predict(input_data: InputData):
    # 요청 받은 데이터 확인
    mac_rssi_data = input_data.mac_rssi
    print(mac_rssi_data)
    return {"message": "Data received", "data": mac_rssi_data}