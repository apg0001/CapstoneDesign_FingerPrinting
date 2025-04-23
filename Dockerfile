# 베이스 이미지로 Python 3.9 사용
FROM python:3.8.16

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치를 위한 requirements.txt 파일 복사
COPY requirements.txt .

# 필요한 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 앱 소스 코드 복사
COPY . .

# 컨테이너가 시작할 때 실행할 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]