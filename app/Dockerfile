# 베이스 이미지로 Python 3.8 사용
FROM python:3.8.16

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# 비특권 사용자 생성
RUN useradd -m -u 1000 appuser

# 의존성 설치를 위한 requirements.txt 파일 복사
COPY requirements.txt .

# 필요한 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 앱 소스 코드 복사
COPY . /app

# 권한 설정
RUN chown -R appuser:appuser /app

# 비특권 사용자로 전환
USER appuser

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 컨테이너가 시작할 때 실행할 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]